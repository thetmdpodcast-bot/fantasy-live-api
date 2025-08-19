# server.py — Draft-first workflow, strict fail-safe, no CSV/pandas dependency
import os, time, asyncio, random, json
from typing import List, Dict, Optional, Set, Tuple, Any
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import httpx

API_KEY = os.getenv("API_KEY")
SLEEPER = "https://api.sleeper.app/v1"

app = FastAPI(title="Fantasy Live Draft API (no-csv)")

# ===== CACHES =====
PLAYERS_CACHE: Dict[str, Dict] = {}
PLAYERS_LOADED_AT: float = 0

# ===== AUTH =====
def auth_or_401(x_api_key: Optional[str]):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

# ===== HTTP HELPERS =====
async def robust_http_get(url: str, headers: Optional[Dict[str,str]]=None,
                          max_attempts: int=3, base: float=0.3, timeout: float=60.0) -> httpx.Response:
    last_err = None
    for i in range(max_attempts):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.get(url, headers=headers)
                if resp.status_code != 200:
                    resp.raise_for_status()
                return resp
        except Exception as e:
            last_err = e
            await asyncio.sleep(base * (2**i) + random.random()*0.25)
    raise HTTPException(status_code=502, detail=f"Upstream error fetching {url}: {last_err}")

async def get_json(url: str) -> Any:
    resp = await robust_http_get(url, timeout=60.0)
    try:
        return resp.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Couldn't parse JSON at {url}: {e}")

# ===== PLAYERS CACHE =====
async def load_players_if_needed() -> None:
    global PLAYERS_CACHE, PLAYERS_LOADED_AT
    if PLAYERS_CACHE and (time.time() - PLAYERS_LOADED_AT) < 3*3600:
        return
    resp = await robust_http_get(f"{SLEEPER}/players/nfl", timeout=90.0)
    data = resp.json()
    PLAYERS_CACHE = {}
    for pid, p in data.items():
        name = p.get("full_name") or f"{p.get('first_name','')} {p.get('last_name','')}".strip()
        PLAYERS_CACHE[pid] = {
            "id": pid,
            "name": name,
            "team": p.get("team"),
            "pos": p.get("position"),
            "bye": p.get("bye_week"),
            # Optional metrics from Sleeper (may be None)
            "adp": p.get("adp"),
            "proj_ros": p.get("fantasy_points_half_ppr"),
            "proj_week": p.get("fantasy_points"),
        }
    PLAYERS_LOADED_AT = time.time()

# ===== DRAFT PARSE =====
def parse_drafted_from_json(draft_json: Any) -> Tuple[Set[str], Dict[int, Set[str]]]:
    drafted: Set[str] = set()
    by_roster: Dict[int, Set[str]] = {}
    picks = draft_json if isinstance(draft_json, list) else draft_json.get("picks", [])
    for p in picks:
        pid = p.get("player_id")
        rid = p.get("roster_id")
        if pid:
            drafted.add(str(pid))
            if isinstance(rid, int):
                by_roster.setdefault(rid, set()).add(str(pid))
    return drafted, by_roster

# ===== SCORING =====
NEED_WEIGHTS = {"QB":1.1, "RB":1.45, "WR":1.45, "TE":1.25, "DST":0.5, "K":0.4}

def roster_sane_cap(pos: str, roster_slots: Dict[str,int]) -> int:
    if pos in {"RB","WR"}:
        return roster_slots.get(pos,0) + roster_slots.get("FLEX",0) + 2
    if pos in {"TE","QB"}:
        return roster_slots.get(pos,0) + 1
    return roster_slots.get(pos,0) + 1

def compute_needs(my_ids: List[str], roster_slots: Dict[str,int]) -> Dict[str,float]:
    counts = {"QB":0,"RB":0,"WR":0,"TE":0,"DST":0,"K":0}
    for pid in my_ids:
        pos = PLAYERS_CACHE.get(pid,{}).get("pos")
        if pos in counts: counts[pos]+=1
    needs = {}
    for pos, base in NEED_WEIGHTS.items():
        gap = max(0, roster_slots.get(pos,0) - counts.get(pos,0))
        needs[pos] = base * (1.0 if gap>0 else 0.35)
    return needs

def score_players(available: List[Dict], my_ids: List[str], roster_slots: Dict[str,int], pick_number:int):
    needs = compute_needs(my_ids, roster_slots)
    pos_counts: Dict[str,int] = {}
    for pid in my_ids:
        pos = PLAYERS_CACHE.get(pid,{}).get("pos")
        if pos: pos_counts[pos] = pos_counts.get(pos,0)+1
    # adp median (optional)
    adps = [p.get("adp") for p in available if isinstance(p.get("adp"), (int,float))]
    adp_median = sorted(adps)[len(adps)//2] if adps else None

    scored = []
    for p in available:
        pos = p["pos"]
        proj = float(p.get("proj_ros") or p.get("proj_week") or 0.0)
        adp = p.get("adp")
        adp_discount = (adp_median - adp) if (adp_median and isinstance(adp,(int,float))) else 0.0
        need_boost = needs.get(pos,1.0)
        cur = pos_counts.get(pos,0)
        cap = roster_sane_cap(pos, roster_slots)
        overdrafted = max(0, cur-cap)
        cap_penalty = 0.85 ** overdrafted
        score = proj + 0.15*adp_discount
        score *= (0.85 + 0.30*need_boost)
        score *= cap_penalty
        scored.append({
            "id": p["id"], "name": p.get("name"), "team": p.get("team"),
            "pos": pos, "bye": p.get("bye"),
            "proj_ros": p.get("proj_ros"), "proj_week": p.get("proj_week"),
            "adp": p.get("adp"),
            "score": round(score,3),
            "explain": f"{pos} need {need_boost:.2f}, cap {cur}/{cap}, pick {pick_number}"
        })
    return sorted(scored, key=lambda x: x["score"], reverse=True)

# ===== MODELS =====
class RecommendReq(BaseModel):
    draft_url: str
    pick_number: int
    roster_id: int
    season: int = 2025
    roster_slots: Optional[Dict[str,int]] = None
    limit: int = 10

# ===== LIFECYCLE =====
@app.on_event("startup")
async def warm():
    await load_players_if_needed()

@app.get("/health")
def health():
    return {"ok": True, "ts": int(time.time())}

# ===== DRAFT TEAM VIEW (for debugging roster mapping) =====
@app.get("/draft/{draft_id}/team/{team_number}")
async def get_team_roster(draft_id: str, team_number: int, x_api_key: Optional[str] = Header(None)):
    auth_or_401(x_api_key)
    await load_players_if_needed()

    # 1) Fetch full draft object (metadata not strictly needed)
    # 2) Fetch picks, strict fail-safe
    picks = await get_json(f"{SLEEPER}/draft/{draft_id}/picks")
    if not isinstance(picks, list):
        raise HTTPException(status_code=502, detail="Unexpected picks payload")

    # Determine unique roster_ids in draft order and map team_number -> roster_id
    roster_ids = sorted({p["roster_id"] for p in picks if "roster_id" in p and isinstance(p["roster_id"], int)})
    if team_number < 1 or team_number > len(roster_ids):
        raise HTTPException(status_code=400, detail="Invalid team number")
    roster_id = roster_ids[team_number - 1]

    # Build live roster for that team
    team_picks = [p for p in picks if p.get("roster_id") == roster_id and p.get("player_id")]
    roster = []
    for p in team_picks:
        pid = str(p["player_id"])
        meta = PLAYERS_CACHE.get(pid, {})
        roster.append({
            "id": pid,
            "name": meta.get("name") or f"{p.get('metadata',{}).get('first_name','')} {p.get('metadata',{}).get('last_name','')}".strip(),
            "pos": meta.get("pos"),
            "team": meta.get("team"),
        })

    return {"team_number": team_number, "roster_id": roster_id, "roster": roster, "total_picks": len(team_picks)}

# ===== MAIN RECOMMENDER =====
@app.post("/recommend_live")
async def recommend_live(body: RecommendReq, x_api_key: Optional[str] = Header(None)):
    auth_or_401(x_api_key)
    await load_players_if_needed()

    # Strict fail-safe: if draft fetch fails, error out
    try:
        draft_json = await get_json(body.draft_url)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Draft fetch failed: {str(e)}")

    drafted_ids, by_roster = parse_drafted_from_json(draft_json)
    my_ids = list(by_roster.get(body.roster_id, set()))
    roster_slots = body.roster_slots or {"QB":1,"RB":2,"WR":2,"TE":1,"FLEX":2,"DST":1,"K":1}

    # Available = not drafted and not on my roster
    available = []
    for pid, pdata in PLAYERS_CACHE.items():
        if not pdata.get("pos"): continue
        if pid in drafted_ids: continue
        if pid in my_ids: continue
        available.append(pdata)

    if not available:
        raise HTTPException(status_code=500, detail="No available players after filtering")

    ranked = score_players(available, my_ids, roster_slots, body.pick_number)[:max(5, body.limit)]

    return {
        "pick": body.pick_number,
        "season_used": body.season,
        "recommended": ranked[:3],
        "alternatives": ranked[3:6],
        "my_team": [PLAYERS_CACHE[pid] for pid in my_ids if pid in PLAYERS_CACHE],
        "all_drafted": [PLAYERS_CACHE[pid] for pid in drafted_ids if pid in PLAYERS_CACHE],
        "drafted_count": len(drafted_ids),
        "my_team_count": len(my_ids),
        "ts": int(time.time())
    }
