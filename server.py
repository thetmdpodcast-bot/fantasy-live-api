# server.py — clean reset: live Sleeper draft feed, best single pick, strict errors
import os, time, asyncio, random
from typing import Any, Dict, List, Optional, Set, Tuple
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import httpx

API_KEY = os.getenv("API_KEY")  # optional in dev
SLEEPER = "https://api.sleeper.app/v1"

app = FastAPI(title="Fantasy Live Draft API (reset)")

# ===== Players cache =====
PLAYERS_CACHE: Dict[str, Dict[str, Any]] = {}
PLAYERS_LOADED_AT: float = 0.0

def auth_or_401(x_api_key: Optional[str]):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

async def http_get(url: str, headers: Optional[Dict[str,str]]=None,
                   max_attempts: int=3, base: float=0.25, timeout: float=45.0) -> httpx.Response:
    last_err = None
    for i in range(max_attempts):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.get(url, headers=headers)
                if r.status_code != 200:
                    r.raise_for_status()
                return r
        except Exception as e:
            last_err = e
            await asyncio.sleep(base*(2**i) + random.random()*0.2)
    raise HTTPException(status_code=502, detail=f"Upstream error fetching {url}: {last_err}")

async def get_json(url: str) -> Any:
    r = await http_get(url)
    try:
        return r.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Couldn't parse JSON at {url}: {e}")

async def load_players_if_needed() -> None:
    global PLAYERS_CACHE, PLAYERS_LOADED_AT
    if PLAYERS_CACHE and (time.time() - PLAYERS_LOADED_AT) < 3*3600:
        return
    data = await get_json(f"{SLEEPER}/players/nfl")
    PLAYERS_CACHE = {}
    for pid, p in data.items():
        name = p.get("full_name") or f"{p.get('first_name','')} {p.get('last_name','')}".strip() or pid
        PLAYERS_CACHE[pid] = {
            "id": pid,
            "name": name,
            "team": p.get("team"),
            "pos": p.get("position"),
            "bye": p.get("bye_week"),
            # optional metrics (may be None)
            "adp": p.get("adp"),
            "proj_ros": p.get("fantasy_points_half_ppr"),
            "proj_week": p.get("fantasy_points"),
        }
    PLAYERS_LOADED_AT = time.time()

def parse_drafted_from_picks(picks_json: Any) -> Tuple[Set[str], Dict[int, Set[str]]]:
    drafted: Set[str] = set()
    by_roster: Dict[int, Set[str]] = {}
    picks = picks_json if isinstance(picks_json, list) else picks_json.get("picks", [])
    for p in picks:
        pid = p.get("player_id")
        rid = p.get("roster_id")
        if pid:
            drafted.add(str(pid))
            if isinstance(rid, int):
                by_roster.setdefault(rid, set()).add(str(pid))
    return drafted, by_roster

# simple, robust scoring
NEED_WEIGHTS = {"QB":1.1, "RB":1.45, "WR":1.45, "TE":1.25, "DST":0.5, "K":0.4}
def roster_cap(pos: str, slots: Dict[str,int]) -> int:
    if pos in {"RB","WR"}: return slots.get(pos,0)+slots.get("FLEX",0)+2
    if pos in {"TE","QB"}: return slots.get(pos,0)+1
    return slots.get(pos,0)+1

def compute_needs(my_ids: List[str], slots: Dict[str,int]) -> Dict[str,float]:
    counts = {k:0 for k in ["QB","RB","WR","TE","DST","K"]}
    for pid in my_ids:
        pos = PLAYERS_CACHE.get(pid,{}).get("pos")
        if pos in counts: counts[pos]+=1
    needs = {}
    for pos, base in NEED_WEIGHTS.items():
        gap = max(0, slots.get(pos,0)-counts.get(pos,0))
        needs[pos] = base if gap>0 else base*0.35
    return needs

def score_available(available: List[Dict[str,Any]], my_ids: List[str], slots: Dict[str,int], pick_number:int):
    needs = compute_needs(my_ids, slots)
    pos_counts: Dict[str,int] = {}
    for pid in my_ids:
        pos = PLAYERS_CACHE.get(pid,{}).get("pos")
        if pos: pos_counts[pos] = pos_counts.get(pos,0)+1
    adps = [p.get("adp") for p in available if isinstance(p.get("adp"), (int,float))]
    adp_median = sorted(adps)[len(adps)//2] if adps else None

    scored = []
    for p in available:
        pos = p["pos"]
        proj = float(p.get("proj_ros") or p.get("proj_week") or 0.0)
        adp = p.get("adp")
        adp_discount = (adp_median - adp) if (adp_median is not None and isinstance(adp,(int,float))) else 0.0
        need_boost = needs.get(pos,1.0)
        cur = pos_counts.get(pos,0)
        cap = roster_cap(pos, slots)
        overdrafted = max(0, cur-cap)
        cap_penalty = 0.85 ** overdrafted
        score = proj + 0.15*adp_discount
        score *= (0.85 + 0.30*need_boost)
        score *= cap_penalty
        scored.append({
            "id": p["id"], "name": p.get("name"), "team": p.get("team"),
            "pos": pos, "bye": p.get("bye"),
            "adp": p.get("adp"),
            "proj_ros": p.get("proj_ros"), "proj_week": p.get("proj_week"),
            "score": round(score,3),
            "explain": f"{pos} need {need_boost:.2f}, cap {cur}/{cap}, pick {pick_number}"
        })
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored

class RecommendReq(BaseModel):
    draft_url: str      # https://api.sleeper.app/v1/draft/<DRAFT_ID>/picks
    roster_id: int      # team slot id from Sleeper picks
    pick_number: int
    roster_slots: Optional[Dict[str,int]] = None

@app.on_event("startup")
async def startup():
    await load_players_if_needed()

@app.get("/health")
def health():
    return {"ok": True, "players_cached": len(PLAYERS_CACHE), "ts": int(time.time())}

@app.get("/draft/{draft_id}/team/{team_number}")
async def team_view(draft_id: str, team_number: int, x_api_key: Optional[str] = Header(None)):
    auth_or_401(x_api_key)
    await load_players_if_needed()
    picks = await get_json(f"{SLEEPER}/draft/{draft_id}/picks")
    if not isinstance(picks, list):
        raise HTTPException(status_code=502, detail="Unexpected picks payload")
    roster_ids = sorted({p["roster_id"] for p in picks if isinstance(p.get("roster_id"), int)})
    if team_number < 1 or team_number > len(roster_ids):
        raise HTTPException(status_code=400, detail="Invalid team number")
    rid = roster_ids[team_number-1]
    team_picks = [p for p in picks if p.get("roster_id")==rid and p.get("player_id")]
    roster = []
    for p in team_picks:
        pid = str(p["player_id"])
        meta = PLAYERS_CACHE.get(pid, {})
        roster.append({
            "id": pid,
            "name": meta.get("name") or pid,
            "pos": meta.get("pos"),
            "team": meta.get("team"),
        })
    return {"team_number": team_number, "roster_id": rid, "roster": roster, "total_picks": len(team_picks)}

@app.post("/recommend_live")
async def recommend_live(body: RecommendReq, x_api_key: Optional[str] = Header(None)):
    auth_or_401(x_api_key)
    await load_players_if_needed()

    # 1) Live draft fetch (STRICT: error if this fails)
    try:
        draft_json = await get_json(body.draft_url)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Draft fetch failed: {str(e)}")

    # 2) Parse drafted + my roster
    drafted_ids, by_roster = parse_drafted_from_picks(draft_json)
    my_ids = list(by_roster.get(body.roster_id, set()))
    slots = body.roster_slots or {"QB":1,"RB":2,"WR":2,"TE":1,"FLEX":2,"DST":1,"K":1}

    # 3) Available pool
    available: List[Dict[str,Any]] = []
    for pid, pdata in PLAYERS_CACHE.items():
        if not pdata.get("pos"): continue
        if pid in drafted_ids: continue
        if pid in my_ids: continue
        available.append(pdata)

    if not available:
        raise HTTPException(status_code=409, detail="No available players after filtering (live draft pool empty)")

    # 4) Score and pick best single option
    ranked = score_available(available, my_ids, slots, body.pick_number)
    if not ranked:
        raise HTTPException(status_code=500, detail="Scoring produced no results")
    best = ranked[0]

    return {
        "pick": body.pick_number,
        "best": best,
        "explain": best["explain"],
        "my_team_count": len(my_ids),
        "drafted_count": len(drafted_ids),
        "ts": int(time.time())
    }
