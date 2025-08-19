# server.py — Live Sleeper draft (normalize URL, soft-stop if feed empty), no CSV/pandas
import os, time, asyncio, random, re, math
from typing import Any, Dict, List, Optional, Set, Tuple
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import httpx

API_KEY = os.getenv("API_KEY")   # optional in dev; if set, require x-api-key header
SLEEPER = "https://api.sleeper.app/v1"

app = FastAPI(title="Fantasy Live Draft API (soft-stop; no CSV)")

# ===== Players cache =====
PLAYERS_CACHE: Dict[str, Dict[str, Any]] = {}
PLAYERS_LOADED_AT: float = 0.0

def auth_or_401(x_api_key: Optional[str]):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

# ===== HTTP helpers =====
async def _http_get(url: str, headers: Optional[Dict[str, str]] = None,
                    max_attempts: int = 3, base_backoff: float = 0.25,
                    timeout: float = 45.0) -> httpx.Response:
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
            await asyncio.sleep(base_backoff * (2 ** i) + random.random() * 0.2)
    raise HTTPException(status_code=502, detail=f"Upstream error fetching {url}: {last_err}")

async def _get_json(url: str) -> Any:
    resp = await _http_get(url)
    try:
        return resp.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Couldn't parse JSON at {url}: {e}")

# ===== Normalize draft URL =====
_DRAFT_ID_RE = re.compile(r"(?P<id>\d{16,20})")

def normalize_draft_picks_url(draft_url_or_page: str) -> Tuple[str, str]:
    """
    Accepts Sleeper web or API URL and returns (draft_id, api_picks_url).
    Examples:
      - https://sleeper.com/draft/nfl/{id}
      - https://sleeper.app/draft/nfl/{id}
      - https://api.sleeper.app/v1/draft/{id}/picks
    """
    m = _DRAFT_ID_RE.search(draft_url_or_page)
    if not m:
        raise HTTPException(status_code=400, detail="Could not extract draft_id from provided URL.")
    draft_id = m.group("id")
    return draft_id, f"{SLEEPER}/draft/{draft_id}/picks"

# ===== Players loading =====
async def load_players_if_needed() -> None:
    global PLAYERS_CACHE, PLAYERS_LOADED_AT
    if PLAYERS_CACHE and (time.time() - PLAYERS_LOADED_AT) < 3 * 3600:
        return
    data = await _get_json(f"{SLEEPER}/players/nfl")
    PLAYERS_CACHE = {}
    for pid, p in data.items():
        name = p.get("full_name") or f"{p.get('first_name','').strip()} {p.get('last_name','').strip()}".strip() or pid
        PLAYERS_CACHE[pid] = {
            "id": pid,
            "name": name,
            "team": p.get("team"),
            "pos": p.get("position"),
            "bye": p.get("bye_week"),
            # Optional metrics (may be None in Sleeper)
            "adp": p.get("adp"),
            "proj_ros": p.get("fantasy_points_half_ppr"),
            "proj_week": p.get("fantasy_points"),
        }
    PLAYERS_LOADED_AT = time.time()

# ===== Draft utils =====
def parse_picks(picks_json: Any) -> List[Dict[str, Any]]:
    if isinstance(picks_json, list):
        return picks_json
    if isinstance(picks_json, dict) and isinstance(picks_json.get("picks"), list):
        return picks_json["picks"]
    return []

def parse_drafted_from_picks(picks: List[Dict[str, Any]]) -> Tuple[Set[str], Dict[int, Set[str]]]:
    drafted: Set[str] = set()
    by_roster: Dict[int, Set[str]] = {}
    for p in picks:
        pid = p.get("player_id")
        rid = p.get("roster_id")
        if pid:
            drafted.add(str(pid))
            if isinstance(rid, int):
                by_roster.setdefault(rid, set()).add(str(pid))
    return drafted, by_roster

async def draft_state(draft_id: str, picks: List[Dict[str, Any]]) -> Dict[str, Any]:
    meta = await _get_json(f"{SLEEPER}/draft/{draft_id}")
    total_teams = int(meta.get("metadata", {}).get("teams") or meta.get("settings", {}).get("teams") or 0)
    if not total_teams:
        roster_ids = {p.get("roster_id") for p in picks if isinstance(p.get("roster_id"), int)}
        total_teams = len(roster_ids) or 12
    made = len([p for p in picks if p.get("player_id")])
    on_index = made + 1
    cur_round = max(1, math.ceil(on_index / total_teams))
    pick_in_round = ((on_index - 1) % total_teams) + 1
    snake_rev = (cur_round % 2 == 0)
    order_index = pick_in_round if not snake_rev else (total_teams - pick_in_round + 1)
    unique_rosters = sorted({p.get("roster_id") for p in picks if isinstance(p.get("roster_id"), int)})
    roster_on_clock = unique_rosters[order_index - 1] if 1 <= order_index <= len(unique_rosters) else None
    return {
        "draft_id": draft_id,
        "total_teams": total_teams,
        "picks_made": made,
        "current_round": cur_round,
        "pick_in_round": pick_in_round,
        "snake_reversed": snake_rev,
        "team_number_on_clock": order_index,
        "roster_id_on_clock": roster_on_clock
    }

# ===== Scoring =====
NEED_WEIGHTS = {"QB":1.1, "RB":1.45, "WR":1.45, "TE":1.25, "DST":0.5, "K":0.4}

def roster_cap(pos: str, slots: Dict[str, int]) -> int:
    if pos in {"RB","WR"}:
        return slots.get(pos, 0) + slots.get("FLEX", 0) + 2
    if pos in {"TE","QB"}:
        return slots.get(pos, 0) + 1
    return slots.get(pos, 0) + 1

def compute_needs(my_ids: List[str], slots: Dict[str, int]) -> Dict[str, float]:
    counts = {"QB":0,"RB":0,"WR":0,"TE":0,"DST":0,"K":0}
    for pid in my_ids:
        pos = PLAYERS_CACHE.get(pid, {}).get("pos")
        if pos in counts:
            counts[pos] += 1
    needs: Dict[str, float] = {}
    for pos, base in NEED_WEIGHTS.items():
        gap = max(0, slots.get(pos, 0) - counts.get(pos, 0))
        needs[pos] = base if gap > 0 else base * 0.35
    return needs

def score_available(available: List[Dict[str, Any]],
                    my_ids: List[str],
                    slots: Dict[str, int],
                    pick_number: int) -> List[Dict[str, Any]]:
    needs = compute_needs(my_ids, slots)
    pos_counts: Dict[str, int] = {}
    for pid in my_ids:
        pos = PLAYERS_CACHE.get(pid, {}).get("pos")
        if pos:
            pos_counts[pos] = pos_counts.get(pos, 0) + 1
    adps = [p.get("adp") for p in available if isinstance(p.get("adp"), (int, float))]
    adp_median = sorted(adps)[len(adps)//2] if adps else None

    scored: List[Dict[str, Any]] = []
    for p in available:
        pos = p["pos"]
        proj = float(p.get("proj_ros") or p.get("proj_week") or 0.0)
        adp = p.get("adp")
        adp_discount = (adp_median - adp) if (adp_median is not None and isinstance(adp, (int, float))) else 0.0
        need_boost = needs.get(pos, 1.0)
        cur = pos_counts.get(pos, 0)
        cap = roster_cap(pos, slots)
        overdrafted = max(0, cur - cap)
        cap_penalty = 0.85 ** overdrafted

        score = proj + 0.15 * adp_discount
        score *= (0.85 + 0.30 * need_boost)
        score *= cap_penalty

        scored.append({
            "id": p["id"], "name": p.get("name"), "team": p.get("team"),
            "pos": pos, "bye": p.get("bye"),
            "adp": p.get("adp"),
            "proj_ros": p.get("proj_ros"), "proj_week": p.get("proj_week"),
            "score": round(score, 3),
            "explain": f"{pos} need {need_boost:.2f}, cap {cur}/{cap}, pick {pick_number}"
        })
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored

# ===== Request model =====
class RecommendReq(BaseModel):
    draft_url: str           # Sleeper web or API URL
    roster_id: int
    pick_number: int
    season: int = 2025
    roster_slots: Optional[Dict[str, int]] = None
    limit: int = 10

# ===== Lifecycle =====
@app.on_event("startup")
async def startup():
    await load_players_if_needed()

@app.get("/health")
def health():
    return {"ok": True, "players_cached": len(PLAYERS_CACHE), "ts": int(time.time())}

# Helper (debug mapping): team number -> roster_id and live roster view
@app.get("/draft/{draft_id}/team/{team_number}")
async def team_debug(draft_id: str, team_number: int, x_api_key: Optional[str] = Header(None)):
    auth_or_401(x_api_key)
    await load_players_if_needed()
    picks_json = await _get_json(f"{SLEEPER}/draft/{draft_id}/picks")
    picks = parse_picks(picks_json)
    unique_rosters = sorted({p.get("roster_id") for p in picks if isinstance(p.get("roster_id"), int)})
    if team_number < 1 or (unique_rosters and team_number > len(unique_rosters)):
        raise HTTPException(status_code=400, detail="Invalid team number for this draft.")
    roster_id = unique_rosters[team_number - 1] if unique_rosters else None
    my_picks = [p for p in picks if roster_id and p.get("roster_id") == roster_id and p.get("player_id")]
    roster = []
    for p in my_picks:
        pid = str(p["player_id"])
        meta = PLAYERS_CACHE.get(pid, {})
        roster.append({"id": pid, "name": meta.get("name"), "pos": meta.get("pos"), "team": meta.get("team")})
    state = await draft_state(draft_id, picks) if picks else {"note": "no picks yet"}
    return {"team_number": team_number, "roster_id": roster_id, "roster": roster, "state": state}

# ===== Main recommender (soft-stop if feed empty) =====
@app.post("/recommend_live")
async def recommend_live(body: RecommendReq, x_api_key: Optional[str] = Header(None)):
    auth_or_401(x_api_key)
    await load_players_if_needed()

    draft_id, picks_url = normalize_draft_picks_url(body.draft_url)
    picks_json = await _get_json(picks_url)
    picks = parse_picks(picks_json)

    # If the user is asking for draft help and the feed is empty, DO NOT proceed.
    if not picks:
        return {
            "status": "waiting_for_live_feed",
            "message": "Live picks feed is empty/invalid. Not proceeding with recommendations.",
            "draft_id": draft_id,
            "picks_url": picks_url,
            "recommended": [],
            "alternatives": [],
            "my_team": [],
            "drafted_count": 0,
            "my_team_count": 0,
            "ts": int(time.time())
        }

    state = await draft_state(draft_id, picks)
    drafted_ids, by_roster = parse_drafted_from_picks(picks)
    my_ids = list(by_roster.get(body.roster_id, set()))

    # Available pool
    available: List[Dict[str, Any]] = []
    for pid, pdata in PLAYERS_CACHE.items():
        if not pdata.get("pos"): continue
        if pid in drafted_ids: continue
        if pid in my_ids: continue
        available.append(pdata)

    if not available:
        return {
            "status": "no_available_after_filtering",
            "message": "No available players after filtering drafted and your roster.",
            "draft_state": state,
            "recommended": [],
            "alternatives": [],
            "my_team": [PLAYERS_CACHE[pid] for pid in my_ids if pid in PLAYERS_CACHE],
            "drafted_count": len(drafted_ids),
            "my_team_count": len(my_ids),
            "ts": int(time.time())
        }

    slots = body.roster_slots or {"QB":1,"RB":2,"WR":2,"TE":1,"FLEX":2,"DST":1,"K":1}
    ranked = score_available(available, my_ids, slots, body.pick_number)
    limit = max(3, body.limit)
    topn = ranked[:limit]

    return {
        "status": "ok",
        "pick": body.pick_number,
        "season_used": body.season,
        "recommended": topn[:3],
        "alternatives": topn[3:limit],
        "my_team": [PLAYERS_CACHE[pid] for pid in my_ids if pid in PLAYERS_CACHE],
        "draft_state": state,
        "drafted_count": len(drafted_ids),
        "my_team_count": len(my_ids),
        "ts": int(time.time())
    }
