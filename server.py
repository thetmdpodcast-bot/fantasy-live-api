# server.py — Live Sleeper draft feed, strict fail-safe, no CSV/pandas
# Usage:
#   POST /recommend_live  with JSON:
#     {
#       "draft_url": "https://api.sleeper.app/v1/draft/<DRAFT_ID>/picks",
#       "roster_id": 3,
#       "pick_number": 19,
#       "limit": 10,                       # optional (default 10)
#       "roster_slots": {                  # optional
#         "QB":1,"RB":2,"WR":2,"TE":1,"FLEX":2,"DST":1,"K":1
#       }
#     }
#
#   GET  /draft/<DRAFT_ID>/team/<team_number>
#     - Handy debug endpoint to see your live roster by team_number (1..N)
#
# Notes:
# - Strict fail-safe: if the live draft fetch fails, we raise 502 (no stale cache).
# - We filter: already drafted + your roster.
# - Scoring uses Sleeper's player payload (simple, safe heuristic).
# - No hard-coded must-picks; no background fallbacks.

import os
import time
import asyncio
import random
from typing import Any, Dict, List, Optional, Set, Tuple

import httpx
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

API_KEY = os.getenv("API_KEY")  # optional in dev
SLEEPER = "https://api.sleeper.app/v1"

app = FastAPI(title="Fantasy Live Draft API (live feed, strict failsafe)")

# ====== Players cache (from Sleeper /players/nfl) =====================================

PLAYERS_CACHE: Dict[str, Dict[str, Any]] = {}
PLAYERS_LOADED_AT: float = 0.0

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
            # jittered exponential backoff
            await asyncio.sleep(base_backoff * (2 ** i) + random.random() * 0.2)
    raise HTTPException(status_code=502, detail=f"Upstream error fetching {url}: {last_err}")

async def _get_json(url: str) -> Any:
    resp = await _http_get(url)
    try:
        return resp.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Couldn't parse JSON at {url}: {e}")

async def load_players_if_needed() -> None:
    """Warm/refresh Sleeper players cache every ~3 hours."""
    global PLAYERS_CACHE, PLAYERS_LOADED_AT
    if PLAYERS_CACHE and (time.time() - PLAYERS_LOADED_AT) < 3 * 3600:
        return
    data = await _get_json(f"{SLEEPER}/players/nfl")
    PLAYERS_CACHE = {}
    for pid, p in data.items():
        name = p.get("full_name")
        if not name:
            # Some entries only have first/last names
            name = f"{p.get('first_name','').strip()} {p.get('last_name','').strip()}".strip()
        PLAYERS_CACHE[pid] = {
            "id": pid,
            "name": name or pid,
            "team": p.get("team"),
            "pos": p.get("position"),
            "bye": p.get("bye_week"),
            # Optional metrics present in some feeds (may be None):
            "adp": p.get("adp"),
            "proj_ros": p.get("fantasy_points_half_ppr"),
            "proj_week": p.get("fantasy_points"),
        }
    PLAYERS_LOADED_AT = time.time()

# ====== Draft feed parsing ==============================================================

def parse_drafted_from_picks(picks_json: Any) -> Tuple[Set[str], Dict[int, Set[str]]]:
    """From Sleeper /draft/{id}/picks (list of picks), build:
       - drafted_ids: set of player_id strings
       - by_roster: roster_id -> set of player_id strings
    """
    drafted: Set[str] = set()
    by_roster: Dict[int, Set[str]] = {}
    if not isinstance(picks_json, list):
        # Some wrappers might return {"picks": [...]}; handle both
        picks = picks_json.get("picks", []) if isinstance(picks_json, dict) else []
    else:
        picks = picks_json

    for p in picks:
        pid = p.get("player_id")
        rid = p.get("roster_id")
        if pid:
            drafted.add(str(pid))
            if isinstance(rid, int):
                by_roster.setdefault(rid, set()).add(str(pid))
    return drafted, by_roster

# ====== Scoring ========================================================================

NEED_WEIGHTS = {"QB": 1.1, "RB": 1.45, "WR": 1.45, "TE": 1.25, "DST": 0.5, "K": 0.4}

def roster_cap(pos: str, slots: Dict[str, int]) -> int:
    # Modest caps so we don't spam same position:
    if pos in {"RB", "WR"}:
        return slots.get(pos, 0) + slots.get("FLEX", 0) + 2
    if pos in {"TE", "QB"}:
        return slots.get(pos, 0) + 1
    return slots.get(pos, 0) + 1

def compute_needs(my_ids: List[str], slots: Dict[str, int]) -> Dict[str, float]:
    counts = {k: 0 for k in ["QB", "RB", "WR", "TE", "DST", "K"]}
    for pid in my_ids:
        pos = PLAYERS_CACHE.get(pid, {}).get("pos")
        if pos in counts:
            counts[pos] += 1
    needs: Dict[str, float] = {}
    for pos, base in NEED_WEIGHTS.items():
        gap = max(0, slots.get(pos, 0) - counts.get(pos, 0))
        # If still under slot target → full weight; otherwise dampen
        needs[pos] = base if gap > 0 else base * 0.35
    return needs

def score_available(available: List[Dict[str, Any]],
                    my_ids: List[str],
                    slots: Dict[str, int],
                    pick_number: int) -> List[Dict[str, Any]]:
    needs = compute_needs(my_ids, slots)

    # Precompute position counts on my roster
    pos_counts: Dict[str, int] = {}
    for pid in my_ids:
        pos = PLAYERS_CACHE.get(pid, {}).get("pos")
        if pos:
            pos_counts[pos] = pos_counts.get(pos, 0) + 1

    # Median ADP for a small value tilt, if present
    adps = [p.get("adp") for p in available if isinstance(p.get("adp"), (int, float))]
    adp_median = sorted(adps)[len(adps) // 2] if adps else None

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
        cap_penalty = 0.85 ** overdrafted  # discourage going way over an informal cap

        score = proj + 0.15 * adp_discount
        score *= (0.85 + 0.30 * need_boost)
        score *= cap_penalty

        scored.append({
            "id": p["id"],
            "name": p.get("name"),
            "team": p.get("team"),
            "pos": pos,
            "bye": p.get("bye"),
            "adp": p.get("adp"),
            "proj_ros": p.get("proj_ros"),
            "proj_week": p.get("proj_week"),
            "score": round(score, 3),
            "explain": f"{pos} need {need_boost:.2f}, cap {cur}/{cap}, pick {pick_number}"
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored

# ====== Models =========================================================================

class RecommendReq(BaseModel):
    draft_url: str        # e.g. https://api.sleeper.app/v1/draft/<DRAFT_ID>/picks
    roster_id: int        # your team slot id (from picks)
    pick_number: int
    season: int = 2025    # not used in scoring here, kept for API continuity
    roster_slots: Optional[Dict[str, int]] = None
    limit: int = 10

# ====== Auth helper ====================================================================

def auth_or_401(x_api_key: Optional[str]):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

# ====== Endpoints ======================================================================

@app.on_event("startup")
async def warmup():
    await load_players_if_needed()

@app.get("/health")
def health():
    return {"ok": True, "players_cached": len(PLAYERS_CACHE), "ts": int(time.time())}

@app.get("/draft/{draft_id}/team/{team_number}")
async def debug_team(draft_id: str, team_number: int, x_api_key: Optional[str] = Header(None)):
    """Helper to map a human team_number (1..N) to a real roster_id & list your current roster."""
    auth_or_401(x_api_key)
    await load_players_if_needed()

    picks = await _get_json(f"{SLEEPER}/draft/{draft_id}/picks")
    if not isinstance(picks, list):
        raise HTTPException(status_code=502, detail="Unexpected picks payload")

    # unique roster_ids in sorted order
    roster_ids = sorted({p["roster_id"] for p in picks if isinstance(p.get("roster_id"), int)})
    if team_number < 1 or team_number > len(roster_ids):
        raise HTTPException(status_code=400, detail="Invalid team number")
    roster_id = roster_ids[team_number - 1]

    team_picks = [p for p in picks if p.get("roster_id") == roster_id and p.get("player_id")]
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
    return {"team_number": team_number, "roster_id": roster_id, "roster": roster, "total_picks": len(team_picks)}

@app.post("/recommend_live")
async def recommend_live(body: RecommendReq, x_api_key: Optional[str] = Header(None)):
    """Live recommendations from the *current* draft pool. Strict: errors if draft fetch fails."""
    auth_or_401(x_api_key)
    await load_players_if_needed()

    # 1) Fetch the live draft picks. STRICT: if this fails → raise.
    try:
        draft_json = await _get_json(body.draft_url)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Draft fetch failed: {str(e)}")

    # 2) Parse drafted + my roster
    drafted_ids, by_roster = parse_drafted_from_picks(draft_json)
    my_ids = list(by_roster.get(body.roster_id, set()))

    # 3) Availability: filter out drafted + my roster; only real positions
    available: List[Dict[str, Any]] = []
    for pid, pdata in PLAYERS_CACHE.items():
        if not pdata.get("pos"):
            continue
        if pid in drafted_ids:
            continue
        if pid in my_ids:
            continue
        available.append(pdata)

    if not available:
        raise HTTPException(status_code=409, detail="No available players after filtering (live draft pool empty)")

    # 4) Scoring
    slots = body.roster_slots or {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 2, "DST": 1, "K": 1}
    scored = score_available(available, my_ids, slots, body.pick_number)
    if not scored:
        # This should be rare since 'available' was non-empty, but keep the guard.
        raise HTTPException(status_code=500, detail="Scoring produced no results")

    # 5) Build response: top N, plus some alternates
    limit = max(3, body.limit)
    ranked = scored[:limit]

    return {
        "pick": body.pick_number,
        "season_used": body.season,
        "recommended": ranked[:3],          # top 3 for the GPT to talk through
        "alternatives": ranked[3:limit],    # next-best
        "my_team": [PLAYERS_CACHE[pid] for pid in my_ids if pid in PLAYERS_CACHE],
        "all_drafted_count": len(drafted_ids),
        "my_team_count": len(my_ids),
        "ts": int(time.time())
    }
