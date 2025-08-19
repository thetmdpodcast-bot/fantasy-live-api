# server.py
# Fantasy Live Draft API (JSON-driven + season-aware)
# - Provide a Sleeper draft-room JSON URL (e.g., /draft/{id}/picks or a saved JSON file).
# - The API will revalidate/refresh that JSON (ETag/Last-Modified) *right before* recommending.
# - Optional roster_id (your draft slot 1..N) makes it roster-aware.
# - Season-aware: pass season (e.g., 2025) + optional projections_url/adp_url for that season.
#
# Security: set env var API_KEY on your host; GPT sends header x-api-key: <same secret>

import os, time
from typing import List, Dict, Optional, Set, Tuple, Any
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import httpx

API_KEY = os.getenv("API_KEY")  # shared secret for your GPT Action (optional in dev)
SLEEPER = "https://api.sleeper.app/v1"

app = FastAPI(title="Fantasy Live Draft API (JSON + Season)")

# =========================
# CACHES
# =========================
PLAYERS_CACHE: Dict[str, Dict] = {}   # Sleeper players (id -> {pos, team, bye})
PLAYERS_LOADED_AT: float = 0

# Per-URL JSON cache with revalidation metadata
JSON_CACHE: Dict[str, Dict[str, Any]] = {}  # url -> {"etag":..., "last_mod":..., "data":..., "ts":...}

# Projections/ADP by season or URL
PROJ_CACHE: Dict[str, Dict[str, Any]] = {}  # key -> {"data":..., "ts":...}
ADP_CACHE:  Dict[str, Dict[str, Any]] = {}

# =========================
# UTILS
# =========================
def auth_or_401(x_api_key: Optional[str]):
    if not API_KEY:
        return  # unsecured for local dev
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

async def http_get_json(url: str, etag: Optional[str]=None, last_mod: Optional[str]=None) -> Tuple[int, Dict, Optional[str], Optional[str]]:
    """GET JSON with conditional headers; returns (status_code, json_or_cached, new_etag, new_last_mod)."""
    headers = {}
    if etag: headers["If-None-Match"] = etag
    if last_mod: headers["If-Modified-Since"] = last_mod

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.get(url, headers=headers)
    if resp.status_code == 304:  # Not Modified
        return 304, {}, resp.headers.get("ETag"), resp.headers.get("Last-Modified")
    resp.raise_for_status()
    try:
        data = resp.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Couldn't parse JSON at {url}: {e}")
    return resp.status_code, data, resp.headers.get("ETag"), resp.headers.get("Last-Modified")

async def get_json_with_cache(url: str, force_revalidate: bool=True) -> Dict:
    """Return JSON for URL with ETag/Last-Modified revalidation. If force_revalidate=True, always revalidate."""
    entry = JSON_CACHE.get(url, {})
    etag = entry.get("etag")
    last_mod = entry.get("last_mod")
    data = entry.get("data")

    if force_revalidate or not data:
        code, new_data, new_etag, new_last = await http_get_json(url, etag, last_mod)
        if code == 304 and data is not None:
            # unchanged
            return data
        # updated or first load
        JSON_CACHE[url] = {"etag": new_etag, "last_mod": new_last, "data": new_data, "ts": time.time()}
        return new_data
    return data

async def load_players_if_needed() -> None:
    """Load Sleeper players catalog (positions/teams); refresh every 3h. Fallback to tiny pool if slow."""
    global PLAYERS_CACHE, PLAYERS_LOADED_AT
    if PLAYERS_CACHE and (time.time() - PLAYERS_LOADED_AT) < 3*3600:
        return
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.get(f"{SLEEPER}/players/nfl")
            r.raise_for_status()
            data = r.json()
        PLAYERS_CACHE = {}
        for pid, p in data.items():
            name = p.get("full_name") or f"{p.get('first_name','')} {p.get('last_name','')}".strip()
            PLAYERS_CACHE[pid] = {
                "id": pid, "name": name, "team": p.get("team"),
                "pos": p.get("position"), "bye": p.get("bye_week"),
                "adp": None, "proj_ros": None, "proj_week": None
            }
        PLAYERS_LOADED_AT = time.time()
    except Exception:
        if not PLAYERS_CACHE:
            # minimal fallback so we never return empty
            PLAYERS_CACHE.update({
                "999001":{"id":"999001","name":"Travis Etienne","pos":"RB","team":"JAX","bye":9,"adp":24,"proj_ros":245},
                "999002":{"id":"999002","name":"DK Metcalf","pos":"WR","team":"SEA","bye":10,"adp":28,"proj_ros":230},
                "999003":{"id":"999003","name":"Josh Allen","pos":"QB","team":"BUF","bye":13,"adp":20,"proj_ros":360}
            })
            PLAYERS_LOADED_AT = time.time()

def merge_projection_fields(player: Dict, proj: Dict):
    """Merge projection/ADP fields into players cache entry."""
    if "proj_ros" in proj and proj["proj_ros"] is not None:
        player["proj_ros"] = proj["proj_ros"]
    if "proj_week" in proj and proj["proj_week"] is not None:
        player["proj_week"] = proj["proj_week"]
    if "adp" in proj and proj["adp"] is not None:
        player["adp"] = proj["adp"]

async def load_projections(season: int, projections_url: Optional[str]) -> None:
    """Load season projections into PLAYERS_CACHE from projections_url JSON (id->metrics) or skip if None."""
    if not projections_url:
        return
    key = f"proj::{season}::{projections_url}"
    entry = PROJ_CACHE.get(key)
    if entry and (time.time() - entry["ts"]) < 3*3600:
        # already loaded recently
        data = entry["data"]
    else:
        data = await get_json_with_cache(projections_url, force_revalidate=True)
        PROJ_CACHE[key] = {"data": data, "ts": time.time()}

    # data can be {player_id: {proj_ros, proj_week, adp}} or array of objects with "id"
    if isinstance(data, dict):
        items = data.items()
    elif isinstance(data, list):
        items = [(str(d.get("id")), d) for d in data]
    else:
        items = []

    for pid, metrics in items:
        p = PLAYERS_CACHE.get(str(pid))
        if p and isinstance(metrics, dict):
            merge_projection_fields(p, metrics)

async def load_adp(season: int, adp_url: Optional[str]) -> None:
    if not adp_url:
        return
    key = f"adp::{season}::{adp_url}"
    entry = ADP_CACHE.get(key)
    if entry and (time.time() - entry["ts"]) < 3*3600:
        data = entry["data"]
    else:
        data = await get_json_with_cache(adp_url, force_revalidate=True)
        ADP_CACHE[key] = {"data": data, "ts": time.time()}

    # Merge ADP by player id
    if isinstance(data, dict):
        items = data.items()
    elif isinstance(data, list):
        items = [(str(d.get("id")), d) for d in data]
    else:
        items = []
    for pid, metrics in items:
        p = PLAYERS_CACHE.get(str(pid))
        if p and isinstance(metrics, dict):
            if "adp" in metrics and metrics["adp"] is not None:
                p["adp"] = metrics["adp"]

# =========================
# DRAFT JSON PARSING
# =========================
def parse_drafted_from_sleeper_picks_json(draft_json: Any) -> Tuple[Set[str], Dict[int, Set[str]]]:
    """
    Supports Sleeper /draft/{id}/picks output (list of picks) or a saved JSON with similar shape.
    Returns (all_drafted_ids, drafted_by_roster_id).
    """
    drafted: Set[str] = set()
    by_roster: Dict[int, Set[str]] = {}
    if isinstance(draft_json, list):
        picks = draft_json
    elif isinstance(draft_json, dict) and "picks" in draft_json:
        picks = draft_json["picks"]
    else:
        picks = []

    for p in picks:
        pid = p.get("player_id")
        rid = p.get("roster_id")
        if pid:
            drafted.add(str(pid))
            if isinstance(rid, int):
                by_roster.setdefault(rid, set()).add(str(pid))
    return drafted, by_roster

# =========================
# SCORING / NEEDS
# =========================
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

# =========================
# REQUEST / RESPONSE MODELS
# =========================
class RecommendReq(BaseModel):
    draft_json_url: str                 # REQUIRED: the Sleeper picks JSON URL (or a hosted JSON file)
    pick_number: int
    season: int = 2025                  # ensure recs are for the current season
    roster_id: Optional[int] = None     # your draft slot 1..N
    roster_slots: Optional[Dict[str,int]] = None  # override if not standard
    projections_url: Optional[str] = None         # optional: season projections JSON (id->metrics)
    adp_url: Optional[str] = None                 # optional: ADP JSON (id->adp)
    limit: int = 10
    refresh: bool = True                # if True, revalidate draft_json_url before recommending

# =========================
# ENDPOINTS
# =========================
@app.get("/health")
def health(): return {"ok": True, "ts": int(time.time())}

@app.get("/")
def root(): return {"ok": True, "hint": "Use POST /recommend_live with draft_json_url. See /docs."}

@app.post("/recommend_live")
async def recommend_live(body: RecommendReq, x_api_key: Optional[str] = Header(None)):
    auth_or_401(x_api_key)

    # Always load players catalog for positions/teams
    await load_players_if_needed()

    # Revalidate/refresh the draft json right now
    draft_json = await get_json_with_cache(body.draft_json_url, force_revalidate=bool(body.refresh))

    # Parse drafted + my picks (by roster slot)
    drafted_ids, by_roster = parse_drafted_from_sleeper_picks_json(draft_json)
    my_ids = list(by_roster.get(int(body.roster_id), set())) if body.roster_id is not None else []

    # Load season-specific projections/ADP if provided (2025 by default)
    await load_projections(body.season, body.projections_url)
    await load_adp(body.season, body.adp_url)

    # Roster slots baseline (can be overridden)
    roster_slots = body.roster_slots or {"QB":1,"RB":2,"WR":2,"TE":1,"FLEX":2,"DST":1,"K":1}

    # Available = players catalog - drafted - mine
    available_ids = [pid for pid in PLAYERS_CACHE if pid not in drafted_ids and pid not in my_ids]
    available = [PLAYERS_CACHE[pid] for pid in available_ids if PLAYERS_CACHE[pid].get("pos")]

    if not available:
        return {
            "pick": body.pick_number,
            "season_used": body.season,
            "recommended": [],
            "alternatives": [],
            "debug": {
                "players_cache": len(PLAYERS_CACHE),
                "drafted_count": len(drafted_ids),
                "mine_count": len(my_ids),
                "note": "Empty availability. If this is a fresh deploy, wait for the players cache to warm; also confirm draft_json_url returns picks."
            }
        }

    # Needs & caps
    needs = compute_needs(my_ids, roster_slots)
    pos_counts = {}
    for pid in my_ids:
        pos = PLAYERS_CACHE.get(pid,{}).get("pos")
        if pos: pos_counts[pos] = pos_counts.get(pos,0) + 1

    # Score (projection -> ADP -> need -> cap penalty)
    scored = []
    # Provide a median ADP for tie-breaks if ADP exists
    adps = [p.get("adp") for p in available if isinstance(p.get("adp"), (int,float))]
    adp_median = None
    if adps:
        adps_sorted = sorted(adps)
        adp_median = adps_sorted[len(adps_sorted)//2]

    for p in available:
        pos = p["pos"]
        proj = float(p.get("proj_ros") or p.get("proj_week") or 0.0)
        adp = p.get("adp")
        adp_discount = 0.0
        if adp_median is not None and isinstance(adp, (int,float)):
            adp_discount = (adp_median - adp)  # lower ADP (earlier) => smaller discount

        need_boost = needs.get(pos, 1.0)
        cur = pos_counts.get(pos,0)
        cap = roster_sane_cap(pos, roster_slots)
        overdrafted = max(0, cur - cap)
        cap_penalty = 0.85 ** overdrafted

        # weight: projections dominate; adp helps tie-break; then need; penalize over-cap stacks
        score = proj
        score += 0.15 * adp_discount
        score *= (0.85 + 0.30 * need_boost)
        score *= cap_penalty

        scored.append({
            "id": p["id"], "name": p.get("name"), "team": p.get("team"),
            "pos": pos, "bye": p.get("bye"),
            "proj_ros": p.get("proj_ros"), "proj_week": p.get("proj_week"),
            "adp": p.get("adp"),
            "score": round(score,3),
            "explain": f"{pos} need {need_boost:.2f}, cap {cur}/{cap}, season {body.season}"
        })

    ranked = sorted(scored, key=lambda x: x["score"], reverse=True)[:max(3, body.limit)]
    return {
        "pick": body.pick_number,
        "season_used": body.season,
        "recommended": ranked[:3],
        "alternatives": ranked[3:body.limit],
        "drafted_count": len(drafted_ids),
        "my_team_count": len(my_ids),
        "ts": int(time.time())
    }
