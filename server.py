# server.py
# Fantasy Live Draft API (JSON-driven + resilient + season-aware)
# - Input: draft_json_url (Sleeper /draft/{id}/picks OR your hosted JSON export)
# - Always revalidates before recommending (ETag/Last-Modified)
# - Keeps a "last good" snapshot if a refresh blips
# - Retries with backoff on external HTTP calls
# - Warms players cache on startup; /warmup endpoint to preheat
# - Season-aware (default 2025). Pass projections_url/adp_url for current season.
# - Roster-aware via roster_id (draft slot 1..N)
#
# Security: set env var API_KEY on host; GPT sends header x-api-key: <same secret>

import os, time, asyncio, random
from typing import List, Dict, Optional, Set, Tuple, Any
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import httpx

API_KEY = os.getenv("API_KEY")  # optional in dev, recommended in prod
SLEEPER = "https://api.sleeper.app/v1"

app = FastAPI(title="Fantasy Live Draft API (JSON + Season + Resilience)")

# ===== CACHES =====
PLAYERS_CACHE: Dict[str, Dict] = {}
PLAYERS_LOADED_AT: float = 0

JSON_CACHE: Dict[str, Dict[str, Any]] = {}      # url -> {"etag","last_mod","data","ts"}
LAST_GOOD_DRAFT: Dict[str, Dict[str, Any]] = {}  # url -> {"data","ts"}

PROJ_CACHE: Dict[str, Dict[str, Any]] = {}  # key -> {"data","ts"}
ADP_CACHE:  Dict[str, Dict[str, Any]] = {}  # key -> {"data","ts"}

# ===== AUTH =====
def auth_or_401(x_api_key: Optional[str]):
    if not API_KEY:
        return  # unsecured (dev)
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

# ===== HTTP HELPERS (robust) =====
async def robust_http_get(url: str, headers: Optional[Dict[str,str]]=None,
                          max_attempts: int=3, base: float=0.3, timeout: float=60.0) -> httpx.Response:
    last_err = None
    for i in range(max_attempts):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.get(url, headers=headers)
                # Treat non-2xx/304 as raise_for_status errors
                if resp.status_code not in (200, 304):
                    resp.raise_for_status()
                return resp
        except Exception as e:
            last_err = e
            await asyncio.sleep(base * (2**i) + random.random()*0.25)
    raise HTTPException(status_code=502, detail=f"Upstream error fetching {url}: {last_err}")

async def http_get_json(url: str, etag: Optional[str]=None, last_mod: Optional[str]=None
                        ) -> Tuple[int, Dict, Optional[str], Optional[str]]:
    headers = {}
    if etag: headers["If-None-Match"] = etag
    if last_mod: headers["If-Modified-Since"] = last_mod
    resp = await robust_http_get(url, headers=headers, max_attempts=3, timeout=60.0)
    if resp.status_code == 304:
        return 304, {}, resp.headers.get("ETag"), resp.headers.get("Last-Modified")
    try:
        data = resp.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Couldn't parse JSON at {url}: {e}")
    return resp.status_code, data, resp.headers.get("ETag"), resp.headers.get("Last-Modified")

async def get_json_with_cache(url: str, force_revalidate: bool=True) -> Dict:
    entry = JSON_CACHE.get(url, {})
    etag, last_mod, data = entry.get("etag"), entry.get("last_mod"), entry.get("data")
    if force_revalidate or data is None:
        code, new_data, new_etag, new_last = await http_get_json(url, etag, last_mod)
        if code == 304 and data is not None:
            return data  # unchanged
        JSON_CACHE[url] = {"etag": new_etag, "last_mod": new_last, "data": new_data, "ts": time.time()}
        return new_data
    return data

# ===== PLAYERS CACHE =====
async def load_players_if_needed() -> None:
    global PLAYERS_CACHE, PLAYERS_LOADED_AT
    if PLAYERS_CACHE and (time.time() - PLAYERS_LOADED_AT) < 3*3600:
        return
    try:
        resp = await robust_http_get(f"{SLEEPER}/players/nfl", timeout=90.0)
        data = resp.json()
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
        # minimal fallback to avoid "empty"
        if not PLAYERS_CACHE:
            PLAYERS_CACHE.update({
                "999001":{"id":"999001","name":"Travis Etienne","pos":"RB","team":"JAX","bye":9,"adp":24,"proj_ros":245},
                "999002":{"id":"999002","name":"DK Metcalf","pos":"WR","team":"SEA","bye":10,"adp":28,"proj_ros":230},
                "999003":{"id":"999003","name":"Josh Allen","pos":"QB","team":"BUF","bye":13,"adp":20,"proj_ros":360}
            })
            PLAYERS_LOADED_AT = time.time()

# ===== PROJECTIONS / ADP MERGE =====
def merge_projection_fields(player: Dict, proj: Dict):
    if isinstance(proj, dict):
        if proj.get("proj_ros") is not None: player["proj_ros"] = proj["proj_ros"]
        if proj.get("proj_week") is not None: player["proj_week"] = proj["proj_week"]
        if proj.get("adp") is not None: player["adp"] = proj["adp"]

async def load_projections(season: int, projections_url: Optional[str]) -> None:
    if not projections_url:
        return
    key = f"proj::{season}::{projections_url}"
    entry = PROJ_CACHE.get(key)
    if entry and (time.time() - entry["ts"]) < 3*3600:
        data = entry["data"]
    else:
        data = await get_json_with_cache(projections_url, force_revalidate=True)
        PROJ_CACHE[key] = {"data": data, "ts": time.time()}
    items = data.items() if isinstance(data, dict) else [(str(d.get("id")), d) for d in (data if isinstance(data, list) else [])]
    for pid, metrics in items:
        p = PLAYERS_CACHE.get(str(pid))
        if p: merge_projection_fields(p, metrics)

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
    items = data.items() if isinstance(data, dict) else [(str(d.get("id")), d) for d in (data if isinstance(data, list) else [])]
    for pid, metrics in items:
        p = PLAYERS_CACHE.get(str(pid))
        if p and isinstance(metrics, dict) and metrics.get("adp") is not None:
            p["adp"] = metrics["adp"]

# ===== DRAFT JSON PARSE =====
def parse_drafted_from_sleeper_picks_json(draft_json: Any) -> Tuple[Set[str], Dict[int, Set[str]]]:
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

# ===== SCORING / NEEDS =====
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

# ===== MODELS =====
class RecommendReq(BaseModel):
    draft_json_url: str
    pick_number: int
    season: int = 2025
    roster_id: Optional[int] = None
    roster_slots: Optional[Dict[str,int]] = None
    projections_url: Optional[str] = None
    adp_url: Optional[str] = None
    limit: int = 10
    refresh: bool = True

# ===== LIFECYCLE & UTILS =====
@app.on_event("startup")
async def warm():
    await load_players_if_needed()

@app.get("/warmup")
async def warmup():
    await load_players_if_needed()
    return {"ok": True, "players_cached": len(PLAYERS_CACHE)}

@app.get("/health")
def health(): return {"ok": True, "ts": int(time.time())}

@app.get("/")
def root(): return {"ok": True, "hint": "POST /recommend_live (see /docs)"}

# ===== MAIN ENDPOINT =====
@app.post("/recommend_live")
async def recommend_live(body: RecommendReq, x_api_key: Optional[str] = Header(None)):
    auth_or_401(x_api_key)
    await load_players_if_needed()

    # Refresh draft JSON (with revalidation)
    drafted_ids: Set[str] = set()
    by_roster: Dict[int, Set[str]] = {}
    try:
        draft_json = await get_json_with_cache(body.draft_json_url, force_revalidate=bool(body.refresh))
        drafted_ids, by_roster = parse_drafted_from_sleeper_picks_json(draft_json)
        if drafted_ids or by_roster:
            LAST_GOOD_DRAFT[body.draft_json_url] = {"data": draft_json, "ts": int(time.time())}
        else:
            # If current fetch yielded nothing, try last-good snapshot
            lg = LAST_GOOD_DRAFT.get(body.draft_json_url)
            if lg:
                draft_json = lg["data"]
                drafted_ids, by_roster = parse_drafted_from_sleeper_picks_json(draft_json)
    except Exception:
        lg = LAST_GOOD_DRAFT.get(body.draft_json_url)
        if lg:
            draft_json = lg["data"]
            drafted_ids, by_roster = parse_drafted_from_sleeper_picks_json(draft_json)
        else:
            raise

    my_ids = list(by_roster.get(int(body.roster_id), set())) if body.roster_id is not None else []

    # Load season data (2025) if provided
    await load_projections(body.season, body.projections_url)
    await load_adp(body.season, body.adp_url)

    # Roster template
    roster_slots = body.roster_slots or {"QB":1,"RB":2,"WR":2,"TE":1,"FLEX":2,"DST":1,"K":1}

    # Available
    available_ids = [pid for pid in PLAYERS_CACHE if pid not in drafted_ids and pid not in my_ids]
    available = [PLAYERS_CACHE[pid] for pid in available_ids if PLAYERS_CACHE[pid].get("pos")]

    if not available:
        return {
            "pick": body.pick_number,
            "season_used": body.season,
            "recommended": [],
            "alternatives": [],
            "drafted_count": len(drafted_ids),
            "my_team_count": len(my_ids),
            "ts": int(time.time()),
            "debug": {
                "players_cache": len(PLAYERS_CACHE),
                "note": "Empty availability after filtering. Likely players cache warming or draft JSON empty. Using /warmup and retry."
            }
        }

    # Needs & caps
    needs = compute_needs(my_ids, roster_slots)
    pos_counts = {}
    for pid in my_ids:
        pos = PLAYERS_CACHE.get(pid,{}).get("pos")
        if pos: pos_counts[pos] = pos_counts.get(pos,0) + 1

    # ADP median for tie-break
    adps = [p.get("adp") for p in available if isinstance(p.get("adp"), (int,float))]
    adp_median = None
    if adps:
        s = sorted(adps); adp_median = s[len(s)//2]

    # Score
    scored = []
    for p in available:
        pos = p["pos"]
        proj = float(p.get("proj_ros") or p.get("proj_week") or 0.0)
        adp = p.get("adp")
        adp_discount = 0.0
        if adp_median is not None and isinstance(adp, (int,float)):
            adp_discount = (adp_median - adp)  # earlier ADP (smaller) -> smaller discount

        need_boost = needs.get(pos, 1.0)
        cur = pos_counts.get(pos,0)
        cap = roster_sane_cap(pos, roster_slots)
        overdrafted = max(0, cur - cap)
        cap_penalty = 0.85 ** overdrafted

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
