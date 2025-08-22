# server.py — Fantasy Live Draft helper (robust, ChatGPT-safe responses)
# - Always returns HTTP 200 with {"status":"ok" | "error", ...} so the Builder doesn't throw ClientResponseError
# - Adds /echo_auth for quick API-key verification
# - More tolerant Sleeper fetch with graceful 404/timeout handling
# - Fuzzy player-name matching for /guess_roster
# - Warmup caches players + rankings

import os, re, time, csv, math, random, asyncio
from typing import Any, Dict, List, Optional, Tuple, Set

import httpx
from fastapi import FastAPI, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel

SLEEPER = "https://api.sleeper.app/v1"
API_KEY = os.getenv("API_KEY")  # if set, require x-api-key

RANKINGS_CSV_PATH = os.getenv("RANKINGS_CSV_PATH", "rankings.csv")
HTTP_TIMEOUT = httpx.Timeout(12.0, connect=6.0, read=10.0)
UA = {"User-Agent": "fantasy-live-api/1.0"}

app = FastAPI()

# ------------ in-memory caches ------------
PLAYERS_RAW: Dict[str, Any] = {}
PLAYERS: Dict[str, Dict[str, Any]] = {}  # kept/usable subset (QB/RB/WR/TE)
PLAYERS_TTL_SEC: Optional[int] = None
PLAYERS_EXPIRE_AT: float = 0.0

RANKINGS: List[Dict[str, Any]] = []
RANKINGS_ROWS: int = 0
RANKINGS_LAST_MERGE: Optional[int] = None
RANKINGS_WARNINGS: List[str] = []

def _now() -> int:
    return int(time.time())

def _ok(payload: Dict[str, Any]) -> JSONResponse:
    payload.setdefault("ts", _now())
    return JSONResponse(status_code=200, content=payload)

def _err(reason: str, **extra) -> JSONResponse:
    out = {"status": "error", "reason": reason}
    out.update(extra)
    out.setdefault("ts", _now())
    return JSONResponse(status_code=200, content=out)

def _auth_bad(x_api_key: Optional[str]) -> Optional[JSONResponse]:
    if API_KEY:
        if not x_api_key:
            return _err("auth", message="x-api-key missing")
        if x_api_key != API_KEY:
            return _err("auth", message="x-api-key mismatch")
    return None

def _draft_id_from_url(url: str) -> Optional[str]:
    # accepts .../draft/nfl/1263988228017369088
    m = re.search(r"/draft/[^/]+/(\d+)", url)
    return m.group(1) if m else None

async def _fetch_json(client: httpx.AsyncClient, url: str) -> Tuple[int, Optional[Any], Optional[str]]:
    try:
        r = await client.get(url, timeout=HTTP_TIMEOUT, headers=UA, follow_redirects=True)
        code = r.status_code
        try:
            data = r.json()
        except Exception:
            data = None
        return code, data, None if data is not None else r.text
    except Exception as e:
        return 0, None, f"{type(e).__name__}: {e}"

def _normalize_name(name: str) -> str:
    s = name.lower()
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ------------- data loading -------------
async def _load_players() -> None:
    global PLAYERS_RAW, PLAYERS, PLAYERS_TTL_SEC, PLAYERS_EXPIRE_AT
    # pull once per ~18 minutes unless warmup called
    if _now() < PLAYERS_EXPIRE_AT and PLAYERS:
        return
    async with httpx.AsyncClient() as client:
        code, data, err = await _fetch_json(client, f"{SLEEPER}/players/nfl")
        if code != 200 or not isinstance(data, dict):
            # keep previous cache if exists
            PLAYERS_RAW = {}
            PLAYERS = {}
            PLAYERS_TTL_SEC = None
            PLAYERS_EXPIRE_AT = _now() + 60  # try again soon
            return
        PLAYERS_RAW = data
        keep_pos = {"QB", "RB", "WR", "TE"}
        kept = {}
        for pid, p in data.items():
            pos = p.get("position")
            if pos in keep_pos:
                kept[pid] = {
                    "id": pid,
                    "name": p.get("full_name") or p.get("first_name", "") + " " + p.get("last_name", ""),
                    "team": p.get("team"),
                    "pos": pos,
                    "bye": p.get("bye_week"),
                }
        PLAYERS = kept
        PLAYERS_TTL_SEC = 1080  # 18 minutes
        PLAYERS_EXPIRE_AT = _now() + PLAYERS_TTL_SEC

def _load_rankings() -> None:
    global RANKINGS, RANKINGS_ROWS, RANKINGS_LAST_MERGE, RANKINGS_WARNINGS
    rows = []
    warns: List[str] = []
    if not os.path.exists(RANKINGS_CSV_PATH):
        RANKINGS = []
        RANKINGS_ROWS = 0
        RANKINGS_LAST_MERGE = _now()
        RANKINGS_WARNINGS = ["rankings.csv not found"]
        return
    try:
        with open(RANKINGS_CSV_PATH, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                name = (r.get("PLAYER") or r.get("name") or "").strip()
                pos = (r.get("POS") or r.get("pos") or "").strip().upper()
                team = (r.get("TEAM") or r.get("team") or "").strip().upper()
                adp = r.get("ADP") or r.get("adp") or ""
                rank_avg = r.get("AVG") or r.get("rank_avg") or r.get("rank") or ""
                proj = r.get("PROJ_ROS") or r.get("proj_ros") or ""
                try:
                    adp_v = float(adp) if adp != "" else None
                except:
                    adp_v = None
                try:
                    rank_v = float(rank_avg) if rank_avg != "" else None
                except:
                    rank_v = None
                try:
                    proj_v = float(proj) if proj != "" else None
                except:
                    proj_v = None
                rows.append({
                    "name": name,
                    "team": team,
                    "pos": pos,
                    "adp": adp_v,
                    "rank_avg": rank_v,
                    "proj_ros": proj_v,
                    "score": (proj_v or 0.0) if proj_v is not None else (200.0 - (rank_v or 200.0))
                })
    except Exception as e:
        warns.append(f"rankings read error: {e}")
    RANKINGS = rows
    RANKINGS_ROWS = len(rows)
    RANKINGS_LAST_MERGE = _now()
    RANKINGS_WARNINGS = warns

def _best_matches(names: List[str], picks: List[Dict[str, Any]]) -> Dict[str, int]:
    """Return roster_id counts for guessed player names using fuzzy contains."""
    want = [_normalize_name(n) for n in names if n]
    counts: Dict[int, int] = {}
    for pk in picks:
        r_id = pk.get("roster_id")
        pname = _normalize_name(pk.get("metadata", {}).get("first_name", "") + " " + pk.get("metadata", {}).get("last_name", ""))
        alt = _normalize_name(pk.get("player", "") or pk.get("player_name", ""))
        for w in want:
            if w and (w in pname or w in alt):
                counts[r_id] = counts.get(r_id, 0) + 1
    return counts

# ------------- models -------------
class InspectDraftReq(BaseModel):
    draft_url: Optional[str] = None
    league_id: Optional[str] = None
    roster_id: Optional[int] = None
    team_slot: Optional[int] = None
    team_name: Optional[str] = None

class GuessRosterReq(BaseModel):
    draft_url: str
    player_names: List[str]

class RecommendLiveReq(BaseModel):
    draft_url: Optional[str] = None
    league_id: Optional[str] = None
    roster_id: Optional[int] = None
    team_slot: Optional[int] = None
    team_name: Optional[str] = None
    pick_number: Optional[int] = None
    season: int = 2025
    roster_slots: Dict[str, int] = {"QB":1, "RB":2, "WR":2, "TE":1, "FLEX":2}
    limit: int = 10

# ------------- routes -------------

@app.get("/echo_auth")
async def echo_auth(x_api_key: Optional[str] = Header(default=None)):
    got_present = bool(x_api_key)
    got_len = len(x_api_key) if x_api_key else 0
    exp_present = bool(API_KEY)
    match = bool(API_KEY and x_api_key == API_KEY)
    return _ok({
        "ok": True,
        "got_present": got_present,
        "got_len": got_len,
        "exp_present": exp_present,
        "match": match
    })

@app.get("/health")
async def health():
    # No auth required
    ttl_left = max(0, int(PLAYERS_EXPIRE_AT - _now())) if PLAYERS_EXPIRE_AT else None
    return _ok({
        "ok": True,
        "players_cached": len(PLAYERS),
        "players_raw": len(PLAYERS_RAW),
        "players_kept": len(PLAYERS),
        "players_ttl_sec": ttl_left,
        "rankings_rows": RANKINGS_ROWS,
        "rankings_last_merge": RANKINGS_LAST_MERGE,
        "rankings_warnings": RANKINGS_WARNINGS,
    })

@app.get("/warmup")
async def warmup():
    # No auth required
    await _load_players()
    _load_rankings()
    return _ok({
        "ok": True,
        "players_cached": len(PLAYERS),
        "players_raw": len(PLAYERS_RAW),
        "players_kept": len(PLAYERS),
        "rankings_rows": RANKINGS_ROWS,
        "rankings_warnings": RANKINGS_WARNINGS,
    })

@app.post("/inspect_draft")
async def inspect_draft(body: InspectDraftReq, x_api_key: Optional[str] = Header(default=None)):
    bad = _auth_bad(x_api_key)
    if bad: return bad

    draft_id = None
    league_id = body.league_id
    if body.draft_url:
        draft_id = _draft_id_from_url(body.draft_url)
        if not draft_id:
            return _err("input", message="draft_url not recognized")
    async with httpx.AsyncClient() as client:
        draft_json = {}
        if draft_id:
            code, djson, err = await _fetch_json(client, f"{SLEEPER}/draft/{draft_id}")
            if code == 200 and isinstance(djson, dict):
                draft_json = djson
                league_id = league_id or djson.get("league_id")
        league_json = {}
        if league_id:
            code, ljson, err = await _fetch_json(client, f"{SLEEPER}/league/{league_id}")
            if code == 200 and isinstance(ljson, dict):
                league_json = ljson
        # picks (may 404 during early/live periods)
        picks: List[Dict[str, Any]] = []
        picks_note = None
        if draft_id:
            code, pjson, err = await _fetch_json(client, f"{SLEEPER}/draft/{draft_id}/picks")
            if code == 200 and isinstance(pjson, list):
                picks = pjson
            else:
                picks_note = f"picks_fetch code={code}; err={err}"
        # minimal counts
        drafted_count = len(picks)
        return _ok({
            "status": "ok",
            "draft_state": league_json or draft_json or {},
            "draft_id": draft_id,
            "league_id": league_id,
            "drafted_count": drafted_count,
            "picks_note": picks_note,
            "input": body.model_dump()
        })

@app.post("/guess_roster")
async def guess_roster(body: GuessRosterReq, x_api_key: Optional[str] = Header(default=None)):
    bad = _auth_bad(x_api_key)
    if bad: return bad
    draft_id = _draft_id_from_url(body.draft_url)
    if not draft_id:
        return _err("input", message="draft_url not recognized")

    async with httpx.AsyncClient() as client:
        code, pjson, err = await _fetch_json(client, f"{SLEEPER}/draft/{draft_id}/picks")
        if code != 200 or not isinstance(pjson, list):
            return _ok({
                "status": "error",
                "reason": "sleeper_picks",
                "draft_id": draft_id,
                "message": f"picks fetch failed code={code}",
                "detail": err
            })
        counts = _best_matches(body.player_names, pjson)
        if not counts:
            return _ok({
                "status": "ok",
                "draft_id": draft_id,
                "candidates": [],
                "guessed_roster_id": None,
                "note": "no matches found"
            })
        # best roster by matches
        best = max(counts.items(), key=lambda kv: kv[1])[0]
        cands = [{"roster_id": rid, "matches": cnt, "players": body.player_names} for rid, cnt in sorted(counts.items(), key=lambda kv: -kv[1])]
        return _ok({
            "status": "ok",
            "draft_id": draft_id,
            "candidates": cands,
            "guessed_roster_id": best
        })

@app.post("/recommend_live")
async def recommend_live(body: RecommendLiveReq, x_api_key: Optional[str] = Header(default=None)):
    bad = _auth_bad(x_api_key)
    if bad: return bad
    await _load_players()
    _load_rankings()

    draft_id = None
    league_id = body.league_id
    if body.draft_url:
        draft_id = _draft_id_from_url(body.draft_url)
    # fetch picks to exclude drafted
    picks: List[Dict[str, Any]] = []
    async with httpx.AsyncClient() as client:
        if draft_id:
            code, pjson, err = await _fetch_json(client, f"{SLEEPER}/draft/{draft_id}/picks")
            if code == 200 and isinstance(pjson, list):
                picks = pjson

    drafted_names: Set[str] = set()
    for pk in picks:
        n = (pk.get("player", "") or pk.get("player_name", "") or "").strip()
        if not n:
            fn = pk.get("metadata", {}).get("first_name", "")
            ln = pk.get("metadata", {}).get("last_name", "")
            n = (fn + " " + ln).strip()
        if n:
            drafted_names.add(_normalize_name(n))

    # naive roster-aware filter (fill caps by POS)
    caps = dict(body.roster_slots or {"QB":1,"RB":2,"WR":2,"TE":1,"FLEX":2})
    counts = {k:0 for k in caps.keys()}

    recs: List[Dict[str, Any]] = []
    for row in RANKINGS:
        n_key = _normalize_name(row["name"])
        if n_key in drafted_names:
            continue
        pos = row["pos"]
        # map position into slot
        slot_key = None
        if pos in caps:
            slot_key = pos
        elif "FLEX" in caps and pos in {"RB","WR","TE"}:
            slot_key = "FLEX"
        else:
            continue
        # simple capacity check — for preview we still show even if caps full
        score = row.get("score") or 0.0
        recs.append({
            "id": row.get("id"),
            "name": row["name"],
            "team": row.get("team"),
            "pos": pos,
            "adp": row.get("adp"),
            "rank_avg": row.get("rank_avg"),
            "proj_ros": row.get("proj_ros"),
            "score": score,
            "explain": f"rank={row.get('rank_avg')}, proj={row.get('proj_ros')}, pick {body.pick_number}" if body.pick_number else ""
        })
        if len(recs) >= max(10, body.limit or 10):
            break

    return _ok({
        "status": "ok",
        "pick": body.pick_number,
        "season_used": body.season,
        "recommended": recs,
        "alternatives": [],
        "my_team": [],
        "draft_state": {},
        "effective_roster_id": body.roster_id,
        "drafted_count": len(picks),
        "my_team_count": 0
    })
