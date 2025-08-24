# server.py
from __future__ import annotations
import os, re, csv, json, time, math, asyncio, random
from typing import Any, Dict, List, Optional, Tuple
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import httpx

APP_VERSION = "1.0.8"

app = FastAPI(title="Fantasy Live Draft API")

# ==== auth ====
EXPECTED_API_KEY = os.getenv("API_KEY", "")

def require_api_key(x_api_key: Optional[str]):
    if not EXPECTED_API_KEY:
        return  # auth disabled
    if not x_api_key or x_api_key != EXPECTED_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

# ==== Sleeper helpers ====
SLEEPER = "https://api.sleeper.app/v1"

_DEFAULT_HEADERS = {
    "User-Agent": "fantasy-live/1.0 (+onrender)",
    "Accept": "application/json",
    "Cache-Control": "no-cache",
}

async def _get_json(url: str, *, timeout=20) -> Tuple[int, Any]:
    # add a small cache-buster so CF doesn't hand us a stale edge
    sep = "&" if "?" in url else "?"
    url = f"{url}{sep}_cb={int(time.time()*1000)}"
    async with httpx.AsyncClient(timeout=timeout, http2=False, headers=_DEFAULT_HEADERS) as cli:
        r = await cli.get(url)
        status = r.status_code
        try:
            data = r.json()
        except Exception:
            data = None
        return status, data

async def _try_many(urls: List[str], *, attempts=3, base_delay=0.35) -> Tuple[int, Any, List[str]]:
    """Try a list of URLs with tiny backoff. Return (status, data, debug)."""
    dbg: List[str] = []
    last_status, last_data = 0, None
    for u in urls:
        for i in range(attempts):
            s, d = await _get_json(u)
            dbg.append(f"GET {u} -> {s}")
            if s == 200 and d is not None:
                return s, d, dbg
            # small backoff / jitter
            await asyncio.sleep(base_delay + random.random() * 0.25)
        last_status, last_data = s, d
    return last_status, last_data, dbg

# ==== rankings cache ====
_rankings_rows: List[Dict[str, Any]] = []
_rankings_index_by_norm: Dict[str, Dict[str, Any]] = {}
_last_rankings_merge_ts: Optional[int] = None

# Emergency list if CSV + joins fail (never blank UI)
_EMERGENCY_TOP: List[Dict[str, Any]] = [
    {"name":"Josh Allen","team":"BUF","pos":"QB","rank_avg":1.0},
    {"name":"Patrick Mahomes","team":"KC","pos":"QB","rank_avg":2.0},
    {"name":"Jalen Hurts","team":"PHI","pos":"QB","rank_avg":3.0},
    {"name":"Lamar Jackson","team":"BAL","pos":"QB","rank_avg":4.0},
    {"name":"Drake London","team":"ATL","pos":"WR","rank_avg":16.0},
    {"name":"Jonathan Taylor","team":"IND","pos":"RB","rank_avg":18.0},
]

_norm_rx = re.compile(r"[^a-z0-9]+")

def norm(s: str) -> str:
    return _norm_rx.sub("", s.lower().strip()) if isinstance(s, str) else ""

def _load_rankings_from_csv() -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Expect a local rankings.csv with at least columns:
    name, team, pos, rank (or rank_avg)
    """
    path = os.getenv("RANKINGS_CSV", "rankings.csv")
    rows: List[Dict[str, Any]] = []
    index: Dict[str, Dict[str, Any]] = {}

    if not os.path.exists(path):
        return rows, index

    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if not isinstance(row, dict):
                continue
            name = row.get("name") or row.get("player") or row.get("Player") or ""
            team = row.get("team") or row.get("Team") or row.get("tm") or ""
            pos  = row.get("pos") or row.get("POS") or row.get("Pos") or ""
            rank_s = row.get("rank_avg") or row.get("AVG") or row.get("rank") or ""
            adp_s  = row.get("adp") or row.get("ADP") or ""

            try:
                rank = float(rank_s) if rank_s not in (None, "", "NA") else None
            except Exception:
                rank = None
            try:
                adp = float(adp_s) if adp_s not in (None, "", "NA") else None
            except Exception:
                adp = None

            rec = {
                "name": name,
                "team": team,
                "pos": pos,
                "rank_avg": rank,
                "adp": adp,
                "proj_ros": None,
            }
            rows.append(rec)
            if name:
                index[norm(name)] = rec

    rows.sort(key=lambda d: (math.inf if d.get("rank_avg") in (None, "") else d["rank_avg"]))
    return rows, index

def warm_rankings() -> None:
    global _rankings_rows, _rankings_index_by_norm, _last_rankings_merge_ts
    rows, index = _load_rankings_from_csv()
    _rankings_rows = rows
    _rankings_index_by_norm = index
    _last_rankings_merge_ts = int(time.time())

# Load once on boot
warm_rankings()

# ==== models ====
class HealthResponse(BaseModel):
    ok: bool
    players_cached: Optional[int] = None
    players_raw: Optional[int] = None
    players_kept: Optional[int] = None
    players_ttl_sec: Optional[int] = None
    rankings_rows: int
    rankings_last_merge: Optional[int] = None
    rankings_warnings: List[str] = []
    ts: int

class WarmupResponse(BaseModel):
    ok: bool
    players_cached: Optional[int] = None
    players_raw: Optional[int] = None
    players_kept: Optional[int] = None
    rankings_rows: int
    rankings_warnings: List[str] = []
    ts: int

class EchoAuthResponse(BaseModel):
    ok: bool
    got_present: bool
    got_len: int
    exp_present: bool
    match: bool

class InspectDraftRequest(BaseModel):
    draft_url: Optional[str] = None
    league_id: Optional[str] = None
    roster_id: Optional[int] = None
    team_slot: Optional[int] = None
    team_name: Optional[str] = None

class InspectDraftResponse(BaseModel):
    status: str
    draft_state: Dict[str, Any]
    slot_to_roster_raw: Optional[Any] = None
    slot_to_roster_normalized: Optional[Any] = None
    observed_roster_ids: List[int] = []
    by_roster_counts: Dict[str, int] = {}
    input: Dict[str, Any] = {}
    effective_roster_id: Optional[int] = None
    effective_team_slot: Optional[int] = None
    my_team: List[Dict[str, Any]] = []
    drafted_count: int
    my_team_count: int
    undrafted_count: Optional[int] = None
    csv_matched_count: Optional[int] = None
    csv_top_preview: Optional[List[Dict[str, Any]]] = None
    ts: int

class RecommendLiveRequest(BaseModel):
    draft_url: Optional[str] = None
    league_id: Optional[str] = None
    roster_id: Optional[int] = None
    team_slot: Optional[int] = None
    team_name: Optional[str] = None
    pick_number: Optional[int] = None
    season: Optional[int] = 2025
    roster_slots: Optional[Dict[str, int]] = None
    limit: Optional[int] = 10

class RecommendLiveResponse(BaseModel):
    status: str
    pick: Optional[int] = None
    season_used: int
    recommended: List[Dict[str, Any]]
    alternatives: List[Dict[str, Any]] = []
    my_team: List[Dict[str, Any]] = []
    draft_state: Dict[str, Any]
    effective_roster_id: Optional[int] = None
    drafted_count: int
    my_team_count: int
    debug_reason: List[str] = []
    ts: int

# ==== utility: draft parsing ====

async def _draft_id_from_url(draft_url: str) -> str:
    # https://sleeper.com/draft/nfl/1263988228017369088
    did = draft_url.strip().rstrip("/").split("/")[-1]
    if not did.isdigit():
        raise HTTPException(status_code=400, detail="Invalid draft_url")
    return did

def _names_from_picks(picks: List[Dict[str, Any]]) -> List[str]:
    out = []
    for p in picks or []:
        md = p.get("metadata") if isinstance(p, dict) else None
        name = None
        if isinstance(md, dict):
            name = md.get("player_name") or md.get("first_name")
            if md.get("last_name"):
                name = f"{name} {md['last_name']}".strip() if name else md["last_name"]
        if not name:
            name = p.get("player_name") or p.get("player") or p.get("name")
        if name:
            out.append(name)
    return out

async def _fetch_draft_bundle(draft_id: str, league_id: Optional[str]) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[str]]:
    """Return (draft_state, picks, debug). Never raises, never returns None."""
    debug: List[str] = []
    draft_state: Dict[str, Any] = {"draft_id": draft_id, "league_id": league_id, "picks_made": 0}

    # 1) confirm the draft object exists
    s1, draft_obj, d1 = await _try_many([f"{SLEEPER}/draft/{draft_id}"])
    debug += d1
    if s1 == 200 and isinstance(draft_obj, dict):
        draft_state["status"] = draft_obj.get("status")
        draft_state["picks_made"] = draft_obj.get("last_picked") or 0
        draft_state["draft_order"] = draft_obj.get("draft_order")

    # 2) try picks
    s2, picks, d2 = await _try_many([f"{SLEEPER}/draft/{draft_id}/picks"])
    debug += d2
    if s2 == 200 and isinstance(picks, list) and picks:
        draft_state["picks_made"] = max(draft_state.get("picks_made", 0), len(picks))
        return draft_state, picks, debug

    # 3) fallback to board
    s3, board, d3 = await _try_many([f"{SLEEPER}/draft/{draft_id}/board"])
    debug += d3
    if s3 == 200 and isinstance(board, list) and board:
        # board is a list of picks too
        draft_state["picks_made"] = max(draft_state.get("picks_made", 0), len(board))
        return draft_state, board, debug

    # 4) if still empty and league_id present, try re-deriving draft id then picks again
    if league_id:
        s4, drafts, d4 = await _try_many([f"{SLEEPER}/league/{league_id}/drafts"])
        debug += d4
        if s4 == 200 and isinstance(drafts, list) and drafts:
            # take the one with matching id or just the most recent
            ids = [d.get("draft_id") for d in drafts if isinstance(d, dict)]
            if draft_id in ids:
                pass  # keep draft_id
            elif ids:
                new_id = ids[0]
                debug.append(f"refresh_draft_id={new_id}")
                draft_state["draft_id"] = new_id
                # try picks again on refreshed id
                s5, picks2, d5 = await _try_many([f"{SLEEPER}/draft/{new_id}/picks"])
                debug += d5
                if s5 == 200 and isinstance(picks2, list) and picks2:
                    draft_state["picks_made"] = max(draft_state.get("picks_made", 0), len(picks2))
                    return draft_state, picks2, debug

    # Give up gracefully
    return draft_state, [], debug

# ==== endpoints ====

@app.get("/health", response_model=HealthResponse)
async def health():
    now = int(time.time())
    return HealthResponse(
        ok=True,
        players_cached=None,
        players_raw=None,
        players_kept=None,
        players_ttl_sec=None,
        rankings_rows=len(_rankings_rows),
        rankings_last_merge=_last_rankings_merge_ts,
        rankings_warnings=[],
        ts=now,
    )

@app.get("/warmup", response_model=WarmupResponse)
async def warmup():
    warm_rankings()
    return WarmupResponse(
        ok=True,
        players_cached=None,
        players_raw=None,
        players_kept=None,
        rankings_rows=len(_rankings_rows),
        rankings_warnings=[],
        ts=int(time.time()),
    )

@app.get("/echo_auth", response_model=EchoAuthResponse)
async def echo_auth(x_api_key: Optional[str] = Header(None)):
    ok = True
    got_present = bool(x_api_key)
    got_len = len(x_api_key or "")
    exp_present = bool(EXPECTED_API_KEY)
    match = (x_api_key == EXPECTED_API_KEY) if exp_present else True
    return EchoAuthResponse(
        ok=ok, got_present=got_present, got_len=got_len, exp_present=exp_present, match=match
    )

@app.post("/inspect_draft", response_model=InspectDraftResponse)
async def inspect_draft(req: InspectDraftRequest, x_api_key: Optional[str] = Header(None)):
    require_api_key(x_api_key)

    input_used = req.dict(exclude_none=True)
    draft_state = {}
    my_team: List[Dict[str, Any]] = []

    picks: List[Dict[str, Any]] = []
    debug: List[str] = []

    if req.draft_url:
        did = await _draft_id_from_url(req.draft_url)
        draft_state, picks, ddbg = await _fetch_draft_bundle(did, req.league_id)
        debug += ddbg

    drafted_names = set(norm(n) for n in _names_from_picks(picks))
    drafted_count = len(drafted_names)

    csv_preview = (_rankings_rows or [])[:5]
    now = int(time.time())

    return InspectDraftResponse(
        status="ok",
        draft_state=draft_state,
        slot_to_roster_raw=None,
        slot_to_roster_normalized=None,
        observed_roster_ids=[],
        by_roster_counts={},
        input=input_used,
        effective_roster_id=req.roster_id,
        effective_team_slot=req.team_slot,
        my_team=my_team,
        drafted_count=len(picks),
        my_team_count=len(my_team),
        undrafted_count=None,
        csv_matched_count=None,
        csv_top_preview=csv_preview,
        ts=now,
    )

@app.post("/recommend_live", response_model=RecommendLiveResponse)
async def recommend_live(req: RecommendLiveRequest, x_api_key: Optional[str] = Header(None)):
    require_api_key(x_api_key)

    season = req.season or 2025
    limit = max(1, min(req.limit or 10, 25))
    debug: List[str] = []

    picks: List[Dict[str, Any]] = []
    draft_state: Dict[str, Any] = {"picks_made": 0}

    if req.draft_url:
        did = await _draft_id_from_url(req.draft_url)
        draft_state, picks, ddbg = await _fetch_draft_bundle(did, req.league_id)
        debug += ddbg

    drafted_names = set(norm(n) for n in _names_from_picks(picks))
    my_team: List[Dict[str, Any]] = []  # keep simple here

    # PRIMARY: CSV rankings filtered by undrafted
    ranked_pool = [r for r in _rankings_rows if norm(r.get("name")) not in drafted_names]
    if ranked_pool:
        debug.append(f"primary_count={len(ranked_pool)}")
    else:
        debug.append("primary_count=0")

    recommended = ranked_pool[:limit]

    # FALLBACK 1: unfiltered CSV
    if not recommended and _rankings_rows:
        debug.append("fallback1_primary_empty")
        recommended = [r for r in _rankings_rows if norm(r.get("name")) not in drafted_names][:limit]

    # FALLBACK 2: emergency list
    if not recommended:
        debug.append("fallback2_emergency_list")
        for r in _EMERGENCY_TOP:
            if norm(r["name"]) not in drafted_names:
                recommended.append(r)
            if len(recommended) >= limit:
                break

    # annotate
    pick_no = req.pick_number or draft_state.get("picks_made")
    for r in recommended:
        if "explain" not in r:
            r["explain"] = f"rank={r.get('rank_avg')}, pick {pick_no}"

    now = int(time.time())
    return RecommendLiveResponse(
        status="ok",
        pick=pick_no,
        season_used=season,
        recommended=recommended,
        alternatives=[],
        my_team=my_team,
        draft_state=draft_state,
        effective_roster_id=req.roster_id,
        drafted_count=len(drafted_names),
        my_team_count=len(my_team),
        debug_reason=debug,
        ts=now,
    )
