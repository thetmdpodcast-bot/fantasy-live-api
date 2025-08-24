# server.py
from __future__ import annotations
import os, re, csv, json, time, math
from typing import Any, Dict, List, Optional, Tuple
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import httpx

APP_VERSION = "1.0.7"

app = FastAPI(title="Fantasy Live Draft API")

# ==== auth ==============================================================
EXPECTED_API_KEY = os.getenv("API_KEY", "")
def require_api_key(x_api_key: Optional[str]):
    if not EXPECTED_API_KEY:
        return  # auth disabled
    if not x_api_key or x_api_key != EXPECTED_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

# ==== tiny Sleeper helpers =============================================
SLEEPER = "https://api.sleeper.app/v1"

async def http_get_json(url: str) -> Any:
    async with httpx.AsyncClient(timeout=20) as cli:
        r = await cli.get(url)
        r.raise_for_status()
        return r.json()

async def _draft_id_from_url(draft_url: str) -> str:
    did = draft_url.strip().rstrip("/").split("/")[-1]
    if not did.isdigit():
        raise HTTPException(status_code=400, detail="Invalid draft_url")
    return did

async def _picks_for_draft_id(draft_id: str) -> List[Dict[str, Any]]:
    url = f"{SLEEPER}/draft/{draft_id}/picks"
    try:
        data = await http_get_json(url)
    except Exception:
        data = []
    return data or []

# Pull a display player name from a Sleeper pick object
def _pick_name(p: Dict[str, Any]) -> Optional[str]:
    if not isinstance(p, dict):
        return None
    md = p.get("metadata") or {}
    if isinstance(md, dict):
        # common fast path
        n = md.get("player_name")
        if n:
            return n
        # fallback build
        first = md.get("first_name") or ""
        last  = md.get("last_name") or ""
        if first or last:
            return (first + " " + last).strip()
    # final fallback
    return p.get("player_name")

def _names_from_picks(picks: List[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for p in picks:
        n = _pick_name(p)
        if n:
            out.append(n)
    return out

# ==== rankings cache ====================================================
_rankings_rows: List[Dict[str, Any]] = []
_rankings_index_by_norm: Dict[str, Dict[str, Any]] = {}
_last_rankings_merge_ts: Optional[int] = None

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
    return _norm_rx.sub("", s.lower().strip()) if s else ""

def _s(v: Any) -> str:
    """coerce any CSV value to a trimmed string (handles lists/None)"""
    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, list):
        return ", ".join(str(x) for x in v).strip()
    return str(v).strip()

def warm_rankings() -> None:
    """
    Load rankings.csv with forgiving headers:
      - allows 'name', 'player', 'Player'
      - rank from 'rank_avg', 'rank', or 'avg'
      - pos from 'pos' or 'position'
    """
    global _rankings_rows, _rankings_index_by_norm, _last_rankings_merge_ts
    path = os.getenv("RANKINGS_CSV", "rankings.csv")
    rows: List[Dict[str, Any]] = []
    index: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for raw in rdr:
                # lower-case keys safely
                lower = { (k or "").strip().lower(): _s(v)
                          for k, v in (raw.items() if raw else []) }
                name = lower.get("name") or lower.get("player") or lower.get("player_name") or ""
                team = lower.get("team") or lower.get("tm") or ""
                pos  = lower.get("pos") or lower.get("position") or ""
                rank_s = lower.get("rank_avg") or lower.get("rank") or lower.get("avg") or ""
                adp_s  = lower.get("adp") or ""
                try:
                    rank = float(rank_s) if rank_s not in ("", "NA") else None
                except:
                    rank = None
                try:
                    adp = float(adp_s) if adp_s not in ("", "NA") else None
                except:
                    adp = None
                if name:
                    rec = {"name": name, "team": team, "pos": pos,
                           "rank_avg": rank, "adp": adp, "proj_ros": None}
                    rows.append(rec)
                    index[norm(name)] = rec
    # stable order
    rows.sort(key=lambda d: (math.inf if d.get("rank_avg") in (None, "") else d["rank_avg"]))
    _rankings_rows = rows
    _rankings_index_by_norm = index
    _last_rankings_merge_ts = int(time.time())

# load once on boot
warm_rankings()

# ==== models ============================================================
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

class GuessRosterRequest(BaseModel):
    draft_url: str
    player_names: List[str]

class GuessRosterCandidate(BaseModel):
    roster_id: int
    matches: int
    players: List[str]

class GuessRosterResponse(BaseModel):
    status: str
    draft_id: str
    candidates: List[GuessRosterCandidate]

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

# ==== endpoints =========================================================
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
    return EchoAuthResponse(ok=ok, got_present=got_present, got_len=got_len,
                            exp_present=exp_present, match=match)

@app.post("/inspect_draft", response_model=InspectDraftResponse)
async def inspect_draft(req: InspectDraftRequest, x_api_key: Optional[str] = Header(None)):
    require_api_key(x_api_key)
    input_used = req.dict(exclude_none=True)
    draft_state: Dict[str, Any] = {}
    picks: List[Dict[str, Any]] = []

    if req.draft_url:
        did = await _draft_id_from_url(req.draft_url)
        picks = await _picks_for_draft_id(did)
        draft_state = {"draft_id": did, "league_id": None, "picks_made": len(picks)}

    my_names: List[str] = []  # keep minimal here
    now = int(time.time())
    csv_preview = (_rankings_rows or [])[:5]

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
        my_team=[{"name": n} for n in my_names],
        drafted_count=len(picks),
        my_team_count=len(my_names),
        undrafted_count=None,
        csv_matched_count=None,
        csv_top_preview=csv_preview,
        ts=now,
    )

@app.post("/guess_roster", response_model=GuessRosterResponse)
async def guess_roster(req: GuessRosterRequest, x_api_key: Optional[str] = Header(None)):
    require_api_key(x_api_key)
    did = await _draft_id_from_url(req.draft_url)
    picks = await _picks_for_draft_id(did)

    want = {norm(n): n for n in (req.player_names or []) if n}
    by_roster: Dict[int, List[str]] = {}
    for p in picks:
        rid = p.get("roster_id")
        if isinstance(rid, int):
            n = _pick_name(p)
            if n:
                by_roster.setdefault(rid, []).append(n)

    cands: List[GuessRosterCandidate] = []
    for rid, names in by_roster.items():
        names_norm = {norm(n) for n in names}
        hits = [pretty for key, pretty in want.items() if key in names_norm]
        if hits:
            cands.append(GuessRosterCandidate(roster_id=rid, matches=len(hits), players=hits))

    # order by matches desc, then roster id
    cands.sort(key=lambda c: (-c.matches, c.roster_id))
    return GuessRosterResponse(status="ok", draft_id=did, candidates=cands)

@app.post("/recommend_live", response_model=RecommendLiveResponse)
async def recommend_live(req: RecommendLiveRequest, x_api_key: Optional[str] = Header(None)):
    require_api_key(x_api_key)

    debug: List[str] = []
    season = req.season or 2025
    limit = max(1, min(req.limit or 10, 25))

    picks: List[Dict[str, Any]] = []
    draft_state: Dict[str, Any] = {"picks_made": 0}
    if req.draft_url:
        did = await _draft_id_from_url(req.draft_url)
        picks = await _picks_for_draft_id(did)
        draft_state.update({"draft_id": did, "picks_made": len(picks)})

    drafted_names = {norm(n) for n in _names_from_picks(picks)}
    drafted_count = len(drafted_names)
    my_team: List[Dict[str, Any]] = []  # you can fill from roster mapping later

    # PRIMARY: CSV-ranked pool excluding drafted (if we can match)
    ranked_pool = [r for r in _rankings_rows if norm(r.get("name")) not in drafted_names]
    if ranked_pool:
        debug.append(f"primary_count={len(ranked_pool)}")
        recommended = ranked_pool[:limit]
    else:
        debug.append("primary_count=0")
        recommended = []

    # FALLBACK 1: if primary is empty, take top CSV (ignore drafted mismatch)
    if not recommended and _rankings_rows:
        debug.append("fallback1_primary_empty")
        recommended = _rankings_rows[:limit]

    # FALLBACK 2: emergency static names
    if not recommended:
        debug.append("fallback2_emergency_list")
        rec = []
        for r in _EMERGENCY_TOP:
            if norm(r["name"]) not in drafted_names:
                rec.append(r)
            if len(rec) >= limit:
                break
        recommended = rec

    pick_no = req.pick_number or draft_state.get("picks_made")
    for r in recommended:
        if "explain" not in r:
            r["explain"] = f"rank={r.get('rank_avg')}, pick {pick_no if pick_no is not None else 'NA'}"

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
        drafted_count=drafted_count,
        my_team_count=len(my_team),
        debug_reason=debug,
        ts=now,
    )
