# server.py
from __future__ import annotations
import os, re, csv, json, time, math, asyncio
from typing import Any, Dict, List, Optional, Tuple
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

APP_VERSION = "1.0.7"

app = FastAPI(title="Fantasy Live Draft API")

# ==== auth ====
EXPECTED_API_KEY = os.getenv("API_KEY", "")

def require_api_key(x_api_key: Optional[str]):
    if not EXPECTED_API_KEY:
        return  # auth disabled
    if not x_api_key or x_api_key != EXPECTED_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

# ==== tiny Sleeper helpers ====
import httpx

SLEEPER = "https://api.sleeper.app/v1"

async def http_get_json(url: str) -> Any:
    async with httpx.AsyncClient(timeout=20) as cli:
        r = await cli.get(url)
        r.raise_for_status()
        return r.json()

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
    return _norm_rx.sub("", s.lower().strip()) if s else ""

def _parse_float(x: str) -> Optional[float]:
    try:
        x = (x or "").strip()
        if x in ("", "NA", "na", "NaN"):
            return None
        return float(x)
    except Exception:
        return None

def _load_rankings_from_csv() -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Reads rankings.csv and tolerates header casing variants like:
    "Player","Team","POS","AVG" or "player","team","pos","rank_avg"
    """
    path = os.getenv("RANKINGS_CSV", "rankings.csv")
    rows: List[Dict[str, Any]] = []
    index: Dict[str, Dict[str, Any]] = {}

    if not os.path.exists(path):
        return rows, index

    # support BOM and different header casings
    with open(path, newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        for raw in r:
            if raw is None:
                continue
            # normalize keys to lowercase and strip values
            row = { (k or "").strip().lower(): (v or "").strip() for k, v in raw.items() }

            name = row.get("name") or row.get("player") or row.get("player_name") or ""
            team = row.get("team") or row.get("tm") or ""
            pos  = row.get("pos") or row.get("position") or ""

            # Accept rank_avg, rank, or avg
            rank = _parse_float(row.get("rank_avg") or row.get("rank") or row.get("avg"))
            # Accept adp or a site-specific ADP column if you have it
            adp  = _parse_float(row.get("adp") or row.get("sleeper") or row.get("yahoo") or "")

            if not name:
                # skip truly empty lines so we don't return blank names
                continue

            rec = {
                "name": name,
                "team": team.upper() if team else team,
                "pos":  pos.upper() if pos else pos,
                "rank_avg": rank,
                "adp": adp,
                "proj_ros": None,
            }
            rows.append(rec)
            index[norm(name)] = rec

    # Sort by rank if available, push None to the end
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

# ==== utility: draft parsing (lightweight) ====

async def _draft_id_from_url(draft_url: str) -> str:
    # https://sleeper.com/draft/nfl/1263988228017369088
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

def _names_from_picks(picks: List[Dict[str, Any]]) -> List[str]:
    out = []
    for p in picks:
        name = None
        md = p.get("metadata") if isinstance(p, dict) else None
        if isinstance(md, dict):
            name = md.get("player_name") or md.get("first_name")
            if md.get("last_name"):
                name = f"{name} {md['last_name']}".strip() if name else md["last_name"]
        if not name:
            name = p.get("player_name")
        if name:
            out.append(name)
    return out

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
    drafted_count = 0

    picks = []
    if req.draft_url:
        did = await _draft_id_from_url(req.draft_url)
        picks = await _picks_for_draft_id(did)
        draft_state = {
            "draft_id": did,
            "league_id": None,
            "picks_made": len(picks),
        }

    my_names = []
    drafted_count = len(picks)
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
        drafted_count=drafted_count,
        my_team_count=len(my_names),
        undrafted_count=None,
        csv_matched_count=None,
        csv_top_preview=csv_preview,
        ts=now,
    )

@app.post("/recommend_live", response_model=RecommendLiveResponse)
async def recommend_live(req: RecommendLiveRequest, x_api_key: Optional[str] = Header(None)):
    """
    Always return ranked recommendations:
    - Use live picks to filter out drafted names when available.
    - If live is empty/flaky, fall back to CSV rankings.
    - Never return an empty list (emergency list last).
    """
    require_api_key(x_api_key)

    debug: List[str] = []
    season = req.season or 2025
    limit = max(1, min(req.limit or 10, 25))

    # Pull picks (if draft_url provided)
    picks: List[Dict[str, Any]] = []
    draft_state: Dict[str, Any] = {"picks_made": 0}
    did: Optional[str] = None

    if req.draft_url:
        did = await _draft_id_from_url(req.draft_url)
        picks = await _picks_for_draft_id(did)
        draft_state.update({
            "draft_id": did,
            "picks_made": len(picks),
        })

    # Build drafted set by normalized name
    drafted_names = set(norm(n) for n in _names_from_picks(picks))
    drafted_count = len(drafted_names)

    # My team (placeholder; can be enhanced later)
    my_team: List[Dict[str, Any]] = []

    # PRIMARY: rankings filtered by undrafted
    ranked_pool = [
        r for r in _rankings_rows
        if norm(r.get("name")) not in drafted_names
    ]

    debug.append(f"primary_count={len(ranked_pool)}")
    recommended = ranked_pool[:limit]

    # FALLBACK #1: if nothing, try the raw CSV order (still filtered vs drafted)
    if not recommended and _rankings_rows:
        debug.append("fallback1_primary_empty")
        recommended = [
            r for r in _rankings_rows
            if norm(r.get("name")) not in drafted_names
        ][:limit]

    # FALLBACK #2: emergency list
    if not recommended:
        debug.append("fallback2_emergency_list")
        rec = []
        for r in _EMERGENCY_TOP:
            if norm(r["name"]) not in drafted_names:
                rec.append(r)
            if len(rec) >= limit:
                break
        recommended = rec

    # Add simple explain
    pick_for_text = req.pick_number or draft_state.get("picks_made") or "NA"
    for r in recommended:
        if "explain" not in r or not r["explain"]:
            r["explain"] = f"rank={r.get('rank_avg')}, pick {pick_for_text}"

    now = int(time.time())
    return RecommendLiveResponse(
        status="ok",
        pick=req.pick_number or draft_state.get("picks_made"),
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
