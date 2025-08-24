# server.py
from __future__ import annotations

import os, re, csv, json, time, math, asyncio
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import httpx

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

def _to_str(v: Any) -> str:
    """Safe string for CSV values; flattens lists/None."""
    if v is None:
        return ""
    if isinstance(v, list):
        return ",".join(_to_str(x) for x in v)
    return str(v)

def _to_float(v: Any) -> Optional[float]:
    s = _to_str(v).strip()
    if s in ("", "NA", "na", "N/A", "null"):
        return None
    try:
        return float(s)
    except:
        return None

def _load_rankings_from_csv() -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Accepts flexible headers (case-insensitive, spaces OK):
      - name:  name | player | player_name
      - team:  team | tm | nfl_team
      - pos:   pos | position
      - rank:  rank_avg | avg | rank | overall
      - adp:   adp
    """
    path = os.getenv("RANKINGS_CSV", "rankings.csv")
    rows: List[Dict[str, Any]] = []
    index: Dict[str, Dict[str, Any]] = {}

    if not os.path.exists(path):
        return rows, index

    with open(path, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f, skipinitialspace=True)
        for raw in rdr:
            # normalize keys and values safely
            lowered: Dict[str, str] = {}
            for k, v in (raw or {}).items():
                key = _to_str(k).strip().lower()
                lowered[key] = _to_str(v).strip()

            # header synonyms
            name = lowered.get("name") or lowered.get("player") or lowered.get("player_name") or ""
            team = lowered.get("team") or lowered.get("tm") or lowered.get("nfl_team") or ""
            pos  = lowered.get("pos")  or lowered.get("position") or ""

            # rank fields - prefer rank_avg/avg if present
            rank = (
                _to_float(lowered.get("rank_avg"))
                or _to_float(lowered.get("avg"))
                or _to_float(lowered.get("rank"))
                or _to_float(lowered.get("overall"))
            )
            adp = _to_float(lowered.get("adp"))

            if not name:
                # skip rows without a player name
                continue

            rec = {
                "name": name,
                "team": team,
                "pos": pos,
                "rank_avg": rank,
                "adp": adp,
                "proj_ros": None,
            }
            rows.append(rec)
            index[norm(name)] = rec

    # Sort by rank if available
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
        md = p.get("metadata") if isinstance(p, dict) else None
        name = None
        if isinstance(md, dict):
            name = md.get("player_name")
            # sometimes split first/last fields
            first = md.get("first_name")
            last = md.get("last_name")
            if not name and (first or last):
                name = f"{first or ''} {last or ''}".strip()
        if not name:
            name = p.get("player_name")
        if name:
            out.append(name)
    return out

# ==== endpoints ====
@app.get("/health", response_model=HealthResponse)
async def health():
    now
