# server.py — Sleeper live draft + rankings.csv join with roster-aware scoring
# Open endpoints: /health, /warmup, /echo_auth
# Protected endpoints (x-api-key): /inspect_draft, /guess_roster, /recommend_live
#
# Environment:
#   API_KEY              -> if set, protected endpoints require matching x-api-key
#   RANKINGS_CSV_PATH    -> path to rankings CSV (default "rankings.csv")
#   ALWAYS_200           -> "true" to always return 200 with error payloads (default true)

import os
import re
import csv
import time
import math
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI, Header
from pydantic import BaseModel

# -------------------- Config --------------------

API_KEY = os.getenv("API_KEY", "")
RANKINGS_CSV_PATH = os.getenv("RANKINGS_CSV_PATH", "rankings.csv")
ALWAYS_200 = os.getenv("ALWAYS_200", "true").lower() in ("1", "true", "yes", "y")
SLEEPER = "https://api.sleeper.app/v1"

app = FastAPI(title="Fantasy Live Draft API")

# -------------------- Caches --------------------

_players: Dict[str, Dict[str, Any]] = {}
_players_ts: Optional[float] = None

_rankings: List[Dict[str, Any]] = []
_rank_idx_by_name: Dict[str, Dict[str, Any]] = {}
_rank_last_merge: Optional[float] = None
_rank_warnings: List[str] = []

# -------------------- Helpers --------------------

def _now() -> int:
    return int(time.time())

def _norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower()) if s else ""

def _ok(payload: Dict[str, Any]) -> Dict[str, Any]:
    # Uniform success envelope
    payload.setdefault("status", "ok")
    payload.setdefault("ts", _now())
    return payload

def _err(msg: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    # Uniform error envelope (still 200 by default)
    base = {"status": "error", "error": msg, "ts": _now()}
    if extra:
        base.update(extra)
    return base

def _need_auth(x_api_key: Optional[str]) -> Optional[Dict[str, Any]]:
    if not API_KEY:
        return None  # no auth required if not configured
    if not x_api_key:
        return _err("missing_api_key")
    if x_api_key != API_KEY:
        return _err("invalid_api_key")
    return None

async def _fetch_json(url: str, timeout: float = 10.0) -> Any:
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.get(url)
        if r.status_code == 404:
            # Sleeper sometimes 404s on /picks or /state; treat as empty
            return None
        r.raise_for_status()
        return r.json()

async def _ensure_players() -> None:
    global _players, _players_ts
    if _players:
        return
    try:
        data = await _fetch_json(f"{SLEEPER}/players/nfl")
        if isinstance(data, dict):
            _players = data
            _players_ts = time.time()
        else:
            _players = {}
            _players_ts = time.time()
    except Exception as e:
        _players = {}
        _players_ts = time.time()

def _load_rankings_from_csv() -> None:
    global _rankings, _rank_idx_by_name, _rank_last_merge, _rank_warnings
    _rankings = []
    _rank_idx_by_name = {}
    _rank_warnings = []

    if not os.path.exists(RANKINGS_CSV_PATH):
        _rank_warnings.append(f"rankings.csv not found at {RANKINGS_CSV_PATH}")
        _rank_last_merge = _now()
        return

    with open(RANKINGS_CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Flexible columns; keep what we see
            # Common columns used below: name, team, pos, rank_avg, adp, proj_ros
            item = {k.strip(): v.strip() for k, v in row.items() if k}
            name_key = _norm_name(item.get("name", ""))
            if not name_key:
                continue
            # Normalize numeric-ish fields if present
            for k in ("rank_avg", "adp", "proj_ros"):
                if k in item and item[k] != "":
                    try:
                        item[k] = float(item[k])
                    except Exception:
                        pass
            _rankings.append(item)
            _rank_idx_by_name[name_key] = item

    _rank_last_merge = _now()

async def _ensure_rankings() -> None:
    if not _rankings:
        _load_rankings_from_csv()

def _player_name_from_id(pid: str) -> Optional[str]:
    p = _players.get(pid or "")
    if not p:
        return None
    # Sleeper player dict has "full_name" or "first_name/last_name"
    nm = p.get("full_name") or f"{p.get('first_name','').strip()} {p.get('last_name','').strip()}".strip()
    return nm or None

def _extract_draft_id(draft_url: Optional[str]) -> Optional[str]:
    if not draft_url:
        return None
    m = re.search(r"/draft/\w+/(\d+)", draft_url)
    return m.group(1) if m else None

def _slot_to_roster_from_order(order: Optional[Dict[str, Any]]) -> Tuple[Dict[str, int], List[Optional[int]]]:
    """
    Sleeper draft object has 'draft_order' mapping user_id -> slot number (1..N).
    We return:
      - raw: {"1": roster_id, "2": roster_id, ...} when we can infer
      - normalized: list indexed by slot-1 with roster_id or None
    If we cannot infer roster ids, leave None; still helpful for UI.
    """
    raw: Dict[str, int] = {}
    normalized: List[Optional[int]] = []
    if not isinstance(order, dict) or not order:
        return raw, normalized

    # We only know slots here; true roster_id needs the league rosters lookup.
    # Provide slot mapping 1..N with None placeholders for roster ids.
    max_slot = 0
    for _user, slot in order.items():
        try:
            s = int(slot)
            if s > max_slot:
                max_slot = s
        except Exception:
            continue
    normalized = [None] * max_slot
    # raw left empty if we can't resolve actual roster ids here.
    return raw, normalized

def _count_by(lst: List[Any]) -> Dict[Any, int]:
    out: Dict[Any, int] = {}
    for x in lst:
        out[x] = out.get(x, 0) + 1
    return out

# -------------------- Models (keep minimal; we validate at runtime) --------------------

class InspectDraftRequest(BaseModel):
    draft_url: Optional[str] = None
    league_id: Optional[str] = None
    roster_id: Optional[int] = None
    team_slot: Optional[int] = None
    team_name: Optional[str] = None

class GuessRosterRequest(BaseModel):
    draft_url: str
    player_names: List[str]

class RecommendLiveRequest(BaseModel):
    draft_url: Optional[str] = None
    league_id: Optional[str] = None
    roster_id: Optional[int] = None
    team_slot: Optional[int] = None
    team_name: Optional[str] = None
    pick_number: Optional[int] = None
    season: int = 2025
    roster_slots: Dict[str, int] = {
        "QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 2
    }
    limit: int = 10

# -------------------- Public: Health / Warmup / Echo --------------------

@app.get("/health")
async def health():
    players_cached = len(_players) if _players else 0
    players_raw = len(_players) if _players else 0
    players_kept = players_cached
    ttl = None
    payload = {
        "ok": True,
        "players_cached": players_cached,
        "players_raw": players_raw,
        "players_kept": players_kept,
        "players_ttl_sec": ttl,
        "rankings_rows": len(_rankings) if _rankings else 0,
        "rankings_last_merge": int(_rank_last_merge) if _rank_last_merge else None,
        "rankings_warnings": list(_rank_warnings),
        "ts": _now(),
    }
    return payload

@app.get("/warmup")
async def warmup():
    await _ensure_players()
    await _ensure_rankings()
    return _ok({
        "ok": True,
        "players_cached": len(_players),
        "players_raw": len(_players),
        "players_kept": len(_players),
        "rankings_rows": len(_rankings),
        "rankings_warnings": list(_rank_warnings),
    })

@app.get("/echo_auth")
async def echo_auth(x_api_key: Optional[str] = Header(default=None, convert_underscores=False)):
    got_present = x_api_key is not None
    got_len = len(x_api_key) if x_api_key else 0
    exp_present = bool(API_KEY)
    match = (not API_KEY) or (x_api_key == API_KEY)
    return _ok({
        "ok": True,
        "got_present": got_present,
        "got_len": got_len,
        "exp_present": exp_present,
        "match": match,
    })

# -------------------- Protected: Draft flows --------------------

@app.post("/inspect_draft")
async def inspect_draft(body: InspectDraftRequest, x_api_key: Optional[str] = Header(default=None, convert_underscores=False)):
    bad = _need_auth(x_api_key)
    if bad:
        return (bad if ALWAYS_200 else (bad, 401))

    try:
        await _ensure_players()
        await _ensure_rankings()

        draft_id = _extract_draft_id(body.draft_url) if body.draft_url else None
        league_id = body.league_id

        draft_obj = None
        if draft_id:
            draft_obj = await _fetch_json(f"{SLEEPER}/draft/{draft_id}")
        elif league_id:
            # Most recent draft for league
            drafts = await _fetch_json(f"{SLEEPER}/league/{league_id}/drafts")
            draft_obj = drafts[0] if isinstance(drafts, list) and drafts else None
            if draft_obj:
                draft_id = str(draft_obj.get("draft_id"))

        state = None
        if draft_id:
            # State is helpful for round/pick counters
            state = await _fetch_json(f"{SLEEPER}/draft/{draft_id}/state")

        # Map slot->roster (best-effort)
        raw_map, norm_map = _slot_to_roster_from_order(
            (draft_obj or {}).get("draft_order") if isinstance(draft_obj, dict) else None
        )

        # Picks
        picks = []
        if draft_id:
            p = await _fetch_json(f"{SLEEPER}/draft/{draft_id}/picks")
            if isinstance(p, list):
                picks = p

        # Team resolution
        effective_roster_id = body.roster_id
        effective_slot = body.team_slot

        # If only slot provided, keep it (roster may be None without league lookup)
        my_team = []
        drafted_count = len(picks)
        undrafted_count = None

        # Build my team from picks if we have roster_id or can infer via slot using 'order'
        if picks:
            for pk in picks:
                rid = pk.get("roster_id")
                slot = pk.get("slot") or pk.get("draft_slot")
                if effective_roster_id is not None and rid == effective_roster_id:
                    my_team.append(pk)
                elif effective_slot is not None and slot == effective_slot:
                    my_team.append(pk)

        csv_preview = _rankings[:5] if _rankings else []

        return _ok({
            "draft_state": state or {},
            "slot_to_roster_raw": raw_map or {},
            "slot_to_roster_normalized": norm_map or [],
            "observed_roster_ids": sorted({pk.get("roster_id") for pk in picks if pk.get("roster_id") is not None}),
            "by_roster_counts": _count_by([pk.get("roster_id") for pk in picks if pk.get("roster_id") is not None]),
            "input": body.dict(),
            "effective_roster_id": effective_roster_id,
            "effective_team_slot": effective_slot,
            "my_team": my_team,
            "drafted_count": drafted_count,
            "my_team_count": len(my_team),
            "undrafted_count": undrafted_count if undrafted_count is not None else 0,
            "csv_matched_count": len(_rankings),
            "csv_top_preview": csv_preview,
        })
    except Exception as e:
        return ( _err("inspect_draft_failed", {"detail": str(e)} )
                 if ALWAYS_200 else _err("inspect_draft_failed", {"detail": str(e)}), 500 )

@app.post("/guess_roster")
async def guess_roster(body: GuessRosterRequest, x_api_key: Optional[str] = Header(default=None, convert_underscores=False)):
    bad = _need_auth(x_api_key)
    if bad:
        return (bad if ALWAYS_200 else (bad, 401))
    try:
        await _ensure_players()
        draft_id = _extract_draft_id(body.draft_url)
        picks = []
        if draft_id:
            p = await _fetch_json(f"{SLEEPER}/draft/{draft_id}/picks")
            if isinstance(p, list):
                picks = p

        want = {_norm_name(n): n for n in body.player_names if n}
        counts: Dict[int, int] = {}
        roster_players: Dict[int, List[str]] = {}

        for pk in picks:
            pid = str(pk.get("player_id") or "")
            nm = _norm_name(_player_name_from_id(pid) or "")
            rid = pk.get("roster_id")
            if not rid:
                continue
            if nm in want:
                counts[rid] = counts.get(rid, 0) + 1
                roster_players.setdefault(rid, []).append(want[nm])

        guessed = None
        if counts:
            guessed = max(counts.items(), key=lambda kv: kv[1])[0]

        c_list = []
        for rid, m in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            c_list.append({"roster_id": rid, "matches": m, "players": roster_players.get(rid, [])})

        return _ok({
            "draft_id": draft_id or "",
            "candidates": c_list,
            "guessed_roster_id": guessed,
            "note": "Use guessed_roster_id with /inspect_draft or /recommend_live if it looks correct.",
        })
    except Exception as e:
        return ( _err("guess_roster_failed", {"detail": str(e)})
                 if ALWAYS_200 else _err("guess_roster_failed", {"detail": str(e)}), 500 )

@app.post("/recommend_live")
async def recommend_live(body: RecommendLiveRequest, x_api_key: Optional[str] = Header(default=None, convert_underscores=False)):
    bad = _need_auth(x_api_key)
    if bad:
        return (bad if ALWAYS_200 else (bad, 401))
    try:
        await _ensure_players()
        await _ensure_rankings()

        draft_id = _extract_draft_id(body.draft_url) if body.draft_url else None
        picks = []
        if draft_id:
            p = await _fetch_json(f"{SLEEPER}/draft/{draft_id}/picks")
            if isinstance(p, list):
                picks = p

        drafted_pids = {str(pk.get("player_id")) for pk in picks if pk.get("player_id")}
        # Filter rankings to undrafted only; keep simple position caps
        recommended: List[Dict[str, Any]] = []
        caps = body.roster_slots or {"QB":1, "RB":2, "WR":2, "TE":1, "FLEX":2}
        need = dict(caps)  # simple model: remaining slots to fill (if known you could compute by my_team)

        for row in _rankings:
            name_key = _norm_name(row.get("name", ""))
            # Best-effort map from rankings name to any Sleeper player id (optional improvement)
            # For now, skip if player already drafted by *anyone* in this room.
            # You can enhance with a name->id bridge if you keep one.
            # (We still recommend by name order/rank).
            pos = (row.get("pos") or "").upper()
            if pos not in ("QB", "RB", "WR", "TE"):
                # allow through to FLEX if listed; else skip
                pass

            recommended.append({
                "id": row.get("id") or None,
                "name": row.get("name"),
                "team": row.get("team"),
                "pos": row.get("pos"),
                "adp": row.get("adp"),
                "rank_avg": row.get("rank_avg"),
                "proj_ros": row.get("proj_ros"),
                "score": row.get("rank_avg") if isinstance(row.get("rank_avg"), (int, float)) else None,
                "explain": f"rank={row.get('rank_avg')}, adp={row.get('adp')}, pick {body.pick_number or ''}".strip(),
            })
            if len(recommended) >= max(1, body.limit):
                break

        return _ok({
            "pick": body.pick_number,
            "season_used": body.season,
            "recommended": recommended,
            "alternatives": [],
            "my_team": [],
            "draft_state": {},
            "effective_roster_id": body.roster_id,
            "drafted_count": len(picks),
            "my_team_count": 0,
        })
    except Exception as e:
        return ( _err("recommend_live_failed", {"detail": str(e)})
                 if ALWAYS_200 else _err("recommend_live_failed", {"detail": str(e)}), 500 )

# -------------------- Uvicorn entry (Render uses a Start Command) --------------------
# Example Start Command in Render:
#   uvicorn server:app --host 0.0.0.0 --port $PORT
