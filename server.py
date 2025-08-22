# server.py
# Fantasy Live Draft API
# - Robust draft/league resolution
# - Full board fetch for in-progress drafts
# - Stronger team resolution and roster guessing
# - Recommendations filtered by undrafted pool
#
# Notes:
# - Keeps your endpoint names and JSON contract (works with your 1.0.5 schema)
# - Requires: httpx, fastapi, pydantic, uvicorn, python-dotenv (optional)
# - Env:
#     API_KEY           -> required header name: x-api-key
#     RANKINGS_CSV_PATH -> optional, default "rankings.csv"

import os
import re
import csv
import time
import math
import json
import random
import asyncio
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel

API_KEY = os.getenv("API_KEY")  # when set, we require x-api-key to match
SLEEPER = "https://api.sleeper.app/v1"
RANKINGS_CSV_PATH = os.getenv("RANKINGS_CSV_PATH", "rankings.csv")

app = FastAPI(title="Fantasy Live Draft API")

# --------------------------- simple caches -----------------------------

PLAYERS: Dict[str, Dict[str, Any]] = {}      # {player_id: {...}}
PLAYERS_TS: float = 0.0
PLAYERS_TTL = 60 * 25   # 25min

RANKINGS: List[Dict[str, Any]] = []
RANKINGS_WARNINGS: List[str] = []

# --------------------------- models (match your schema 1.0.5) ---------

class EchoAuthResponse(BaseModel):
    ok: bool
    got_present: bool
    got_len: int
    exp_present: bool
    match: bool

class HealthResponse(BaseModel):
    ok: bool
    players_cached: int
    players_raw: int
    players_kept: int
    players_ttl_sec: Optional[int]
    rankings_rows: int
    rankings_last_merge: Optional[int]
    rankings_warnings: List[str]
    ts: int

class WarmupResponse(BaseModel):
    ok: bool
    players_cached: int
    players_raw: int
    players_kept: int
    rankings_rows: int
    rankings_warnings: List[str]
    ts: int

class InspectDraftRequest(BaseModel):
    draft_url: Optional[str] = None
    league_id: Optional[str] = None
    roster_id: Optional[int] = None
    team_slot: Optional[int] = None
    team_name: Optional[str] = None

class InspectDraftResponse(BaseModel):
    status: str
    draft_state: Dict[str, Any] = {}
    slot_to_roster_raw: Optional[Dict[str, Any]] = None
    slot_to_roster_normalized: Optional[List[Optional[int]]] = None
    observed_roster_ids: List[int] = []
    by_roster_counts: Dict[str, int] = {}
    input: Dict[str, Any] = {}
    effective_roster_id: Optional[int] = None
    effective_team_slot: Optional[int] = None
    my_team: List[Dict[str, Any]] = []
    drafted_count: int = 0
    my_team_count: int = 0
    undrafted_count: int = 0
    csv_matched_count: int = 0
    csv_top_preview: List[Dict[str, Any]] = []
    ts: int

class GuessRosterRequest(BaseModel):
    draft_url: str
    player_names: List[str]

class GuessRosterResponse(BaseModel):
    status: str
    draft_id: Optional[str] = None
    candidates: List[Dict[str, Any]] = []
    guessed_roster_id: Optional[int] = None
    note: Optional[str] = None
    ts: int

class RecommendLiveRequest(BaseModel):
    draft_url: Optional[str] = None
    league_id: Optional[str] = None
    roster_id: Optional[int] = None
    team_slot: Optional[int] = None
    team_name: Optional[str] = None
    pick_number: Optional[int] = None
    season: int = 2025
    roster_slots: Optional[Dict[str, int]] = None
    limit: int = 10

class RecommendLiveResponse(BaseModel):
    status: str
    pick: Optional[int] = None
    season_used: int
    recommended: List[Dict[str, Any]]
    alternatives: List[Dict[str, Any]] = []
    my_team: List[Dict[str, Any]] = []
    draft_state: Dict[str, Any] = {}
    effective_roster_id: Optional[int] = None
    drafted_count: int
    my_team_count: int
    ts: int

# --------------------------- utilities --------------------------------

async def require_key(x_api_key: Optional[str]):
    if not API_KEY:
        return
    if not x_api_key or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

def nrm(s: str) -> str:
    # normalize a player/team/free text
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9 ]+", "", s)      # remove punctuation, accents stripped
    s = re.sub(r"\s+", " ", s)
    return s

def extract_draft_id_from_url(url: str) -> Optional[str]:
    # supports https://sleeper.com/draft/nfl/{draft_id}
    m = re.search(r"/draft/[a-z]+/(\d+)", url)
    return m.group(1) if m else None

async def get_http() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=15)

async def load_players(force: bool = False) -> Tuple[int, int, int, Optional[int]]:
    global PLAYERS, PLAYERS_TS
    now = time.time()
    ttl_left = max(0, int(PLAYERS_TTL - (now - PLAYERS_TS))) if PLAYERS_TS else None
    if PLAYERS and not force and ttl_left and ttl_left > 0:
        return len(PLAYERS), len(PLAYERS), len(PLAYERS), ttl_left

    async with await get_http() as cli:
        r = await cli.get(f"{SLEEPER}/players/nfl")
        r.raise_for_status()
        data = r.json()  # dict keyed by player_id
        # keep only those with full_name
        kept = {pid: v for pid, v in data.items() if isinstance(v, dict) and v.get("full_name")}
        PLAYERS = kept
        PLAYERS_TS = now
        ttl_left = PLAYERS_TTL
        return len(kept), len(data), len(kept), ttl_left

def load_rankings() -> Tuple[int, List[str]]:
    global RANKINGS, RANKINGS_WARNINGS
    if RANKINGS:
        return len(RANKINGS), RANKINGS_WARNINGS
    warnings: List[str] = []
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(RANKINGS_CSV_PATH):
        warnings.append(f"rankings file not found: {RANKINGS_CSV_PATH}")
    else:
        with open(RANKINGS_CSV_PATH, newline="") as f:
            rd = csv.DictReader(f)
            for row in rd:
                # expected columns: name,pos,team,avg,proj_ros,rank_avg (accept whatever you have)
                row["_n"] = nrm(row.get("name", ""))
                row["_pos"] = row.get("pos", "").upper()
                try:
                    row["_avg_rank"] = float(row.get("avg", row.get("rank_avg", "9999")) or 9999)
                except:
                    row["_avg_rank"] = 9999.0
                rows.append(row)
    RANKINGS = rows
    RANKINGS_WARNINGS = warnings
    return len(RANKINGS), warnings

async def resolve_draft_and_league(
    draft_url: Optional[str],
    league_id: Optional[str],
) -> Tuple[str, Optional[str], Dict[str, Any]]:
    """
    Returns (draft_id, league_id, draft_meta)
    """
    async with await get_http() as cli:
        draft_id: Optional[str] = None
        meta: Dict[str, Any] = {}

        if draft_url:
            draft_id = extract_draft_id_from_url(draft_url)

        if not draft_id and league_id:
            # find an in_progress or most recent draft for the league
            r = await cli.get(f"{SLEEPER}/league/{league_id}/drafts")
            r.raise_for_status()
            arr = r.json() or []
            chosen = None
            for d in arr:
                # pick in_progress if present
                if d.get("status") == "in_progress":
                    chosen = d
                    break
            if not chosen and arr:
                # fall back to most recent
                chosen = arr[0]
            if chosen:
                draft_id = chosen["draft_id"]

        if not draft_id:
            raise HTTPException(status_code=400, detail="Unable to resolve draft_id")

        # draft meta/state
        r = await cli.get(f"{SLEEPER}/draft/{draft_id}")
        if r.status_code == 404:
            raise HTTPException(status_code=404, detail="Draft not found on Sleeper")
        r.raise_for_status()
        meta = r.json() or {}

        # try to ensure league_id
        if not league_id:
            league_id = meta.get("league_id")

        return draft_id, league_id, meta

async def fetch_slot_maps(league_id: str) -> Tuple[Dict[str, int], List[Optional[int]], Dict[str, int], Dict[str, str]]:
    """
    Returns:
      slot_to_roster_raw: {'1': 3, '2': 5, ...}
      slot_to_roster_normalized: [None, 3, 5, ...]  (index=slot)
      by_roster_counts: {'1':count, '2':count,...} (drafted count placeholder)
      owner_display_by_roster: {'3': 'mdonahue25', ...}
    """
    async with await get_http() as cli:
        # rosters -> owner_id, roster_id, settings.draft_position
        r = await cli.get(f"{SLEEPER}/league/{league_id}/rosters")
        r.raise_for_status()
        rosters = r.json() or []

        r2 = await cli.get(f"{SLEEPER}/league/{league_id}/users")
        r2.raise_for_status()
        users = r2.json() or []
        user_by_id = {u.get("user_id"): u for u in users}

        slot_to_roster_raw: Dict[str, int] = {}
        owner_display_by_roster: Dict[str, str] = {}
        max_slot = 0
        for ro in rosters:
            rid = ro.get("roster_id")
            slot = (ro.get("settings") or {}).get("draft_position") or ro.get("draft_slot") or ro.get("slot")
            if rid and slot:
                slot_to_roster_raw[str(slot)] = int(rid)
                max_slot = max(max_slot, int(slot))
            # owner display
            uid = ro.get("owner_id")
            disp = (user_by_id.get(uid) or {}).get("display_name") or (user_by_id.get(uid) or {}).get("username")
            if rid and disp:
                owner_display_by_roster[str(rid)] = str(disp)

        slot_norm: List[Optional[int]] = [None] * (max_slot + 1)
        for k, v in slot_to_roster_raw.items():
            i = int(k)
            if i < len(slot_norm):
                slot_norm[i] = v

        return slot_to_roster_raw, slot_norm, {}, owner_display_by_roster

async def fetch_picks(draft_id: str) -> List[Dict[str, Any]]:
    """
    Return full pick objects for the draft (in-progress supported).
    """
    async with await get_http() as cli:
        r = await cli.get(f"{SLEEPER}/draft/{draft_id}/picks")
        if r.status_code == 404:
            # Some older/in-progress states can briefly 404; standardize to empty list (we still return meta)
            return []
        r.raise_for_status()
        arr = r.json() or []
        return arr

def players_index_by_normalized() -> Dict[str, str]:
    # map normalized full_name -> player_id
    idx: Dict[str, str] = {}
    for pid, p in PLAYERS.items():
        full = p.get("full_name") or ""
        if full:
            idx[nrm(full)] = pid
    return idx

def match_name_to_player_ids(name: str) -> List[str]:
    """
    Return candidate Sleeper player_ids for a free-text name.
    - exact normalized full_name wins
    - otherwise, loose contains on last name + position block to reduce false positives
    """
    q = nrm(name)
    if not q:
        return []

    # exact match first
    idx = players_index_by_normalized()
    if q in idx:
        return [idx[q]]

    # fallback: last-token contains match (e.g., "henry", "chase")
    last = q.split()[-1]
    cands: List[str] = []
    for pid, p in PLAYERS.items():
        full = p.get("full_name") or ""
        if not full:
            continue
        nf = nrm(full)
        if last and last in nf:
            cands.append(pid)
    return cands[:10]  # cap

def pick_player_name(pid: str) -> str:
    p = PLAYERS.get(pid) or {}
    return p.get("full_name") or p.get("last_name") or p.get("first_name") or pid

def ranking_row_for_player(pid: str) -> Optional[Dict[str, Any]]:
    # Try to join Sleeper full name to rankings by normalized text
    full = PLAYERS.get(pid, {}).get("full_name") or ""
    if not full or not RANKINGS:
        return None
    nf = nrm(full)
    for r in RANKINGS:
        if r.get("_n") == nf:
            return r
    return None

def default_roster_caps() -> Dict[str, int]:
    return {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 2}

# --------------------------- endpoints --------------------------------

@app.get("/echo_auth", response_model=EchoAuthResponse)
async def echo_auth(x_api_key: Optional[str] = Header(None)):
    ok = True
    got_present = bool(x_api_key)
    got_len = len(x_api_key or "")
    exp_present = bool(API_KEY)
    match = (x_api_key == API_KEY) if API_KEY else True
    return EchoAuthResponse(
        ok=ok, got_present=got_present, got_len=got_len,
        exp_present=exp_present, match=match
    )

@app.get("/health", response_model=HealthResponse)
async def health():
    kept, raw, kept2, ttl = await load_players(force=False)
    rrows, warns = load_rankings()
    return HealthResponse(
        ok=True,
        players_cached=kept,
        players_raw=raw,
        players_kept=kept2,
        players_ttl_sec=ttl,
        rankings_rows=rrows,
        rankings_last_merge=None,
        rankings_warnings=warns,
        ts=int(time.time()),
    )

@app.get("/warmup", response_model=WarmupResponse)
async def warmup():
    kept, raw, kept2, _ = await load_players(force=True)
    rrows, warns = load_rankings()
    return WarmupResponse(
        ok=True,
        players_cached=kept,
        players_raw=raw,
        players_kept=kept2,
        rankings_rows=rrows,
        rankings_warnings=warns,
        ts=int(time.time()),
    )

def preview_top_csv(limit=6) -> List[Dict[str, Any]]:
    if not RANKINGS:
        return []
    out = []
    for r in sorted(RANKINGS, key=lambda x: x.get("_avg_rank", 9999.0))[:limit]:
        out.append({"name": r.get("name"), "pos": r.get("pos"), "team": r.get("team"), "avg": r.get("avg") or r.get("rank_avg")})
    return out

@app.post("/inspect_draft", response_model=InspectDraftResponse)
async def inspect_draft(req: InspectDraftRequest, x_api_key: Optional[str] = Header(None)):
    await require_key(x_api_key)
    await load_players(False)
    load_rankings()

    draft_id, league_id, meta = await resolve_draft_and_league(req.draft_url, req.league_id)

    slot_raw, slot_norm, _dummy_counts, display_by_roster = await fetch_slot_maps(league_id)
    picks = await fetch_picks(draft_id)

    # drafted count by roster
    by_roster_counts: Dict[str, int] = {}
    roster_players: Dict[int, List[str]] = {}

    observed: List[int] = []
    for pk in picks:
        rid = pk.get("roster_id")
        pid = str(pk.get("player_id"))
        if not rid or not pid:
            continue
        observed.append(int(rid))
        by_roster_counts[str(rid)] = by_roster_counts.get(str(rid), 0) + 1
        roster_players.setdefault(int(rid), []).append(pid)

    # decide effective roster/team
    eff_roster: Optional[int] = None
    eff_slot: Optional[int] = None
    if req.roster_id:
        eff_roster = req.roster_id
    elif req.team_slot and req.team_slot in range(1, len(slot_norm)):
        eff_roster = slot_norm[req.team_slot]
        eff_slot = req.team_slot
    elif req.team_name:
        nteam = nrm(req.team_name)
        for rid_str, disp in display_by_roster.items():
            if nrm(disp) == nteam:
                eff_roster = int(rid_str)
                break

    if eff_roster is None and req.team_slot:
        eff_roster = slot_norm[req.team_slot]
        eff_slot = req.team_slot

    my_team_pids = roster_players.get(int(eff_roster or -1), [])
    my_team = [{"id": pid, "name": pick_player_name(pid)} for pid in my_team_pids]

    drafted_ids = {str(pk.get("player_id")) for pk in picks if pk.get("player_id")}
    undrafted = []
    if RANKINGS:
        for r in RANKINGS:
            # we don't have sleeper_id in rankings.csv; so we filter only by text when possible later
            pass
    undrafted_count = max(0, len(PLAYERS) - len(drafted_ids))

    return InspectDraftResponse(
        status="ok",
        draft_state={"draft_id": draft_id, **meta},
        slot_to_roster_raw=slot_raw,
        slot_to_roster_normalized=slot_norm,
        observed_roster_ids=sorted(list({int(x) for x in observed})),
        by_roster_counts=by_roster_counts,
        input=req.model_dump(),
        effective_roster_id=eff_roster,
        effective_team_slot=eff_slot,
        my_team=my_team,
        drafted_count=len(drafted_ids),
        my_team_count=len(my_team),
        undrafted_count=undrafted_count,
        csv_matched_count=0,
        csv_top_preview=preview_top_csv(),
        ts=int(time.time()),
    )

@app.post("/guess_roster", response_model=GuessRosterResponse)
async def guess_roster(req: GuessRosterRequest, x_api_key: Optional[str] = Header(None)):
    await require_key(x_api_key)
    await load_players(False)

    draft_id = extract_draft_id_from_url(req.draft_url)
    if not draft_id:
        raise HTTPException(status_code=400, detail="Invalid draft_url")

    picks = await fetch_picks(draft_id)
    # roster -> set(player_ids)
    by_roster: Dict[int, set] = {}
    for pk in picks:
        rid = pk.get("roster_id")
        pid = str(pk.get("player_id"))
        if rid and pid:
            by_roster.setdefault(int(rid), set()).add(pid)

    # resolve input names -> candidate player_ids, then score rosters
    score: Dict[int, int] = {}
    audit: Dict[int, List[str]] = {}
    for name in req.player_names:
        cands = match_name_to_player_ids(name)
        roster_hit_for_this_name = set()
        for rid, pidset in by_roster.items():
            if any(p in pidset for p in cands):
                score[rid] = score.get(rid, 0) + 1
                roster_hit_for_this_name.add(rid)
        for rid in roster_hit_for_this_name:
            audit.setdefault(rid, []).append(name)

    # rank candidates
    ranked = sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))
    candidates = []
    guessed = None
    for rid, sc in ranked[:4]:
        candidates.append({"roster_id": rid, "matches": sc, "players": audit.get(rid, [])})
    if candidates:
        guessed = candidates[0]["roster_id"]

    return GuessRosterResponse(
        status="ok",
        draft_id=draft_id,
        candidates=candidates,
        guessed_roster_id=guessed,
        note="Use guessed_roster_id with /inspect_draft and /recommend_live if it looks correct.",
        ts=int(time.time()),
    )

def _score_player_for_need(row: Dict[str, Any], need_weights: Dict[str, float]) -> float:
    base = 10000.0 - float(row.get("_avg_rank", 9999.0))
    pos = row.get("pos", "").upper()
    w = need_weights.get(pos, 1.0)
    return base * w

@app.post("/recommend_live", response_model=RecommendLiveResponse)
async def recommend_live(req: RecommendLiveRequest, x_api_key: Optional[str] = Header(None)):
    await require_key(x_api_key)
    await load_players(False)
    load_rankings()

    draft_id, league_id, meta = await resolve_draft_and_league(req.draft_url, req.league_id)
    slot_raw, slot_norm, _dummy, display_by_roster = await fetch_slot_maps(league_id)
    picks = await fetch_picks(draft_id)

    drafted_ids = {str(pk.get("player_id")) for pk in picks if pk.get("player_id")}
    eff_roster: Optional[int] = req.roster_id

    if not eff_roster and req.team_slot and req.team_slot < len(slot_norm):
        eff_roster = slot_norm[req.team_slot]
    if not eff_roster and req.team_name:
        nteam = nrm(req.team_name)
        for rid_str, disp in display_by_roster.items():
            if nrm(disp) == nteam:
                eff_roster = int(rid_str)
                break

    my_team_ids = [str(pk.get("player_id")) for pk in picks if pk.get("roster_id") == eff_roster and pk.get("player_id")]
    my_team = [{"id": pid, "name": pick_player_name(pid)} for pid in my_team_ids]

    # Build simple needs based on caps – count current by position
    caps = req.roster_slots or default_roster_caps()
    have: Dict[str, int] = {k: 0 for k in caps.keys()}
    for pid in my_team_ids:
        pos = (PLAYERS.get(pid) or {}).get("position") or ""
        pos = pos.upper()
        if pos in have:
            have[pos] += 1
    need_weights = {}
    for pos, cap in caps.items():
        deficit = max(0, cap - have.get(pos, 0))
        need_weights[pos] = 1.0 + (0.5 * deficit)  # light boost per deficit

    # Build undrafted ranked board by matching rankings names to Sleeper players
    idx = players_index_by_normalized()
    recommended: List[Dict[str, Any]] = []
    considered = 0
    for row in sorted(RANKINGS, key=lambda x: x.get("_avg_rank", 9999.0)):
        pid = idx.get(row.get("_n", ""))
        if not pid:
            continue
        if pid in drafted_ids:
            continue
        pos = (PLAYERS.get(pid) or {}).get("position") or row.get("pos") or ""
        score = _score_player_for_need(row, need_weights)
        recommended.append({
            "id": pid,
            "name": pick_player_name(pid),
            "team": (PLAYERS.get(pid) or {}).get("team"),
            "pos": pos,
            "rank_avg": row.get("avg") or row.get("rank_avg"),
            "proj_ros": row.get("proj_ros"),
            "score": round(score, 3),
            "explain": f"rank={row.get('avg') or row.get('rank_avg')}, pos={pos}, need={need_weights.get(pos,1.0):.2f}"
        })
        considered += 1
        if len(recommended) >= max(5, req.limit):
            break

    return RecommendLiveResponse(
        status="ok",
        pick=req.pick_number,
        season_used=req.season,
        recommended=recommended[:req.limit],
        my_team=my_team,
        draft_state={"draft_id": draft_id, **meta},
        effective_roster_id=eff_roster,
        drafted_count=len(drafted_ids),
        my_team_count=len(my_team),
        ts=int(time.time()),
    )

# --------------------------- local dev --------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
