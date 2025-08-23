# server.py
# Fantasy Live Draft API — Sleeper + local rankings.csv
# Endpoints: /health, /warmup, /echo_auth, /inspect_draft, /guess_roster, /recommend_live

import os
import csv
import time
import asyncio
from typing import Any, Dict, List, Optional, Tuple, Set

import httpx
from fastapi import FastAPI, Header, HTTPException

# -------------------------
# Config / globals
# -------------------------

API_KEY = os.getenv("API_KEY")  # if set, require x-api-key header
SLEEPER = "https://api.sleeper.app/v1"
RANKINGS_CSV_PATH = os.getenv("RANKINGS_CSV_PATH", "rankings.csv")

app = FastAPI(title="Fantasy Live Draft API")

# Caches
PLAYERS_CACHE: Dict[str, Dict[str, Any]] = {}
PLAYERS_TS: Optional[int] = None

RANKINGS_CACHE: List[Dict[str, Any]] = []
RANKINGS_TS: Optional[int] = None

# optional name->id map (filled when we see data)
NAME_TO_ID: Dict[str, str] = {}


# -------------------------
# Utilities
# -------------------------

def now_ts() -> int:
    return int(time.time())

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def extract_draft_id_from_url(draft_url: str) -> Optional[str]:
    # Accepts url like https://sleeper.com/draft/nfl/1263988228017369088
    if not draft_url:
        return None
    parts = draft_url.strip("/").split("/")
    if not parts:
        return None
    return parts[-1]

async def http_get_json(url: str) -> Any:
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(url)
        r.raise_for_status()
        return r.json()

def load_rankings_csv(force: bool = False) -> Tuple[int, List[str]]:
    """Load rankings.csv into RANKINGS_CACHE."""
    global RANKINGS_CACHE, RANKINGS_TS
    warnings: List[str] = []

    if (not force) and RANKINGS_CACHE:
        return len(RANKINGS_CACHE), warnings

    rows: List[Dict[str, Any]] = []
    if not os.path.exists(RANKINGS_CSV_PATH):
        warnings.append(f"rankings.csv not found at {RANKINGS_CSV_PATH}")
        RANKINGS_CACHE = []
        RANKINGS_TS = now_ts()
        return 0, warnings

    with open(RANKINGS_CSV_PATH, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = {
                "name": raw.get("name") or raw.get("player") or "",
                "team": raw.get("team"),
                "pos": raw.get("pos") or raw.get("position"),
                "rank_avg": raw.get("rank_avg") or raw.get("rank"),
                "adp": raw.get("adp"),
                "proj_ros": raw.get("proj_ros") or raw.get("proj"),
            }
            rows.append(row)

    RANKINGS_CACHE = rows
    RANKINGS_TS = now_ts()
    return len(RANKINGS_CACHE), warnings

async def load_sleeper_players(force: bool = False) -> Tuple[int, List[str]]:
    """Load Sleeper NFL players dict into PLAYERS_CACHE."""
    global PLAYERS_CACHE, PLAYERS_TS, NAME_TO_ID
    warnings: List[str] = []

    if (not force) and PLAYERS_CACHE:
        return len(PLAYERS_CACHE), warnings

    url = f"{SLEEPER}/players/nfl"
    data = await http_get_json(url)
    PLAYERS_CACHE = data if isinstance(data, dict) else {}
    PLAYERS_TS = now_ts()

    # Build a simple name->id index (best-effort)
    NAME_TO_ID = {}
    for pid, p in PLAYERS_CACHE.items():
        name = p.get("full_name") or p.get("first_name")
        if name:
            NAME_TO_ID[name] = pid

    return len(PLAYERS_CACHE), warnings

async def sleeper_draft(draft_id: str) -> Dict[str, Any]:
    return await http_get_json(f"{SLEEPER}/draft/{draft_id}")

async def sleeper_draft_picks(draft_id: str) -> List[Dict[str, Any]]:
    return await http_get_json(f"{SLEEPER}/draft/{draft_id}/picks")

async def sleeper_draft_participants(draft_id: str) -> List[Dict[str, Any]]:
    return await http_get_json(f"{SLEEPER}/draft/{draft_id}/participants")

async def sleeper_league_rosters(league_id: str) -> List[Dict[str, Any]]:
    return await http_get_json(f"{SLEEPER}/league/{league_id}/rosters")


# -------------------------
# Context resolution
# -------------------------

class ResolvedContext:
    def __init__(self):
        self.draft_id: Optional[str] = None
        self.league_id: Optional[str] = None
        self.my_roster_id: Optional[int] = None
        self.my_team_names: List[str] = []
        self.picks_made: Optional[int] = None
        self.drafted_player_ids: Set[str] = set()

async def resolve_context(
    draft_url: Optional[str] = None,
    league_id: Optional[str] = None,
    roster_id: Optional[int] = None,
    team_slot: Optional[int] = None,
    team_name: Optional[str] = None,
) -> ResolvedContext:
    """
    Resolve draft_id, league_id, my roster_id, my team names, picks count, drafted ids.
    Works with:
      - draft_url only
      - league_id + (team_slot or team_name or roster_id)
      - any combo above
    """
    ctx = ResolvedContext()

    # Prefer draft_url if provided
    draft_id: Optional[str] = extract_draft_id_from_url(draft_url) if draft_url else None
    if draft_id:
        d_meta = await sleeper_draft(draft_id)
        ctx.draft_id = draft_id
        ctx.league_id = d_meta.get("league_id") or league_id
    else:
        ctx.league_id = league_id

    # Picks
    picks: List[Dict[str, Any]] = []
    if ctx.draft_id:
        try:
            picks = await sleeper_draft_picks(ctx.draft_id)
        except Exception:
            picks = []

    # Participants + rosters to help map team_slot/team_name -> roster_id
    owner_map: Dict[str, int] = {}  # owner_id -> roster_id
    slot_to_owner: Dict[int, str] = {}
    participants: List[Dict[str, Any]] = []
    if ctx.draft_id:
        try:
            participants = await sleeper_draft_participants(ctx.draft_id)
            for p in participants:
                slot = p.get("slot") or p.get("draft_slot") or p.get("spot")
                if slot is not None:
                    slot_to_owner[int(slot)] = p.get("user_id")
        except Exception:
            participants = []

    league_rosters: List[Dict[str, Any]] = []
    if ctx.league_id:
        try:
            league_rosters = await sleeper_league_rosters(ctx.league_id)
            for r in league_rosters:
                owner_id = r.get("owner_id") or r.get("owner")
                rid = r.get("roster_id")
                if owner_id and rid:
                    owner_map[str(owner_id)] = int(rid)
        except Exception:
            league_rosters = []

    # Determine my roster_id
    my_roster_id: Optional[int] = None

    # 1) Explicit roster_id
    if roster_id:
        my_roster_id = int(roster_id)

    # 2) team_slot -> owner -> roster
    if (my_roster_id is None) and team_slot:
        owner = slot_to_owner.get(int(team_slot))
        if owner:
            my_roster_id = owner_map.get(str(owner))

    # 3) team_name -> best-effort match against participants + rosters
    if (my_roster_id is None) and team_name:
        tnorm = team_name.strip().lower()
        candidate_owner: Optional[str] = None

        # try participants 'metadata.team_name' or 'display_name'
        for p in participants:
            mtn = (p.get("metadata") or {}).get("team_name")
            disp = p.get("display_name")
            for field in [mtn, disp]:
                if field and str(field).strip().lower() == tnorm:
                    candidate_owner = p.get("user_id")
                    break
            if candidate_owner:
                break

        if candidate_owner:
            my_roster_id = owner_map.get(str(candidate_owner))

        # Fallback: if picks have 'roster_id' and contain a player name we can match, skip here (handled later)

    # 4) Last fallback — if there is only one roster in picks with known drafted players and user has given two names, that’s handled in /guess_roster; here we leave it None.

    ctx.my_roster_id = my_roster_id

    # Build drafted sets & my team names
    drafted_ids: Set[str] = set()
    my_team_names: List[str] = []

    for pk in picks or []:
        pid = pk.get("player_id")
        rid = pk.get("roster_id")
        if pid:
            drafted_ids.add(str(pid))
        if my_roster_id is not None and rid == my_roster_id:
            # try to map id -> name for “my team”
            name = None
            if PLAYERS_CACHE:
                p = PLAYERS_CACHE.get(str(pid)) if pid else None
                name = p.get("full_name") if p else None
            if not name and RANKINGS_CACHE:
                # loose match by name via NAME_TO_ID reversed (optional)
                pass
            if name:
                my_team_names.append(name)

    ctx.drafted_player_ids = drafted_ids
    ctx.my_team_names = my_team_names
    ctx.picks_made = len(picks) if picks else 0

    return ctx


# -------------------------
# Endpoints
# -------------------------

@app.get("/health")
async def health():
    players_cached = len(PLAYERS_CACHE) if PLAYERS_CACHE else 0
    players_raw = players_cached
    players_kept = players_cached
    players_ttl_sec = None
    rankings_rows = len(RANKINGS_CACHE) if RANKINGS_CACHE else 0
    return {
        "ok": True,
        "players_cached": players_cached,
        "players_raw": players_raw,
        "players_kept": players_kept,
        "players_ttl_sec": players_ttl_sec,
        "rankings_rows": rankings_rows,
        "rankings_last_merge": None,
        "rankings_warnings": [],
        "ts": now_ts(),
    }

@app.get("/warmup")
async def warmup():
    rc, rw = load_rankings_csv(force=True)
    pc, pw = await load_sleeper_players(force=True)
    return {
        "ok": True,
        "players_cached": pc,
        "players_raw": pc,
        "players_kept": pc,
        "rankings_rows": rc,
        "rankings_warnings": rw + pw,
        "ts": now_ts(),
    }

@app.get("/echo_auth")
async def echo_auth(x_api_key: Optional[str] = Header(None)):
    got_present = x_api_key is not None
    exp_present = API_KEY is not None
    got_len = len(x_api_key) if x_api_key else 0
    match = (API_KEY is None) or (x_api_key == API_KEY)
    # Always 200; this is a diagnostic
    return {
        "ok": True,
        "got_present": got_present,
        "got_len": got_len,
        "exp_present": exp_present,
        "match": match,
        "ts": now_ts(),
    }

@app.post("/inspect_draft")
async def inspect_draft(payload: Dict[str, Any], x_api_key: Optional[str] = Header(None)):
    if API_KEY and (not x_api_key or x_api_key != API_KEY):
        raise HTTPException(status_code=401, detail="Invalid API key")

    draft_url = payload.get("draft_url")
    league_id = payload.get("league_id")
    roster_id = payload.get("roster_id")
    team_slot = payload.get("team_slot")
    team_name = payload.get("team_name")

    ctx = await resolve_context(
        draft_url=draft_url,
        league_id=league_id,
        roster_id=roster_id,
        team_slot=team_slot,
        team_name=team_name,
    )

    # Simple slot-to-roster map (best-effort) for surface
    slot_to_roster_raw = None
    slot_to_roster_normalized = None
    observed_roster_ids: List[int] = []

    if ctx.draft_id:
        try:
            prts = await sleeper_draft_participants(ctx.draft_id)
            if ctx.league_id:
                rosters = await sleeper_league_rosters(ctx.league_id)
                owner_map = {str(r.get("owner_id")): r.get("roster_id") for r in rosters}
            else:
                owner_map = {}

            slot_to_roster_raw = {}
            slot_to_roster_normalized = []
            for p in prts:
                slot = p.get("slot") or p.get("draft_slot") or p.get("spot")
                owner = p.get("user_id")
                rid = owner_map.get(str(owner))
                if slot is not None:
                    slot_to_roster_raw[str(slot)] = rid
            # normalized as array (1-indexed)
            if slot_to_roster_raw:
                max_slot = max(int(s) for s in slot_to_roster_raw.keys())
                slot_to_roster_normalized = []
                for s in range(1, max_slot + 1):
                    slot_to_roster_normalized.append(slot_to_roster_raw.get(str(s)))
                observed_roster_ids = [rid for rid in slot_to_roster_raw.values() if rid]
        except Exception:
            pass

    resp = {
        "status": "ok",
        "draft_state": {"draft_id": ctx.draft_id, "league_id": ctx.league_id, "picks_made": ctx.picks_made},
        "slot_to_roster_raw": slot_to_roster_raw,
        "slot_to_roster_normalized": slot_to_roster_normalized,
        "observed_roster_ids": observed_roster_ids,
        "by_roster_counts": {},
        "input": payload,
        "effective_roster_id": ctx.my_roster_id,
        "effective_team_slot": team_slot,
        "my_team": [{"name": n} for n in ctx.my_team_names],
        "drafted_count": len(ctx.drafted_player_ids),
        "my_team_count": len(ctx.my_team_names),
        "undrafted_count": None,
        "csv_matched_count": None,
        "csv_top_preview": RANKINGS_CACHE[:5] if RANKINGS_CACHE else [],
        "ts": now_ts(),
    }
    return resp

@app.post("/guess_roster")
async def guess_roster(payload: Dict[str, Any], x_api_key: Optional[str] = Header(None)):
    if API_KEY and (not x_api_key or x_api_key != API_KEY):
        raise HTTPException(status_code=401, detail="Invalid API key")

    draft_url = payload.get("draft_url")
    names = payload.get("player_names") or []
    draft_id = extract_draft_id_from_url(draft_url)
    if not draft_id:
        raise HTTPException(status_code=400, detail="Missing or invalid draft_url")

    try:
        picks = await sleeper_draft_picks(draft_id)
    except Exception:
        picks = []

    # Build map: roster_id -> set of player names
    roster_to_names: Dict[int, Set[str]] = {}
    for pk in picks or []:
        rid = pk.get("roster_id")
        pid = pk.get("player_id")
        if rid is None or pid is None:
            continue
        rid = int(rid)
        p = PLAYERS_CACHE.get(str(pid), {})
        nm = p.get("full_name")
        if not nm:
            continue
        roster_to_names.setdefault(rid, set()).add(nm)

    # Score candidates by name matches (case-insensitive)
    wanted = set(n.strip().lower() for n in names if n and isinstance(n, str))
    candidates = []
    for rid, got in roster_to_names.items():
        score = len([1 for g in got if g and g.lower() in wanted])
        if score > 0:
            candidates.append({"roster_id": rid, "matches": score, "players": sorted(list(got))})

    candidates.sort(key=lambda x: (-x["matches"], x["roster_id"]))
    guessed = candidates[0]["roster_id"] if candidates else None

    return {
        "status": "ok",
        "draft_id": draft_id,
        "candidates": candidates,
        "guessed_roster_id": guessed,
        "note": "best-effort name match against drafted picks",
        "ts": now_ts(),
    }

@app.post("/recommend_live")
async def recommend_live(payload: Dict[str, Any], x_api_key: Optional[str] = Header(None)):
    # Require API key if configured
    if API_KEY and (not x_api_key or x_api_key != API_KEY):
        raise HTTPException(status_code=401, detail="Invalid API key")

    draft_url = payload.get("draft_url")
    league_id = payload.get("league_id")
    roster_id = payload.get("roster_id")
    team_slot = payload.get("team_slot")
    team_name = payload.get("team_name")
    pick_number = payload.get("pick_number")
    season = int(payload.get("season", 2025))
    limit = int(payload.get("limit", 10))

    # Ensure caches are present
    if not RANKINGS_CACHE:
        load_rankings_csv(force=True)
    if not PLAYERS_CACHE:
        await load_sleeper_players(force=False)

    ctx = await resolve_context(
        draft_url=draft_url,
        league_id=league_id,
        roster_id=roster_id,
        team_slot=team_slot,
        team_name=team_name,
    )

    # Derive pick # if missing (best-effort)
    if not pick_number:
        try:
            pick_number = int(ctx.picks_made) + 1 if ctx.picks_made is not None else None
        except Exception:
            pick_number = None

    # Build available list from rankings, remove drafted + my own players
    drafted_ids = ctx.drafted_player_ids or set()
    my_names = set(n for n in (ctx.my_team_names or []) if n)

    def is_available(row: Dict[str, Any]) -> bool:
        nm = (row.get("name") or "").strip()
        if not nm:
            return False
        # filter out already on my team (by name)
        if nm in my_names:
            return False
        # filter out drafted by id if we can map name -> id (best-effort)
        pid = NAME_TO_ID.get(nm)
        if pid and (pid in drafted_ids):
            return False
        return True

    primary = [r for r in RANKINGS_CACHE if is_available(r)]

    # You can add stricter position-cap logic here; we keep it simple to avoid empty lists
    candidates = primary

    def rank_key(r: Dict[str, Any]) -> float:
        v = r.get("rank_avg")
        try:
            return float(v)
        except Exception:
            return 999999.0

    candidates.sort(key=rank_key)
    top = candidates[:limit]

    # Graceful fallbacks
    if not top:
        # ignore drafted id and caps; only remove my own players by name
        fallback = [r for r in RANKINGS_CACHE if (r.get("name") or "") not in my_names]
        fallback.sort(key=rank_key)
        top = fallback[:limit]

    if not top:
        # Still empty? return top N of raw rankings
        tmp = list(RANKINGS_CACHE)
        tmp.sort(key=rank_key)
        top = tmp[:limit]

    recommended = []
    for r in top:
        recommended.append({
            "name": r.get("name"),
            "team": r.get("team"),
            "pos": r.get("pos"),
            "rank_avg": safe_float(r.get("rank_avg")),
            "adp": safe_float(r.get("adp")),
            "proj_ros": safe_float(r.get("proj_ros")),
            "explain": f"rank={r.get('rank_avg')}" + (f", pick {pick_number}" if pick_number else ""),
        })

    resp = {
        "status": "ok" if recommended else "empty",
        "pick": pick_number,
        "season_used": season,
        "recommended": recommended,
        "alternatives": [],
        "my_team": [{"name": n} for n in (ctx.my_team_names or [])],
        "draft_state": {"draft_id": ctx.draft_id, "picks_made": ctx.picks_made},
        "effective_roster_id": ctx.my_roster_id,
        "drafted_count": len(drafted_ids) if drafted_ids else None,
        "my_team_count": len(ctx.my_team_names or []),
        "ts": now_ts(),
    }
    return resp


# -------------------------
# Main
# -------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
