# server.py
# Fantasy Live Draft API — Sleeper live draft + rankings.csv

from __future__ import annotations

import asyncio
import csv
import hashlib
import os
import random
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx
from httpx import HTTPStatusError, RequestError
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import JSONResponse

# ---------------------------
# Config
# ---------------------------
API_KEY = (os.getenv("API_KEY") or "").strip()                 # required on secured routes
RANKINGS_CSV_PATH = os.getenv("RANKINGS_CSV_PATH", "rankings.csv")
PLAYERS_TTL_SEC = int(os.getenv("PLAYERS_TTL_SEC", "1080"))

SLEEPER = "https://api.sleeper.app/v1"
HTTP_HEADERS = {
    "User-Agent": "fantasy-live-api/1.0 (+https://fantasy-live-api.onrender.com)",
    "Accept": "application/json",
}

DEFAULT_SLOTS: Dict[str, int] = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 2}
DEFAULT_LIMIT = 10

# ---------------------------
# App
# ---------------------------
app = FastAPI(title="Fantasy Live Draft API", version="1.0.4")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Caches
# ---------------------------
PLAYERS_RAW: Dict[str, Dict[str, Any]] = {}
PLAYERS_KEEP: Dict[str, Dict[str, Any]] = {}
PLAYERS_BY_NAME: Dict[str, str] = {}
PLAYERS_FETCHED_AT: Optional[int] = None

RANKINGS_ROWS: List[Dict[str, Any]] = []
RANKINGS_WARNINGS: List[str] = []
RANKINGS_LAST_MERGE: Optional[int] = None

def now_ts() -> int:
    return int(time.time())

# ---------------------------
# Helpers
# ---------------------------
def norm_name(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return re.sub(r"\s+", " ", s)

def convert_num(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    try:
        xx = str(x).replace(",", "").strip()
        if xx in ("", "NA", "N/A", "null", "None", "-"):
            return None
        return float(xx)
    except Exception:
        return None

def extract_draft_id(s: str) -> Optional[str]:
    if not s:
        return None
    m = re.search(r"(\d{10,})", s)
    return m.group(1) if m else None

async def http_get_json(url: str, *, tolerate_404=False) -> Any:
    """GET JSON with short retry; logs upstream status. 404 can be empty list."""
    for attempt in (1, 2):
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(10.0, read=10.0),
                headers=HTTP_HEADERS,
                follow_redirects=True,
            ) as client:
                r = await client.get(url)
            if r.status_code == 404 and tolerate_404:
                return []
            r.raise_for_status()
            try:
                return r.json()
            except Exception:
                return r.text
        except HTTPStatusError as e:
            body = e.response.text[:400]
            print(f"[sleeper] {url} -> {e.response.status_code} {body}")
            if attempt == 2:
                raise
            await asyncio.sleep(0.35 + random.random() * 0.4)
        except RequestError as e:
            print(f"[sleeper] network {url} -> {e}")
            if attempt == 2:
                raise
            await asyncio.sleep(0.35 + random.random() * 0.4)

def player_display_name(p: Dict[str, Any]) -> str:
    return p.get("full_name") or f"{p.get('first_name','')} {p.get('last_name','')}".strip()

def should_keep_player(p: Dict[str, Any]) -> bool:
    pos = p.get("position")
    return pos in {"QB", "RB", "WR", "TE", "K", "DEF"} and (p.get("status") != "Inactive")

def pick_value(d: Dict[str, Any], keys: List[str], default=None):
    for k in keys:
        if k in d and d[k] not in (None, "", "NA", "N/A", "-"):
            return d[k]
    return default

# ---------------------------
# Auth
# ---------------------------
PROTECTED_PATHS = {"/inspect_draft", "/guess_roster", "/recommend_live"}

def provided_api_key(request: Request) -> Optional[str]:
    key = request.headers.get("x-api-key")
    if key:
        return key.strip()
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth.split(None, 1)[1].strip()
    return None

@app.middleware("http")
async def auth_and_log_mw(request: Request, call_next: RequestResponseEndpoint):
    if request.url.path in PROTECTED_PATHS:
        got = (provided_api_key(request) or "")
        exp = API_KEY
        got_h = hashlib.sha256(got.encode()).hexdigest()[:8] if got else "-"
        exp_h = hashlib.sha256(exp.encode()).hexdigest()[:8] if exp else "-"
        print(f"[auth] {request.url.path} got={bool(got)} got_sha8={got_h} exp_set={bool(exp)} exp_sha8={exp_h} match={got == exp}")
        if API_KEY and got != API_KEY:
            return JSONResponse({"detail": "Invalid API key"}, status_code=401)
    return await call_next(request)

# ---------------------------
# Loaders
# ---------------------------
async def ensure_players_loaded() -> Tuple[int, int]:
    global PLAYERS_RAW, PLAYERS_KEEP, PLAYERS_BY_NAME, PLAYERS_FETCHED_AT
    if PLAYERS_FETCHED_AT and now_ts() - PLAYERS_FETCHED_AT < PLAYERS_TTL_SEC:
        return len(PLAYERS_RAW), len(PLAYERS_KEEP)

    print("[players] fetch /players/nfl")
    data = await http_get_json(f"{SLEEPER}/players/nfl")
    if not isinstance(data, dict):
        raise HTTPException(status_code=502, detail="Failed to load Sleeper players")

    PLAYERS_RAW = data
    PLAYERS_KEEP = {}
    PLAYERS_BY_NAME = {}
    for pid, p in data.items():
        if not isinstance(p, dict):
            continue
        if should_keep_player(p):
            PLAYERS_KEEP[pid] = p
            n = norm_name(player_display_name(p))
            if n and n not in PLAYERS_BY_NAME:
                PLAYERS_BY_NAME[n] = pid

    PLAYERS_FETCHED_AT = now_ts()
    print(f"[players] raw={len(PLAYERS_RAW)} kept={len(PLAYERS_KEEP)}")
    return len(PLAYERS_RAW), len(PLAYERS_KEEP)

def map_name_to_pid(name: str) -> Optional[str]:
    if not name:
        return None
    return PLAYERS_BY_NAME.get(norm_name(name))

def open_rankings() -> Tuple[int, List[str]]:
    global RANKINGS_ROWS, RANKINGS_WARNINGS, RANKINGS_LAST_MERGE
    RANKINGS_ROWS = []
    RANKINGS_WARNINGS = []
    RANKINGS_LAST_MERGE = None

    if not os.path.exists(RANKINGS_CSV_PATH):
        RANKINGS_WARNINGS.append(f"rankings csv not found: {RANKINGS_CSV_PATH}")
        return 0, RANKINGS_WARNINGS

    with open(RANKINGS_CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = pick_value(row, ["name", "player", "Player", "PLAYER"])
            pos = pick_value(row, ["pos", "position", "POS"])
            team = pick_value(row, ["team", "Team", "TEAM"])
            adp = convert_num(pick_value(row, ["ADP", "adp"]))
            avg = convert_num(pick_value(row, ["AVG", "avg", "Rank", "rank"]))
            proj = convert_num(pick_value(row, ["proj", "proj_ros", "Proj", "Projection"]))
            r: Dict[str, Any] = {
                "name": name, "pos": pos, "team": team,
                "adp": adp, "rank_avg": avg, "proj_ros": proj, "pid": None,
            }
            pid = map_name_to_pid(name)
            if pid:
                r["pid"] = pid
            RANKINGS_ROWS.append(r)

    RANKINGS_LAST_MERGE = now_ts()
    return len(RANKINGS_ROWS), RANKINGS_WARNINGS

async def warmup() -> Dict[str, Any]:
    raw, kept = await ensure_players_loaded()
    rows, warns = open_rankings()
    return {
        "ok": True,
        "players_cached": kept,
        "players_raw": raw,
        "players_kept": kept,
        "rankings_rows": rows,
        "rankings_warnings": warns,
        "ts": now_ts(),
    }

@app.on_event("startup")
async def _auto_warm():
    try:
        await warmup()
        print("[startup] warmup ok")
    except Exception as e:
        print("[startup] warmup failed:", e)

# ---------------------------
# Sleeper fetchers
# ---------------------------
async def fetch_draft_and_picks(draft_url: Optional[str], league_id: Optional[str]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    draft_id: Optional[str] = None
    if draft_url:
        draft_id = extract_draft_id(draft_url)
        if not draft_id:
            raise HTTPException(status_code=400, detail="Could not extract draft_id from draft_url")
    elif league_id:
        drafts = await http_get_json(f"{SLEEPER}/league/{league_id}/drafts")
        if not drafts:
            raise HTTPException(status_code=404, detail="No drafts found for league_id")
        draft_id = str(drafts[0]["draft_id"])
    else:
        raise HTTPException(status_code=400, detail="Provide draft_url or league_id")

    draft_obj = await http_get_json(f"{SLEEPER}/draft/{draft_id}")
    picks = await http_get_json(f"{SLEEPER}/draft/{draft_id}/picks", tolerate_404=True)
    if not isinstance(picks, list):
        picks = []
    return draft_obj, picks

def roster_team(picks: List[Dict[str, Any]], roster_id: int) -> List[Dict[str, Any]]:
    team: List[Dict[str, Any]] = []
    for pk in picks:
        if int(pk.get("roster_id", -1)) != int(roster_id):
            continue
        pid = str(pk.get("player_id") or "")
        sp = PLAYERS_KEEP.get(pid) or PLAYERS_RAW.get(pid) or {}
        team.append({
            "pid": pid,
            "name": player_display_name(sp) or pk.get("metadata", {}).get("player_name"),
            "pos": sp.get("position"),
            "team": sp.get("team"),
            "bye": sp.get("bye_week") or sp.get("bye"),
            "round": pk.get("round"),
            "pick_no": pk.get("pick_no") or pk.get("pick_no_str"),
        })
    return team

def drafted_set(picks: List[Dict[str, Any]]) -> set[str]:
    out: set[str] = set()
    for pk in picks:
        pid = str(pk.get("player_id") or "")
        if pid:
            out.add(pid)
    return out

def count_by_pos(players: List[Dict[str, Any]]) -> Dict[str, int]:
    c: Dict[str, int] = {}
    for p in players:
        pos = p.get("pos")
        if pos:
            c[pos] = c.get(pos, 0) + 1
    return c

# ---------------------------
# Pydantic models
# ---------------------------
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
    roster_slots: Optional[Dict[str, int]] = None
    limit: int = DEFAULT_LIMIT

# ---------------------------
# Public routes
# ---------------------------
@app.get("/health")
async def health():
    kept = len(PLAYERS_KEEP)
    raw = len(PLAYERS_RAW)
    rows = len(RANKINGS_ROWS)
    ttl = PLAYERS_TTL_SEC if PLAYERS_FETCHED_AT else None
    return {
        "ok": True,
        "players_cached": kept,
        "players_raw": raw,
        "players_kept": kept,
        "players_ttl_sec": ttl,
        "rankings_rows": rows,
        "rankings_last_merge": RANKINGS_LAST_MERGE,
        "rankings_warnings": RANKINGS_WARNINGS,
        "ts": now_ts(),
    }

@app.get("/warmup")
async def warmup_route():
    return await warmup()

# Debug helper: verify key presence/match from Builder
@app.get("/echo_auth")
async def echo_auth(request: Request):
    got = provided_api_key(request) or ""
    return {
        "ok": True,
        "got_present": bool(got),
        "got_len": len(got),
        "exp_present": bool(API_KEY),
        "match": bool(API_KEY and got == API_KEY),
    }

# ---------------------------
# Secured routes
# ---------------------------
@app.post("/inspect_draft")
async def inspect_draft(req: InspectDraftRequest, request: Request):
    if API_KEY and (provided_api_key(request) != API_KEY):
        raise HTTPException(status_code=401, detail="Invalid API key")

    await ensure_players_loaded()
    if not RANKINGS_ROWS:
        open_rankings()

    try:
        draft_obj, picks = await fetch_draft_and_picks(req.draft_url, req.league_id)
    except HTTPStatusError as e:
        raise HTTPException(status_code=502, detail={"upstream": "sleeper", "status": e.response.status_code, "url": str(e.request.url), "body": e.response.text[:200]})
    except RequestError as e:
        raise HTTPException(status_code=502, detail={"upstream": "sleeper", "error": str(e)})

    observed: List[int] = []
    for pk in picks:
        rid = pk.get("roster_id")
        if rid is not None:
            ir = int(rid)
            if ir not in observed:
                observed.append(ir)

    effective_roster_id = req.roster_id
    my_team: List[Dict[str, Any]] = []
    if effective_roster_id:
        my_team = roster_team(picks, effective_roster_id)

    drafted = drafted_set(picks)
    csv_matched = sum(1 for r in RANKINGS_ROWS if r.get("pid") and r["pid"] not in drafted)

    csv_preview = []
    for r in RANKINGS_ROWS[:10]:
        csv_preview.append({
            "name": r["name"], "pos": r["pos"], "team": r["team"],
            "adp": r["adp"], "rank_avg": r["rank_avg"], "pid": r["pid"]
        })

    return {
        "status": "ok",
        "draft_state": draft_obj,
        "slot_to_roster_raw": None,
        "slot_to_roster_normalized": observed or None,
        "observed_roster_ids": observed,
        "by_roster_counts": {str(r): 0 for r in observed},
        "input": req.dict(),
        "effective_roster_id": effective_roster_id,
        "effective_team_slot": req.team_slot,
        "my_team": my_team,
        "drafted_count": len(drafted),
        "my_team_count": len(my_team),
        "undrafted_count": max(0, len(PLAYERS_KEEP) - len(drafted)),
        "csv_matched_count": csv_matched,
        "csv_top_preview": csv_preview,
        "ts": now_ts(),
    }

@app.post("/guess_roster")
async def guess_roster(req: GuessRosterRequest, request: Request):
    if API_KEY and (provided_api_key(request) != API_KEY):
        raise HTTPException(status_code=401, detail="Invalid API key")

    await ensure_players_loaded()
    try:
        draft_obj, picks = await fetch_draft_and_picks(req.draft_url, None)
    except HTTPStatusError as e:
        raise HTTPException(status_code=502, detail={"upstream": "sleeper", "status": e.response.status_code, "url": str(e.request.url), "body": e.response.text[:200]})
    except RequestError as e:
        raise HTTPException(status_code=502, detail={"upstream": "sleeper", "error": str(e)})

    want_pids = {pid for nm in req.player_names if (pid := map_name_to_pid(nm))}
    by_roster: Dict[int, set] = {}
    for pk in picks:
        rid = int(pk.get("roster_id", -1))
        if rid < 0:
            continue
        by_roster.setdefault(rid, set()).add(str(pk.get("player_id") or ""))

    cands = []
    best = None
    for rid, pidset in by_roster.items():
        matches = len(want_pids & pidset)
        players = [player_display_name(PLAYERS_RAW.get(pid, {})) for pid in pidset]
        item = {"roster_id": rid, "matches": matches, "players": sorted(players)}
        cands.append(item)
        if matches and (best is None or matches > best["matches"]):
            best = item

    return {
        "status": "ok",
        "draft_id": draft_obj.get("draft_id"),
        "candidates": sorted(cands, key=lambda x: -x["matches"]),
        "guessed_roster_id": best["roster_id"] if best else None,
        "note": "Use guessed_roster_id with /inspect_draft and /recommend_live if it looks correct.",
        "ts": now_ts(),
    }

@app.post("/recommend_live")
async def recommend_live(req: RecommendLiveRequest, request: Request):
    if API_KEY and (provided_api_key(request) != API_KEY):
        raise HTTPException(status_code=401, detail="Invalid API key")

    await ensure_players_loaded()
    if not RANKINGS_ROWS:
        open_rankings()

    try:
        draft_obj, picks = await fetch_draft_and_picks(req.draft_url, req.league_id)
    except HTTPStatusError as e:
        raise HTTPException(status_code=502, detail={"upstream": "sleeper", "status": e.response.status_code, "url": str(e.request.url), "body": e.response.text[:200]})
    except RequestError as e:
        raise HTTPException(status_code=502, detail={"upstream": "sleeper", "error": str(e)})

    drafted = drafted_set(picks)
    effective_roster_id = req.roster_id
    my_team: List[Dict[str, Any]] = []
    if effective_roster_id:
        my_team = roster_team(picks, effective_roster_id)

    caps = req.roster_slots or DEFAULT_SLOTS
    counts = count_by_pos(my_team)
    needs_weight: Dict[str, float] = {}
    for pos, cap in caps.items():
        have = counts.get(pos, 0)
        remain = max(0, cap - have)
        needs_weight[pos] = 1.0 + (0.5 if remain > 0 else 0.0)

    board = []
    for r in RANKINGS_ROWS:
        pid = r.get("pid")
        if pid and pid in drafted:
            continue
        name = r.get("name")
        pos = r.get("pos")
        team = r.get("team")
        adp = r.get("adp")
        rank_avg = r.get("rank_avg")

        base = 999.0
        if rank_avg is not None:
            base = float(rank_avg)
        elif adp is not None:
            base = float(adp) + 30.0

        w = needs_weight.get(pos or "", 1.0)
        score = base / w

        board.append({
            "id": str(pid) if pid else None,
            "name": name,
            "team": team,
            "pos": pos,
            "bye": None,
            "adp": adp,
            "rank_avg": rank_avg,
            "proj_ros": r.get("proj_ros"),
            "score": round(score, 3),
            "explain": f"rank={rank_avg if rank_avg is not None else 'NA'}, adp={adp if adp is not None else 'NA'}, need weight={w:.2f}, pick {req.pick_number if req.pick_number else 'NA'}",
        })

    board.sort(key=lambda x: (x["score"], (x["rank_avg"] if x["rank_avg"] is not None else 9999)))
    lim = max(1, req.limit or DEFAULT_LIMIT)
    recs = board[:lim]
    alts = board[lim: max(2 * lim, 40)]

    return {
        "status": "ok",
        "pick": req.pick_number,
        "season_used": req.season,
        "recommended": recs,
        "alternatives": alts,
        "my_team": my_team,
        "draft_state": draft_obj,
        "effective_roster_id": effective_roster_id,
        "drafted_count": len(drafted),
        "my_team_count": len(my_team),
        "ts": now_ts(),
    }

# ---------------------------
# Local dev
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
