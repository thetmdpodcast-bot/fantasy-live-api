# server.py
# Fantasy Live Draft API
# - FastAPI service that merges a local rankings.csv with Sleeper live draft data
# - Robust to Sleeper hiccups: protected endpoints always return HTTP 200 with a "status" field
# - No pandas; pure stdlib + httpx

import csv
import math
import os
import random
import re
import time
import asyncio
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# -------------------- Configuration --------------------

API_KEY = os.getenv("API_KEY")  # require x-api-key for protected routes if set
RANKINGS_CSV_PATH = os.getenv("RANKINGS_CSV_PATH", "rankings.csv")
PLAYERS_TTL_SEC = int(os.getenv("PLAYERS_TTL_SEC", "1800"))

SLEEPER = "https://api.sleeper.app/v1"
UA = {"User-Agent": "fantasy-live-api/1.0 (+https://fantasy-live-api.onrender.com)"}

app = FastAPI(title="Fantasy Live Draft API")

# -------------------- Simple caches --------------------

_rankings_cache: Dict[str, Any] = {
    "rows": [],                  # List[dict] from rankings.csv
    "by_id": {},                 # id(str) -> row
    "by_name": {},               # normalized_name -> list[id]
    "warn": [],                  # list[str]
    "ts": 0,                     # last load ts
}

_players_cache: Dict[str, Any] = {
    # We mirror counts for health; we don't pull all Sleeper players (huge).
    "raw": 0,
    "kept": 0,
    "ts": 0,
}

def _now() -> int:
    return int(time.time())

def _auth_or_401(x_api_key: Optional[str]):
    if API_KEY and (x_api_key or "") != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

def _normalize_name(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _extract_draft_id(draft_url: str) -> str:
    # accepts https://sleeper.com/draft/nfl/<id> OR raw id
    if not draft_url:
        raise HTTPException(status_code=400, detail="Missing draft_url")
    m = re.search(r"(\d{12,20})", draft_url)
    if not m:
        raise HTTPException(status_code=400, detail="Could not extract draft_id from draft_url")
    return m.group(1)

# -------------------- HTTP helpers --------------------

async def _http_get(url: str, *, headers: Optional[Dict[str, str]] = None,
                    max_attempts: int = 3, base_backoff: float = 0.35, timeout: float = 45.0) -> httpx.Response:
    last_err = None
    hdrs = UA if not headers else {**UA, **headers}
    for attempt in range(max_attempts):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.get(url, headers=hdrs)
                if resp.status_code != 200:
                    resp.raise_for_status()
                return resp
        except Exception as e:
            last_err = e
            await asyncio.sleep(base_backoff * (2 ** attempt) + random.random() * 0.2)
    raise HTTPException(status_code=502, detail=f"Upstream error fetching {url}: {last_err}")

async def _get_json(url: str) -> Any:
    resp = await _http_get(url)
    try:
        return resp.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Could not parse JSON from {url}: {e}")

# -------------------- Rankings loading --------------------

def _safe_float(x: str) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None

def _load_rankings(force: bool = False):
    """Load rankings.csv into cache; expect at least columns: id, name. Optional: pos, rank_avg, score, proj_ros."""
    try:
        mtime = os.path.getmtime(RANKINGS_CSV_PATH)
    except Exception:
        mtime = 0

    if not force and _rankings_cache["ts"] >= mtime and _rankings_cache["rows"]:
        return

    rows: List[Dict[str, Any]] = []
    by_id: Dict[str, Dict[str, Any]] = {}
    by_name: Dict[str, List[str]] = {}
    warn: List[str] = []

    try:
        with open(RANKINGS_CSV_PATH, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                r = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in r.items()}
                pid = str(r.get("id") or r.get("player_id") or "").strip()
                name = r.get("name") or r.get("player") or ""
                r["pos"] = (r.get("pos") or r.get("position") or "").upper()
                r["rank_avg"] = _safe_float(r.get("rank_avg") or r.get("avg") or r.get("adp"))
                r["score"] = _safe_float(r.get("score"))
                r["proj_ros"] = _safe_float(r.get("proj_ros"))

                if not pid or not name:
                    warn.append(f"Skipping row missing id/name: {r}")
                    continue

                rows.append(r)
                by_id[pid] = r
                key = _normalize_name(name)
                by_name.setdefault(key, []).append(pid)

        _rankings_cache.update({"rows": rows, "by_id": by_id, "by_name": by_name, "warn": warn, "ts": _now()})
        _players_cache.update({"raw": len(rows), "kept": len(rows), "ts": _now()})

    except FileNotFoundError:
        _rankings_cache.update({
            "rows": [], "by_id": {}, "by_name": {},
            "warn": [f"rankings file not found: {RANKINGS_CSV_PATH}"], "ts": _now()
        })

# -------------------- Sleeper helpers --------------------

async def _get_draft_meta(draft_id: str) -> Dict[str, Any]:
    return await _get_json(f"{SLEEPER}/draft/{draft_id}")

async def _get_draft_picks(draft_id: str) -> List[Dict[str, Any]]:
    data = await _get_json(f"{SLEEPER}/draft/{draft_id}/picks")
    if not isinstance(data, list):
        raise HTTPException(status_code=502, detail="Sleeper picks returned non-list")
    return data

async def _draft_id_from_league(league_id: str) -> str:
    drafts = await _get_json(f"{SLEEPER}/league/{league_id}/drafts")
    if not drafts:
        raise HTTPException(status_code=404, detail="No drafts for league")
    def keyer(d: Dict[str, Any]):
        status = (d.get("status") or d.get("state") or "")
        updated = d.get("updated_at") or d.get("created") or 0
        return (status == "in_progress", updated)
    best = sorted(drafts, key=keyer, reverse=True)[0]
    did = str(best.get("draft_id") or best.get("id"))
    if not did:
        raise HTTPException(status_code=502, detail="Draft object missing draft_id")
    return did

def _overall_pick_no(p: Dict[str, Any]) -> Optional[int]:
    if isinstance(p.get("pick_no"), int):
        return p["pick_no"]
    if isinstance(p.get("pick"), int):
        return p["pick"]
    # if round + draft_slot exist, computing overall can be snake-sensitive; skip
    return None

def _slot_from_pick(overall: int, total_teams: int) -> int:
    if overall <= 0 or total_teams <= 0:
        return 0
    return ((overall - 1) % total_teams) + 1

# -------------------- Response wrapper --------------------

def always_200(fn):
    @wraps(fn)  # preserve FastAPI signature (fixes 422 "query/args/kwargs" issue)
    async def _inner(*args, **kwargs):
        try:
            return await fn(*args, **kwargs)
        except HTTPException as e:
            return JSONResponse(
                {"status": "upstream_error", "error": e.detail, "code": e.status_code},
                status_code=200,
            )
        except Exception as e:
            return JSONResponse(
                {"status": "upstream_error", "error": str(e)},
                status_code=200,
            )
    return _inner

# -------------------- Models --------------------

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
    season: Optional[int] = 2025
    roster_slots: Optional[Dict[str, int]] = None
    limit: Optional[int] = 10

# -------------------- Open endpoints: health / warmup / echo_auth --------------------

@app.get("/health")
async def health():
    _load_rankings(False)
    ttl_left = None
    if _players_cache["ts"]:
        ttl_left = max(0, PLAYERS_TTL_SEC - (_now() - _players_cache["ts"])) if PLAYERS_TTL_SEC else None
    return {
        "ok": True,
        "players_cached": _players_cache["kept"],
        "players_raw": _players_cache["raw"],
        "players_kept": _players_cache["kept"],
        "players_ttl_sec": ttl_left,
        "rankings_rows": len(_rankings_cache["rows"]),
        "rankings_last_merge": _rankings_cache["ts"] or None,
        "rankings_warnings": _rankings_cache["warn"],
        "ts": _now(),
    }

@app.get("/warmup")
async def warmup():
    _load_rankings(True)
    return {
        "ok": True,
        "players_cached": _players_cache["kept"],
        "players_raw": _players_cache["raw"],
        "players_kept": _players_cache["kept"],
        "rankings_rows": len(_rankings_cache["rows"]),
        "rankings_warnings": _rankings_cache["warn"],
        "ts": _now(),
    }

@app.get("/echo_auth")
async def echo_auth(x_api_key: Optional[str] = Header(None)):
    got = x_api_key or ""
    want = API_KEY or ""
    return {
        "ok": True,
        "got_present": bool(got),
        "got_length": len(got),
        "expected_present": bool(want),
        "match": (got == want),
        "ts": _now()
    }

# -------------------- Core helpers for draft analysis --------------------

async def _resolve_draft_inputs(body: InspectDraftRequest) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]]]:
    if body.draft_url:
        draft_id = _extract_draft_id(body.draft_url)
    elif body.league_id:
        draft_id = await _draft_id_from_league(body.league_id)
    else:
        raise HTTPException(status_code=400, detail="Provide draft_url or league_id")

    meta = await _get_draft_meta(draft_id)
    picks = await _get_draft_picks(draft_id)
    return draft_id, meta, picks

def _slot_to_roster_from_picks(picks: List[Dict[str, Any]], total_teams: int) -> Dict[int, int]:
    slot_to_roster: Dict[int, int] = {}
    for p in picks:
        overall = _overall_pick_no(p)
        if overall is None:
            continue
        if overall <= total_teams:  # only use round-1
            slot = _slot_from_pick(overall, total_teams)
            rid = p.get("roster_id")
            if isinstance(slot, int) and slot >= 1 and rid:
                slot_to_roster.setdefault(slot, rid)
        if len(slot_to_roster) >= total_teams:
            break
    return slot_to_roster

def _my_team_from_roster(picks: List[Dict[str, Any]], roster_id: int) -> List[str]:
    ids: List[str] = []
    for p in picks:
        if p.get("roster_id") == roster_id:
            pid = str(p.get("player_id") or "")
            if pid:
                ids.append(pid)
    return ids

def _counts_by_pos(player_ids: List[str]) -> Dict[str, int]:
    cnt: Dict[str, int] = {"QB":0, "RB":0, "WR":0, "TE":0, "FLEX":0}
    for pid in player_ids:
        row = _rankings_cache["by_id"].get(pid)
        if not row:
            continue
        pos = (row.get("pos") or "").upper()
        if pos in ("QB","RB","WR","TE"):
            cnt[pos] = cnt.get(pos, 0) + 1
    cnt["FLEX"] = max(0, cnt["RB"] + cnt["WR"] + cnt["TE"])
    return cnt

def _default_roster_caps() -> Dict[str,int]:
    return {"QB":1,"RB":2,"WR":2,"TE":1,"FLEX":2}

def _need_weights(caps: Dict[str,int], have: Dict[str,int]) -> Dict[str,float]:
    weights: Dict[str,float] = {}
    for pos, cap in caps.items():
        if cap <= 0:
            weights[pos] = 0.8
            continue
        left = max(0, cap - have.get(pos, 0))
        weights[pos] = 0.8 + (0.5 * (left / cap))  # 1.3 empty -> ~0.8 filled
    return weights

def _score_row(row: Dict[str, Any], needed: Dict[str,float]) -> float:
    rank = row.get("rank_avg")
    base = max(0.0, 300.0 - float(rank)) if isinstance(rank, (int,float)) else 0.0
    pos = (row.get("pos") or "").upper()
    need_w = needed.get(pos, 1.0)
    return base * need_w

# -------------------- Protected endpoints --------------------

@app.post("/inspect_draft")
@always_200
async def inspect_draft(body: InspectDraftRequest, x_api_key: Optional[str] = Header(None)):
    _auth_or_401(x_api_key)
    _load_rankings(False)

    draft_id, meta, picks = await _resolve_draft_inputs(body)
    total = int(meta.get("total_teams") or meta.get("teams") or 0)

    slot_to_roster_raw = _slot_to_roster_from_picks(picks, total)
    slot_to_roster_normalized: List[Optional[int]] = [None] * (total or 0)
    for s, r in slot_to_roster_raw.items():
        if 1 <= s <= len(slot_to_roster_normalized):
            slot_to_roster_normalized[s-1] = r

    effective_roster_id = None
    effective_team_slot = None

    if body.roster_id:
        effective_roster_id = body.roster_id
    elif body.team_slot and slot_to_roster_raw.get(body.team_slot):
        effective_roster_id = slot_to_roster_raw[body.team_slot]
        effective_team_slot = body.team_slot

    drafted_ids_all = [str(p.get("player_id")) for p in picks if p.get("player_id")]
    undrafted_count = max(0, len(_rankings_cache["rows"]) - len(drafted_ids_all))

    my_team_ids: List[str] = []
    if effective_roster_id:
        my_team_ids = _my_team_from_roster(picks, effective_roster_id)

    return {
        "status": "ok",
        "draft_state": meta,
        "slot_to_roster_raw": slot_to_roster_raw,
        "slot_to_roster_normalized": slot_to_roster_normalized,
        "observed_roster_ids": sorted(list({p.get("roster_id") for p in picks if p.get("roster_id")})),
        "input": body.dict(),
        "effective_roster_id": effective_roster_id,
        "effective_team_slot": effective_team_slot,
        "my_team": [{"player_id": pid, **(_rankings_cache["by_id"].get(pid) or {})} for pid in my_team_ids],
        "drafted_count": len(drafted_ids_all),
        "my_team_count": len(my_team_ids),
        "undrafted_count": undrafted_count,
        "csv_matched_count": len(_rankings_cache["rows"]),
        "csv_top_preview": _rankings_cache["rows"][:5],
        "ts": _now()
    }

@app.post("/guess_roster")
@always_200
async def guess_roster(body: GuessRosterRequest, x_api_key: Optional[str] = Header(None)):
    _auth_or_401(x_api_key)
    _load_rankings(False)

    draft_id = _extract_draft_id(body.draft_url)
    picks = await _get_draft_picks(draft_id)

    provided_ids: List[str] = []
    for n in body.player_names:
        key = _normalize_name(n)
        provided_ids.extend(_rankings_cache["by_name"].get(key, []))

    tally: Dict[int,int] = {}
    roster_players: Dict[int,List[str]] = {}
    for p in picks:
        rid = p.get("roster_id")
        pid = str(p.get("player_id") or "")
        if not rid or not pid:
            continue
        if pid in provided_ids:
            tally[rid] = tally.get(rid, 0) + 1
            roster_players.setdefault(rid, [])
            nm = (_rankings_cache["by_id"].get(pid) or {}).get("name") or pid
            roster_players[rid].append(nm)

    candidates = [{"roster_id": rid, "matches": m, "players": roster_players.get(rid, [])}
                  for rid, m in sorted(tally.items(), key=lambda kv: kv[1], reverse=True)]
    guessed = candidates[0]["roster_id"] if candidates else None

    return {
        "status": "ok",
        "draft_id": draft_id,
        "candidates": candidates,
        "guessed_roster_id": guessed,
        "note": "Use guessed_roster_id with /inspect_draft or /recommend_live if it looks correct.",
        "ts": _now()
    }

@app.post("/recommend_live")
@always_200
async def recommend_live(body: RecommendLiveRequest, x_api_key: Optional[str] = Header(None)):
    _auth_or_401(x_api_key)
    _load_rankings(False)

    draft_id = None
    picks: List[Dict[str, Any]] = []
    meta: Dict[str, Any] = {}

    if body.draft_url or body.league_id:
        draft_id, meta, picks = await _resolve_draft_inputs(
            InspectDraftRequest(draft_url=body.draft_url, league_id=body.league_id)
        )

    my_roster_id = body.roster_id
    if not my_roster_id and body.team_slot:
        total = int(meta.get("total_teams") or 0)
        if total:
            map_raw = _slot_to_roster_from_picks(picks, total)
            my_roster_id = map_raw.get(body.team_slot)

    my_team_ids: List[str] = []
    if my_roster_id:
        my_team_ids = _my_team_from_roster(picks, my_roster_id)

    drafted_ids_all = {str(p.get("player_id")) for p in picks if p.get("player_id")}
    undrafted = [r for r in _rankings_cache["rows"] if str(r.get("id") or r.get("player_id") or "") not in drafted_ids_all]

    caps = body.roster_slots or _default_roster_caps()
    have = _counts_by_pos(my_team_ids)
    need_w = _need_weights(caps, have)

    scored = []
    for r in undrafted:
        pid = str(r.get("id") or r.get("player_id") or "")
        if not pid:
            continue
        score = _score_row(r, need_w)
        item = {
            "id": pid,
            "name": r.get("name"),
            "team": r.get("team"),
            "pos": r.get("pos"),
            "rank_avg": r.get("rank_avg"),
            "proj_ros": r.get("proj_ros"),
            "score": round(score, 3),
            "explain": f"rank={r.get('rank_avg')}, need_w={need_w.get((r.get('pos') or '').upper(),1.0):.2f}"
        }
        scored.append(item)

    scored.sort(key=lambda x: (-x["score"], (x["rank_avg"] if x["rank_avg"] is not None else 9999)))
    limit = max(1, int(body.limit or 10))
    top = scored[:limit]
    alts = scored[limit:limit+limit]

    return {
        "status": "ok",
        "pick": body.pick_number,
        "season_used": body.season or 2025,
        "recommended": top,
        "alternatives": alts,
        "my_team": [{"player_id": pid, **(_rankings_cache["by_id"].get(pid) or {})} for pid in my_team_ids],
        "draft_state": {"draft_id": draft_id} if draft_id else {},
        "effective_roster_id": my_roster_id,
        "drafted_count": len(drafted_ids_all),
        "my_team_count": len(my_team_ids),
        "ts": _now()
    }

# -------------------- Main --------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
