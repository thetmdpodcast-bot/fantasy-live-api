# server.py — Sleeper live draft → rankings.csv ("AVG") name-join + roster-aware scoring
# Robustly maps user-provided team slot or roster_id to the correct roster_id.
# Handles slot_to_roster as list OR dict, 1- or 0-indexed, string or int keys.
# Includes /inspect_draft for debug and /recommend_live for picks.

import os
import time
import asyncio
import random
import re
import math
import csv
from typing import Any, Dict, List, Optional, Set, Tuple

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import httpx

API_KEY = os.getenv("API_KEY")  # if set, require x-api-key header
SLEEPER = "https://api.sleeper.app/v1"
RANKINGS_CSV_PATH = os.getenv("RANKINGS_CSV_PATH", "rankings.csv")

ALLOWED_POS = {"QB", "RB", "WR", "TE"}
NFL_TEAMS = {
    "ARI","ATL","BAL","BUF","CAR","CHI","CIN","CLE","DAL","DEN","DET","GB",
    "HOU","IND","JAX","KC","LAC","LAR","LV","MIA","MIN","NE","NO","NYG",
    "NYJ","PHI","PIT","SEA","SF","TB","TEN","WAS"
}

app = FastAPI(title="Fantasy Live Draft (rankings.csv name-join)")

# -------------------- caches --------------------
PLAYERS_CACHE: Dict[str, Dict[str, Any]] = {}
PLAYERS_LOADED_AT: float = 0.0
PLAYERS_RAW_COUNT: int = 0
PLAYERS_KEPT_COUNT: int = 0

RANK_IDX: Dict[str, List[Dict[str, Any]]] = {}  # name_key -> list of {name,pos,team,avg,adp,proj}
RANKINGS_ROWS: int = 0
RANKINGS_WARNINGS: List[str] = []
RANKINGS_LAST_MERGE_TS: float = 0.0

# -------------------- auth --------------------
def auth_or_401(x_api_key: Optional[str]):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

# -------------------- HTTP helpers --------------------
async def _http_get(url: str, headers: Optional[Dict[str,str]]=None,
                    max_attempts: int=3, base_backoff: float=0.25, timeout: float=45.0) -> httpx.Response:
    last_err = None
    for i in range(max_attempts):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.get(url, headers=headers)
                if resp.status_code != 200:
                    resp.raise_for_status()
                return resp
        except Exception as e:
            last_err = e
            await asyncio.sleep(base_backoff*(2**i) + random.random()*0.2)
    raise HTTPException(status_code=502, detail=f"Upstream error fetching {url}: {last_err}")

async def _get_json(url: str) -> Any:
    resp = await _http_get(url)
    try:
        return resp.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Couldn't parse JSON at {url}: {e}")

# -------------------- draft URL normalize --------------------
_DRAFT_ID_RE = re.compile(r"(?P<id>\d{16,20})")
def normalize_draft_picks_url(draft_url_or_page: str) -> Tuple[str,str]:
    m = _DRAFT_ID_RE.search(draft_url_or_page)
    if not m:
        raise HTTPException(status_code=400, detail="Could not extract draft_id from provided URL.")
    draft_id = m.group("id")
    return draft_id, f"{SLEEPER}/draft/{draft_id}/picks"

# -------------------- utils --------------------
_SUFFIXES = {"jr","sr","ii","iii","iv","v"}
_KEEP_INITIALS = {"dk","aj","cj","bj","pj","tj","kj","jj","dj","mj","rj"}

def norm_name(name: str) -> str:
    s = name.lower().strip()
    if "," in s:
        parts = [p.strip() for p in s.split(",")]
        if len(parts) == 2 and parts[0] and parts[1]:
            s = f"{parts[1]} {parts[0]}"
    s = re.sub(r"[.\'\-,]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    parts = s.split()
    if parts and parts[-1] in _SUFFIXES:
        parts = parts[:-1]
    clean = [p for p in parts if len(p) > 1 or p in _KEEP_INITIALS]
    return "".join(clean)

def _to_float(x) -> Optional[float]:
    try:
        if x is None or x == "": return None
        return float(x)
    except Exception:
        return None

# -------------------- Sleeper players --------------------
def _valid_player(p: Dict[str,Any]) -> bool:
    pos = p.get("position"); team = p.get("team")
    if pos not in ALLOWED_POS: return False
    if not team or team not in NFL_TEAMS: return False
    if (p.get("status") or "").lower() == "retired": return False
    return True

async def load_players_if_needed() -> None:
    global PLAYERS_CACHE, PLAYERS_LOADED_AT, PLAYERS_RAW_COUNT, PLAYERS_KEPT_COUNT
    if PLAYERS_CACHE and (time.time() - PLAYERS_LOADED_AT) < 3*3600:
        return
    data = await _get_json(f"{SLEEPER}/players/nfl")
    PLAYERS_CACHE.clear()
    PLAYERS_RAW_COUNT = len(data) if isinstance(data, dict) else 0
    kept = 0
    for pid, p in data.items():
        if not isinstance(p, dict): continue
        if not _valid_player(p): continue
        name = p.get("full_name") or f"{p.get('first_name','').strip()} {p.get('last_name','').strip()}".strip()
        if not name: continue
        PLAYERS_CACHE[pid] = {
            "id": pid,
            "name": name,
            "name_key": norm_name(name),
            "team": p.get("team"),
            "pos": p.get("position"),
            "bye": p.get("bye_week"),
            "adp": _to_float(p.get("adp")),
            "proj_ros": None,
            "rank_avg": None
        }
        kept += 1
    PLAYERS_KEPT_COUNT = kept
    PLAYERS_LOADED_AT = time.time()

# -------------------- rankings.csv index --------------------
def _row_get(row: Dict[str,str], *keys: str) -> Optional[str]:
    for k in keys:
        if k in row and row[k] != "": return row[k]
    return None

def _parse_name(row: Dict[str,str]) -> Optional[str]:
    v = _row_get(row, "Player","Name","player","name","Full Name","FullName")
    if v: return re.sub(r"\s*\(.*?\)\s*$","",v).strip()
    return None

def _parse_pos(row: Dict[str,str]) -> Optional[str]:
    v = _row_get(row, "POS","Pos","Position","position","pos")
    if v:
        v = v.strip().upper()
        m = re.match(r"[A-Z]+", v)
        if m:
            v = m.group(0)
            if v in ALLOWED_POS: return v
    return None

def _parse_team(row: Dict[str,str]) -> Optional[str]:
    v = _row_get(row, "Team","team","NFL Team","NFLTeam")
    if v:
        v = v.strip().upper()
        aliases = {"JAC":"JAX","WSH":"WAS","KAN":"KC","GNB":"GB","NOR":"NO","NWE":"NE","SFO":"SF","TAM":"TB","LVR":"LV","LA":"LAR"}
        v = aliases.get(v,v)
        if v in NFL_TEAMS: return v
    return None

def _parse_avg(row: Dict[str,str]) -> Optional[float]:
    return _to_float(_row_get(row, "AVG","avg","Avg","Average"))

def _parse_adp(row: Dict[str,str]) -> Optional[float]:
    return _to_float(_row_get(row, "ADP","Adp","adp","Average Draft Position"))

def _parse_proj(row: Dict[str,str]) -> Optional[float]:
    return _to_float(_row_get(row, "Proj Pts","Proj","FPts","FPTS","Projected Points","Points"))

def build_rank_index(path: str) -> Tuple[int, List[str], Dict[str, List[Dict[str,Any]]]]:
    warnings: List[str] = []
    idx: Dict[str, List[Dict[str,Any]]] = {}
    rows = 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                name = _parse_name(row)
                if not name: continue
                key = norm_name(name)
                ent = {
                    "name": name,
                    "pos": _parse_pos(row),
                    "team": _parse_team(row),
                    "avg": _parse_avg(row),
                    "adp": _parse_adp(row),
                    "proj": _parse_proj(row)
                }
                idx.setdefault(key, []).append(ent)
                rows += 1
    except FileNotFoundError:
        warnings.append(f"{path} not found (skipping rankings).")
    except Exception as e:
        warnings.append(f"Error reading {path}: {e}")
    return rows, warnings, idx

def ensure_rankings_loaded(force: bool=False) -> None:
    global RANK_IDX, RANKINGS_ROWS, RANKINGS_WARNINGS, RANKINGS_LAST_MERGE_TS
    if force or RANKINGS_LAST_MERGE_TS == 0.0:
        rows, warns, idx = build_rank_index(RANKINGS_CSV_PATH)
        RANK_IDX = idx
        RANKINGS_ROWS = rows
        RANKINGS_WARNINGS = warns
        RANKINGS_LAST_MERGE_TS = time.time()

# -------------------- draft helpers --------------------
def parse_picks(picks_json: Any) -> List[Dict[str, Any]]:
    if isinstance(picks_json, list): return picks_json
    if isinstance(picks_json, dict) and isinstance(picks_json.get("picks"), list):
        return picks_json["picks"]
    return []

def parse_drafted_from_picks(picks: List[Dict[str, Any]]) -> Tuple[Set[str], Dict[int, Set[str]]]:
    drafted: Set[str] = set()
    by_roster: Dict[int, Set[str]] = {}
    for p in picks:
        pid = p.get("player_id")
        rid = p.get("roster_id")
        if pid:
            drafted.add(str(pid))
            if isinstance(rid, int):
                by_roster.setdefault(rid, set()).add(str(pid))
    return drafted, by_roster

async def get_draft_meta(draft_id: str) -> Dict[str,Any]:
    return await _get_json(f"{SLEEPER}/draft/{draft_id}")

def normalize_slot_to_roster(slot_to_roster_raw: Any, total_teams: int) -> List[Optional[int]]:
    """
    Normalize Sleeper's slot_to_roster into a 1-indexed list where index i (0-based) = team slot (i+1).
    Handles:
      - list: [rid1, rid2, ...]
      - dict: {"1": rid1, "2": rid2, ...} or {1: rid1, 2: rid2, ...} or 0-based keys
    Returns a list (length may be <= total_teams if upstream is inconsistent).
    """
    result: List[Optional[int]] = []
    if isinstance(slot_to_roster_raw, list):
        # already ordered by slot (assumed 1..N)
        result = [int(x) if isinstance(x, (int, str)) and str(x).isdigit() else None for x in slot_to_roster_raw]
    elif isinstance(slot_to_roster_raw, dict):
        # try 1-based numeric/string keys
        tmp: List[Optional[int]] = []
        for i in range(1, total_teams + 1):
            v = slot_to_roster_raw.get(i)
            if v is None:
                v = slot_to_roster_raw.get(str(i))
            if v is None:
                # maybe dict is 0-based keys
                v = slot_to_roster_raw.get(i - 1) or slot_to_roster_raw.get(str(i - 1))
            if v is None:
                tmp.append(None)
            else:
                try:
                    tmp.append(int(v))
                except Exception:
                    tmp.append(None)
        result = tmp
    else:
        result = []
    return result

def map_slot_or_roster_id(input_id: int, slot_to_roster: Any, by_roster: Dict[int, Set[str]], total_teams: int) -> Tuple[int, Optional[int]]:
    """
    Returns (effective_roster_id, effective_team_slot or None).
    - If input_id is an actual roster_id (observed in picks), use it.
    - Else, treat it as a team slot and map via slot_to_roster (robust to dict/list and indexing).
    - If mapping fails, fall back to input_id as roster_id (but do not crash).
    """
    if input_id in by_roster:
        return input_id, None

    normalized = normalize_slot_to_roster(slot_to_roster, total_teams)
    eff_slot = None
    eff_rid: Optional[int] = None

    if normalized:
        idx = int(input_id) - 1  # team slot 1..N -> 0-based
        if 0 <= idx < len(normalized):
            eff_rid = normalized[idx]
            if isinstance(eff_rid, int):
                eff_slot = int(input_id)

    if isinstance(eff_rid, int):
        return eff_rid, eff_slot

    # last resort: if slot_to_roster missing, but picks show roster ids in order, map nth smallest
    try:
        uniq = sorted(by_roster.keys())
        idx = int(input_id) - 1
        if 0 <= idx < len(uniq):
            return uniq[idx], int(input_id)
    except Exception:
        pass

    # final fallback: return the provided value as a roster_id (best effort, never crash)
    return int(input_id), None

async def draft_state(draft_id: str, picks: List[Dict[str, Any]]) -> Dict[str, Any]:
    meta = await get_draft_meta(draft_id)
    total_teams = int(meta.get("metadata", {}).get("teams")
                      or meta.get("settings", {}).get("teams")
                      or 0)
    raw_slot_to_roster = meta.get("slot_to_roster_id") or meta.get("slot_to_roster") or {}
    normalized_slots = normalize_slot_to_roster(raw_slot_to_roster, total_teams)

    made = len([p for p in picks if p.get("player_id")])

    if not total_teams:
        roster_ids = {p.get("roster_id") for p in picks if isinstance(p.get("roster_id"), int)}
        total_teams = len(roster_ids) or 12

    on_index = made + 1
    cur_round = max(1, math.ceil(on_index / total_teams))
    pick_in_round = ((on_index - 1) % total_teams) + 1
    snake_rev = (cur_round % 2 == 0)

    order_index = pick_in_round if not snake_rev else (total_teams - pick_in_round + 1)
    rid_on_clock = None
    if normalized_slots and 0 < order_index <= len(normalized_slots):
        rid_on_clock = normalized_slots[order_index - 1]
    else:
        uniq = sorted({p.get("roster_id") for p in picks if isinstance(p.get("roster_id"), int)})
        if 1 <= order_index <= len(uniq):
            rid_on_clock = uniq[order_index - 1]

    return {
        "draft_id": draft_id,
        "total_teams": total_teams,
        "picks_made": made,
        "current_round": cur_round,
        "pick_in_round": pick_in_round,
        "snake_reversed": snake_rev,
        "team_number_on_clock": order_index,
        "roster_id_on_clock": rid_on_clock,
        "slot_to_roster": raw_slot_to_roster,
        "slot_to_roster_normalized": normalized_slots
    }

# -------------------- name-join: UNDRAFTED ↔ rankings.csv --------------------
def match_players_with_rankings(undrafted_pids: List[str]) -> List[Dict[str, Any]]:
    result: List[Dict[str,Any]] = []
    for pid in undrafted_pids:
        p = PLAYERS_CACHE.get(pid)
        if not p: continue
        rows = RANK_IDX.get(p.get("name_key"), [])
        if not rows: continue

        chosen = None
        for r in rows:
            if r.get("pos") == p.get("pos"):
                chosen = r; break
        if not chosen: chosen = rows[0]

        avg = chosen.get("avg")
        proj = chosen.get("proj")
        if avg is not None:
            p["rank_avg"] = avg
            p["adp"] = avg
            p["proj_ros"] = max(float(p.get("proj_ros") or 0.0), max(0.0, 400.0 - float(avg)))
        if proj is not None:
            p["proj_ros"] = proj

        result.append(p)
    return result

# -------------------- scoring --------------------
NEED_WEIGHTS = {"QB":1.1, "RB":1.45, "WR":1.45, "TE":1.25}

def roster_cap(pos: str, slots: Dict[str,int]) -> int:
    if pos in {"RB","WR"}: return slots.get(pos,0) + slots.get("FLEX",0) + 2
    if pos in {"TE","QB"}: return slots.get(pos,0) + 1
    return slots.get(pos,0) + 1

def compute_needs(my_ids: List[str], slots: Dict[str,int]) -> Dict[str,float]:
    counts = {k:0 for k in NEED_WEIGHTS.keys()}
    for pid in my_ids:
        pos = PLAYERS_CACHE.get(pid,{}).get("pos")
        if pos in counts: counts[pos]+=1
    needs: Dict[str,float] = {}
    for pos, base in NEED_WEIGHTS.items():
        gap = max(0, slots.get(pos,0) - counts.get(pos,0))
        needs[pos] = base if gap>0 else base*0.35
    return needs

def score_available(available: List[Dict[str,Any]], my_ids: List[str], slots: Dict[str,int], pick_number: int) -> List[Dict[str,Any]]:
    needs = compute_needs(my_ids, slots)
    pos_counts: Dict[str,int] = {}
    for pid in my_ids:
        pos = PLAYERS_CACHE.get(pid,{}).get("pos")
        if pos: pos_counts[pos] = pos_counts.get(pos,0)+1

    adps = [p.get("adp") for p in available if isinstance(p.get("adp"), (int,float))]
    adp_median = sorted(adps)[len(adps)//2] if adps else None

    scored: List[Dict[str,Any]] = []
    for p in available:
        pos = p["pos"]
        proj = float(p.get("proj_ros") or 0.0)
        adp = p.get("adp")
        adp_discount = (adp_median - adp) if (adp_median is not None and isinstance(adp,(int,float))) else 0.0
        need_boost = needs.get(pos,1.0)
        cur = pos_counts.get(pos,0)
        cap = roster_cap(pos, slots)
        overdrafted = max(0, cur - cap)
        cap_penalty = 0.85 ** overdrafted

        score = proj + 0.15 * adp_discount
        score *= (0.85 + 0.30 * need_boost)
        score *= cap_penalty

        scored.append({
            "id": p["id"], "name": p.get("name"), "team": p.get("team"),
            "pos": pos, "bye": p.get("bye"),
            "adp": p.get("adp"), "rank_avg": p.get("rank_avg"),
            "proj_ros": p.get("proj_ros"),
            "score": round(score,3),
            "explain": f"{pos} need {need_boost:.2f}, cap {cur}/{cap}, pick {pick_number}"
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored

# -------------------- request models --------------------
class RecommendReq(BaseModel):
    draft_url: str
    roster_id: int          # may be a real roster_id OR the team slot number (1..N)
    pick_number: int
    season: int = 2025
    roster_slots: Optional[Dict[str,int]] = None
    limit: int = 10

class InspectReq(BaseModel):
    draft_url: str
    roster_id: int          # may be a real roster_id OR the team slot number

# -------------------- lifecycle --------------------
@app.on_event("startup")
async def startup():
    await load_players_if_needed()
    ensure_rankings_loaded(force=True)

@app.get("/warmup")
def warmup():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(load_players_if_needed())
    ensure_rankings_loaded(force=True)
    return {
        "ok": True,
        "players_cached": len(PLAYERS_CACHE),
        "players_raw": PLAYERS_RAW_COUNT,
        "players_kept": PLAYERS_KEPT_COUNT,
        "rankings_rows": RANKINGS_ROWS,
        "rankings_warnings": RANKINGS_WARNINGS,
        "ts": int(time.time())
    }

@app.get("/health")
def health():
    return {
        "ok": True,
        "players_cached": len(PLAYERS_CACHE),
        "players_raw": PLAYERS_RAW_COUNT,
        "players_kept": PLAYERS_KEPT_COUNT,
        "rankings_rows": RANKINGS_ROWS,
        "rankings_last_merge": int(RANKINGS_LAST_MERGE_TS) if RANKINGS_LAST_MERGE_TS else None,
        "rankings_warnings": RANKINGS_WARNINGS,
        "ts": int(time.time())
    }

# -------------------- endpoints --------------------
@app.post("/inspect_draft")
async def inspect_draft(body: InspectReq, x_api_key: Optional[str] = Header(None)):
    auth_or_401(x_api_key)
    await load_players_if_needed()
    ensure_rankings_loaded()

    draft_id, picks_url = normalize_draft_picks_url(body.draft_url)
    picks_json = await _get_json(picks_url)
    picks = parse_picks(picks_json)
    drafted_ids, by_roster = parse_drafted_from_picks(picks)
    state = await draft_state(draft_id, picks)

    eff_roster_id, eff_slot = map_slot_or_roster_id(
        body.roster_id,
        state.get("slot_to_roster"),
        by_roster,
        state.get("total_teams") or 12
    )

    my_ids = list(by_roster.get(eff_roster_id, set()))
    my_team = [{
        "id": pid,
        "name": PLAYERS_CACHE[pid]["name"],
        "pos": PLAYERS_CACHE[pid]["pos"],
        "team": PLAYERS_CACHE[pid]["team"],
    } for pid in my_ids if pid in PLAYERS_CACHE]

    undrafted_pids = [pid for pid in PLAYERS_CACHE if pid not in drafted_ids and pid not in my_ids]
    available_enriched = match_players_with_rankings(undrafted_pids)
    sample = sorted(
        [{"name": p["name"], "pos": p["pos"], "team": p["team"], "rank_avg": p.get("rank_avg")} for p in available_enriched],
        key=lambda x: (9999.0 if x["rank_avg"] is None else x["rank_avg"])
    )[:10]

    observed_roster_ids = sorted(list(by_roster.keys()))
    by_counts = {rid: len(ids) for rid, ids in by_roster.items()}

    return {
        "status": "ok",
        "draft_state": {k: v for k, v in state.items() if k not in ("slot_to_roster",)},
        "slot_to_roster_raw": state.get("slot_to_roster"),
        "slot_to_roster_normalized": state.get("slot_to_roster_normalized"),
        "observed_roster_ids": observed_roster_ids,
        "by_roster_counts": by_counts,
        "input_roster_or_slot": body.roster_id,
        "effective_roster_id": eff_roster_id,
        "effective_team_slot": eff_slot,
        "my_team": my_team,
        "drafted_count": len(drafted_ids),
        "my_team_count": len(my_ids),
        "undrafted_count": len(undrafted_pids),
        "csv_matched_count": len(available_enriched),
        "csv_top_preview": sample,
        "ts": int(time.time()),
    }

@app.post("/recommend_live")
async def recommend_live(body: RecommendReq, x_api_key: Optional[str] = Header(None)):
    auth_or_401(x_api_key)
    await load_players_if_needed()
    ensure_rankings_loaded()

    draft_id, picks_url = normalize_draft_picks_url(body.draft_url)
    try:
        picks_json = await _get_json(picks_url)
    except HTTPException as e:
        if "404" in str(e.detail):
            return {
                "status": "draft_not_found",
                "message": f"Sleeper returned 404 for draft {draft_id}. Re-check the link.",
                "draft_id": draft_id, "picks_url": picks_url,
                "recommended": [], "alternatives": [], "my_team": [],
                "drafted_count": 0, "my_team_count": 0, "ts": int(time.time())
            }
        raise

    picks = parse_picks(picks_json)
    if not picks:
        return {
            "status": "waiting_for_live_feed",
            "message": "Live picks feed is empty/invalid. Not proceeding with recommendations.",
            "draft_id": draft_id, "picks_url": picks_url,
            "recommended": [], "alternatives": [], "my_team": [],
            "drafted_count": 0, "my_team_count": 0, "ts": int(time.time())
        }

    state = await draft_state(draft_id, picks)
    drafted_ids, by_roster = parse_drafted_from_picks(picks)

    eff_roster_id, _ = map_slot_or_roster_id(
        body.roster_id,
        state.get("slot_to_roster"),
        by_roster,
        state.get("total_teams") or 12
    )
    my_ids = list(by_roster.get(eff_roster_id, set()))

    undrafted_pids: List[str] = [pid for pid in PLAYERS_CACHE if pid not in drafted_ids and pid not in my_ids]
    available_enriched = match_players_with_rankings(undrafted_pids)

    if not available_enriched:
        return {
            "status": "no_available_after_filtering",
            "message": "No available players after filtering (no CSV matches).",
            "draft_state": {k: v for k, v in state.items() if k not in ("slot_to_roster",)},
            "recommended": [], "alternatives": [],
            "my_team": [PLAYERS_CACHE[pid] for pid in my_ids if pid in PLAYERS_CACHE],
            "drafted_count": len(drafted_ids), "my_team_count": len(my_ids),
            "ts": int(time.time())
        }

    slots = body.roster_slots or {"QB":1,"RB":2,"WR":2,"TE":1,"FLEX":2}
    ranked = score_available(available_enriched, my_ids, slots, body.pick_number)
    limit = max(3, body.limit)
    topn = ranked[:limit]

    return {
        "status": "ok",
        "pick": body.pick_number,
        "season_used": body.season,
        "recommended": topn[:3],
        "alternatives": topn[3:limit],
        "my_team": [PLAYERS_CACHE[pid] for pid in my_ids if pid in PLAYERS_CACHE],
        "draft_state": {k: v for k, v in state.items() if k not in ("slot_to_roster",)},
        "effective_roster_id": eff_roster_id,
        "drafted_count": len(drafted_ids),
        "my_team_count": len(my_ids),
        "ts": int(time.time())
    }
