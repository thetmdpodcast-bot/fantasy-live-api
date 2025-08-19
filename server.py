# server.py — Live Sleeper draft + FantasyPros rankings.csv merge (no pandas)
# - Accepts Sleeper web URL + roster_id
# - Loads Sleeper players (QB/RB/WR/TE, real NFL teams)
# - Merges local FantasyPros rankings.csv into scoring (ECR/ADP/proj)
# - Soft-stops (200 JSON) if live picks feed is empty

import os, time, asyncio, random, re, math, csv
from typing import Any, Dict, List, Optional, Set, Tuple
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import httpx

API_KEY = os.getenv("API_KEY")  # if set, require x-api-key header
SLEEPER = "https://api.sleeper.app/v1"

# ===== POS/TEAM filters =====
ALLOWED_POS = {"QB", "RB", "WR", "TE"}   # add "K","DST" later if desired
NFL_TEAMS = {
    "ARI","ATL","BAL","BUF","CAR","CHI","CIN","CLE","DAL","DEN","DET","GB",
    "HOU","IND","JAX","KC","LAC","LAR","LV","MIA","MIN","NE","NO","NYG",
    "NYJ","PHI","PIT","SEA","SF","TB","TEN","WAS"
}

# ===== FantasyPros CSV path (local) =====
RANKINGS_CSV_PATH = os.getenv("RANKINGS_CSV_PATH", "rankings.csv")

app = FastAPI(title="Fantasy Live Draft API (FantasyPros CSV + soft-stop)")

# ===== Caches =====
PLAYERS_CACHE: Dict[str, Dict[str, Any]] = {}
PLAYERS_LOADED_AT: float = 0.0
PLAYERS_RAW_COUNT: int = 0
PLAYERS_KEPT_COUNT: int = 0

RANKINGS_ROWS: int = 0
RANKINGS_WARNINGS: List[str] = []
RANKINGS_LAST_MERGE_TS: float = 0.0

# ===== Auth =====
def auth_or_401(x_api_key: Optional[str]):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

# ===== HTTP helpers =====
async def _http_get(url: str, headers: Optional[Dict[str, str]] = None,
                    max_attempts: int = 3, base_backoff: float = 0.25,
                    timeout: float = 45.0) -> httpx.Response:
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
            await asyncio.sleep(base_backoff * (2 ** i) + random.random() * 0.2)
    raise HTTPException(status_code=502, detail=f"Upstream error fetching {url}: {last_err}")

async def _get_json(url: str) -> Any:
    resp = await _http_get(url)
    try:
        return resp.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Couldn't parse JSON at {url}: {e}")

# ===== Normalize draft URL =====
_DRAFT_ID_RE = re.compile(r"(?P<id>\d{16,20})")

def normalize_draft_picks_url(draft_url_or_page: str) -> Tuple[str, str]:
    """
    Accepts Sleeper web or API URL and returns (draft_id, api_picks_url).
    Examples:
      - https://sleeper.com/draft/nfl/{id}
      - https://sleeper.app/draft/nfl/{id}
      - https://api.sleeper.app/v1/draft/{id}/picks
    """
    m = _DRAFT_ID_RE.search(draft_url_or_page)
    if not m:
        raise HTTPException(status_code=400, detail="Could not extract draft_id from provided URL.")
    draft_id = m.group("id")
    return draft_id, f"{SLEEPER}/draft/{draft_id}/picks"

# ===== Helpers =====
def _to_float(x) -> Optional[float]:
    try:
        if x is None or x == "": return None
        return float(x)
    except Exception:
        return None

def _to_int(x) -> Optional[int]:
    try:
        if x is None or x == "": return None
        return int(float(x))
    except Exception:
        return None

def _norm(s: str) -> str:
    return re.sub(r"[^a-z]", "", (s or "").lower())

def _name_key(name: str, team: Optional[str], pos: Optional[str]) -> str:
    return f"{_norm(name)}|{(team or '').upper()}|{(pos or '').upper()}"

# ===== Players (RELAXED filter; draftable only) =====
def _valid_player(p: Dict[str, Any]) -> bool:
    pos = p.get("position")
    team = p.get("team")
    if pos not in ALLOWED_POS: return False
    if not team or team not in NFL_TEAMS: return False
    status = (p.get("status") or "").lower()
    if status in {"retired"}: return False
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
            "team": p.get("team"),
            "pos": p.get("position"),
            "bye": p.get("bye_week"),
            # Optional metrics (often None from Sleeper)
            "adp": _to_float(p.get("adp")),
            "proj_ros": _to_float(p.get("fantasy_points_half_ppr")),
            "proj_week": _to_float(p.get("fantasy_points")),
        }
        kept += 1
    PLAYERS_KEPT_COUNT = kept
    PLAYERS_LOADED_AT = time.time()

# ===== Rankings CSV merge (FantasyPros) =====
def _row_get(row: Dict[str,str], *keys: str) -> Optional[str]:
    for k in keys:
        if k in row and row[k] != "":
            return row[k]
    return None

def _parse_pos(row: Dict[str,str]) -> Optional[str]:
    v = _row_get(row, "POS", "Pos", "Position", "position", "pos")
    if v:
        v = v.strip().upper()
        v = re.split(r"\s|#", v)[0]
        if v in ALLOWED_POS:
            return v
    return None

def _parse_team(row: Dict[str,str]) -> Optional[str]:
    v = _row_get(row, "Team", "team", "NFL Team", "NFLTeam")
    if v:
        v = v.strip().upper()
        aliases = {"JAC":"JAX","WSH":"WAS","KAN":"KC","GNB":"GB","NOR":"NO","NWE":"NE","SFO":"SF","TAM":"TB","LVR":"LV","LA":"LAR"}
        v = aliases.get(v, v)
        if v in NFL_TEAMS:
            return v
    return None

def _parse_name(row: Dict[str,str]) -> Optional[str]:
    v = _row_get(row, "Player", "Name", "player", "name", "Full Name", "FullName")
    if v:
        v = re.sub(r"\s*\(.*?\)\s*$", "", v).strip()
        return v
    return None

def _parse_ecr_rank(row: Dict[str,str]) -> Optional[int]:
    return _to_int(_row_get(row, "Rank", "ECR", "Overall", "Overall Rank"))

def _parse_adp(row: Dict[str,str]) -> Optional[float]:
    return _to_float(_row_get(row, "ADP", "ADP ", "Avg", "Average Draft Position"))

def _parse_proj_points(row: Dict[str,str]) -> Optional[float]:
    return _to_float(_row_get(row, "Proj Pts", "Proj", "Points", "Projected Points", "FPts", "FPTS", "Fpts"))

def _score_from_rank(rank: Optional[int]) -> Optional[float]:
    if rank is None: return None
    return max(0.0, 400.0 - float(rank))  # lower rank -> higher score

def _merge_into_player(p: Dict[str,Any], ecr_rank: Optional[int], adp: Optional[float], proj_pts: Optional[float]):
    if adp is not None:
        p["adp"] = adp
    if proj_pts is not None:
        p["proj_ros"] = proj_pts
    else:
        derived = _score_from_rank(ecr_rank)
        if derived is not None:
            cur = _to_float(p.get("proj_ros"))
            p["proj_ros"] = max(cur or 0.0, derived)

def merge_rankings_csv(path: str) -> Tuple[int, List[str]]:
    warnings: List[str] = []
    rows_merged = 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            # Pre-index PLAYERS_CACHE for fast lookup by (name,team,pos)
            index: Dict[str, Dict[str, Any]] = {}
            for p in PLAYERS_CACHE.values():
                index[_name_key(p["name"], p.get("team"), p.get("pos"))] = p

            for row in rdr:
                name = _parse_name(row)
                pos  = _parse_pos(row)
                team = _parse_team(row)
                if not name or not pos or not team:
                    continue
                key = _name_key(name, team, pos)
                target = index.get(key)
                if not target:
                    # fallback: match by name+pos ignoring team (aliases etc.)
                    nm = _norm(name)
                    for p in PLAYERS_CACHE.values():
                        if _norm(p["name"]) == nm and p.get("pos") == pos:
                            target = p
                            break
                if not target:
                    continue

                ecr = _parse_ecr_rank(row)
                adp = _parse_adp(row)
                proj = _parse_proj_points(row)
                _merge_into_player(target, ecr, adp, proj)
                rows_merged += 1
    except FileNotFoundError:
        warnings.append(f"{path} not found (skipping rankings merge).")
    except Exception as e:
        warnings.append(f"Error reading {path}: {e}")

    return rows_merged, warnings

def ensure_rankings_merged(force: bool=False) -> None:
    """Merge rankings.csv if never merged or if force=True."""
    global RANKINGS_ROWS, RANKINGS_WARNINGS, RANKINGS_LAST_MERGE_TS
    if force or RANKINGS_LAST_MERGE_TS == 0.0:
        rows, warns = merge_rankings_csv(RANKINGS_CSV_PATH)
        RANKINGS_ROWS = rows
        RANKINGS_WARNINGS = warns
        RANKINGS_LAST_MERGE_TS = time.time()

# ===== Draft utils =====
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

async def draft_state(draft_id: str, picks: List[Dict[str, Any]]) -> Dict[str, Any]:
    meta = await _get_json(f"{SLEEPER}/draft/{draft_id}")
    total_teams = int(meta.get("metadata", {}).get("teams") or meta.get("settings", {}).get("teams") or 0)
    if not total_teams:
        roster_ids = {p.get("roster_id") for p in picks if isinstance(p.get("roster_id"), int)}
        total_teams = len(roster_ids) or 12
    made = len([p for p in picks if p.get("player_id")])
    on_index = made + 1
    cur_round = max(1, math.ceil(on_index / total_teams))
    pick_in_round = ((on_index - 1) % total_teams) + 1
    snake_rev = (cur_round % 2 == 0)
    order_index = pick_in_round if not snake_rev else (total_teams - pick_in_round + 1)
    unique_rosters = sorted({p.get("roster_id") for p in picks if isinstance(p.get("roster_id"), int)})
    roster_on_clock = unique_rosters[order_index - 1] if 1 <= order_index <= len(unique_rosters) else None
    return {
        "draft_id": draft_id,
        "total_teams": total_teams,
        "picks_made": made,
        "current_round": cur_round,
        "pick_in_round": pick_in_round,
        "snake_reversed": snake_rev,
        "team_number_on_clock": order_index,
        "roster_id_on_clock": roster_on_clock
    }

# ===== Scoring =====
NEED_WEIGHTS = {"QB":1.1, "RB":1.45, "WR":1.45, "TE":1.25}

def roster_cap(pos: str, slots: Dict[str, int]) -> int:
    if pos in {"RB","WR"}:
        return slots.get(pos, 0) + slots.get("FLEX", 0) + 2
    if pos in {"TE","QB"}:
        return slots.get(pos, 0) + 1
    return slots.get(pos, 0) + 1

def compute_needs(my_ids: List[str], slots: Dict[str, int]) -> Dict[str, float]:
    counts = {k:0 for k in NEED_WEIGHTS.keys()}
    for pid in my_ids:
        pos = PLAYERS_CACHE.get(pid, {}).get("pos")
        if pos in counts:
            counts[pos] += 1
    needs: Dict[str, float] = {}
    for pos, base in NEED_WEIGHTS.items():
        gap = max(0, slots.get(pos, 0) - counts.get(pos, 0))
        needs[pos] = base if gap > 0 else base * 0.35
    return needs

def score_available(available: List[Dict[str, Any]],
                    my_ids: List[str],
                    slots: Dict[str, int],
                    pick_number: int) -> List[Dict[str, Any]]:
    needs = compute_needs(my_ids, slots)
    pos_counts: Dict[str, int] = {}
    for pid in my_ids:
        pos = PLAYERS_CACHE.get(pid, {}).get("pos")
        if pos:
            pos_counts[pos] = pos_counts.get(pos, 0) + 1

    adps = [p.get("adp") for p in available if isinstance(p.get("adp"), (int, float))]
    adp_median = sorted(adps)[len(adps)//2] if adps else None

    scored: List[Dict[str, Any]] = []
    for p in available:
        pos = p["pos"]
        proj = float(p.get("proj_ros") or p.get("proj_week") or 0.0)
        adp = p.get("adp")
        adp_discount = (adp_median - adp) if (adp_median is not None and isinstance(adp, (int, float))) else 0.0
        need_boost = needs.get(pos, 1.0)
        cur = pos_counts.get(pos, 0)
        cap = roster_cap(pos, slots)
        overdrafted = max(0, cur - cap)
        cap_penalty = 0.85 ** overdrafted

        score = proj + 0.15 * adp_discount
        score *= (0.85 + 0.30 * need_boost)
        score *= cap_penalty

        scored.append({
            "id": p["id"], "name": p.get("name"), "team": p.get("team"),
            "pos": pos, "bye": p.get("bye"),
            "adp": p.get("adp"),
            "proj_ros": p.get("proj_ros"), "proj_week": p.get("proj_week"),
            "score": round(score, 3),
            "explain": f"{pos} need {need_boost:.2f}, cap {cur}/{cap}, pick {pick_number}"
        })
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored

# ===== Models =====
class RecommendReq(BaseModel):
    draft_url: str
    roster_id: int
    pick_number: int
    season: int = 2025
    roster_slots: Optional[Dict[str, int]] = None
    limit: int = 10

# ===== Lifecycle =====
@app.on_event("startup")
async def startup():
    await load_players_if_needed()
    ensure_rankings_merged(force=True)

@app.get("/warmup")
def warmup():
    """Reload players and re-merge rankings.csv (for testing)."""
    loop = asyncio.get_event_loop()
    loop.run_until_complete(load_players_if_needed())
    ensure_rankings_merged(force=True)
    return {
        "ok": True,
        "players_cached": len(PLAYERS_CACHE),
        "rankings_rows": RANKINGS_ROWS,
        "rankings_warnings": RANKINGS_WARNINGS
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

# ===== Helper: team number -> roster_id + roster snapshot =====
@app.get("/draft/{draft_id}/team/{team_number}")
async def team_debug(draft_id: str, team_number: int, x_api_key: Optional[str] = Header(None)):
    auth_or_401(x_api_key)
    await load_players_if_needed()
    picks_json = await _get_json(f"{SLEEPER}/draft/{draft_id}/picks")
    picks = parse_picks(picks_json)
    unique_rosters = sorted({p.get("roster_id") for p in picks if isinstance(p.get("roster_id"), int)})
    if team_number < 1 or (unique_rosters and team_number > len(unique_rosters)):
        raise HTTPException(status_code=400, detail="Invalid team number for this draft.")
    roster_id = unique_rosters[team_number - 1] if unique_rosters else None
    my_picks = [p for p in picks if roster_id and p.get("roster_id") == roster_id and p.get("player_id")]
    roster = []
    for p in my_picks:
        pid = str(p["player_id"])
        meta = PLAYERS_CACHE.get(pid, {})
        roster.append({"id": pid, "name": meta.get("name"), "pos": meta.get("pos"), "team": meta.get("team")})
    state = await draft_state(draft_id, picks) if picks else {"note": "no picks yet"}
    return {"team_number": team_number, "roster_id": roster_id, "roster": roster, "state": state}

# ===== Main (soft-stop if live feed empty) =====
@app.post("/recommend_live")
async def recommend_live(body: RecommendReq, x_api_key: Optional[str] = Header(None)):
    auth_or_401(x_api_key)
    await load_players_if_needed()
    ensure_rankings_merged()  # merge once if not yet merged

    draft_id, picks_url = normalize_draft_picks_url(body.draft_url)
    picks_json = await _get_json(picks_url)
    picks = parse_picks(picks_json)

    if not picks:
        return {
            "status": "waiting_for_live_feed",
            "message": "Live picks feed is empty/invalid. Not proceeding with recommendations.",
            "draft_id": draft_id,
            "picks_url": picks_url,
            "recommended": [],
            "alternatives": [],
            "my_team": [],
            "drafted_count": 0,
            "my_team_count": 0,
            "ts": int(time.time())
        }

    state = await draft_state(draft_id, picks)
    drafted_ids, by_roster = parse_drafted_from_picks(picks)
    my_ids = list(by_roster.get(body.roster_id, set()))

    # Available pool
    available: List[Dict[str, Any]] = []
    for pid, pdata in PLAYERS_CACHE.items():
        if pid in drafted_ids: continue
        if pid in my_ids: continue
        available.append(pdata)

    if not available:
        return {
            "status": "no_available_after_filtering",
            "message": "No available players after filtering drafted and your roster.",
            "draft_state": state,
            "recommended": [],
            "alternatives": [],
            "my_team": [PLAYERS_CACHE[pid] for pid in my_ids if pid in PLAYERS_CACHE],
            "drafted_count": len(drafted_ids),
            "my_team_count": len(my_ids),
            "ts": int(time.time())
        }

    slots = body.roster_slots or {"QB":1,"RB":2,"WR":2,"TE":1,"FLEX":2}
    ranked = score_available(available, my_ids, slots, body.pick_number)
    limit = max(3, body.limit)
    topn = ranked[:limit]

    return {
        "status": "ok",
        "pick": body.pick_number,
        "season_used": body.season,
        "recommended": topn[:3],
        "alternatives": topn[3:limit],
        "my_team": [PLAYERS_CACHE[pid] for pid in my_ids if pid in PLAYERS_CACHE],
        "draft_state": state,
        "drafted_count": len(drafted_ids),
        "my_team_count": len(my_ids),
        "ts": int(time.time())
    }
