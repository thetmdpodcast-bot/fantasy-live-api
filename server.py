# server.py — Sleeper live draft → name-join to rankings.csv ("AVG") + roster-aware ranking
# Adds: robust draft_state using slot_to_roster_id + /inspect_draft for debugging
# No pandas dependency

import os, time, asyncio, random, re, math, csv
from typing import Any, Dict, List, Optional, Set, Tuple
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import httpx

API_KEY = os.getenv("API_KEY")  # if set, require x-api-key header
SLEEPER = "https://api.sleeper.app/v1"
RANKINGS_CSV_PATH = os.getenv("RANKINGS_CSV_PATH", "rankings.csv")

ALLOWED_POS = {"QB","RB","WR","TE"}
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
                    max_attempts: int=3, base_backoff: float=0.25,
                    timeout: float=45.0) -> httpx.Response:
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
    parts = s.split(" ")
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
        if k in row and row[k] != "":
            return row[k]
    return None

def _parse_name(row: Dict[str,str]) -> Optional[str]:
    v = _row_get(row, "Player","Name","player","name","Full Name","FullName")
    if v: return re.sub(r"\s*\(.*?\)\s*$","",v).strip()
    return None

def _parse_pos(row: Dict[str,str]) -> Optional[str]:
    v = _row_get(row, "POS","Pos","Position","position","pos")
    if v:
        v = v.strip().upper()
        m = re.match(r"[A-Z]+", v)  # WR1 -> WR
        if m:
            v = m.group(0)
            if v in ALLOWED_POS:
                return v
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

async def draft_state(draft_id: str, picks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Use /draft/{id} to derive teams, round, pick, and who's on the clock.
    Prefers 'slot_to_roster_id' when present (correct Sleeper order).
    """
    meta = await _get_json(f"{SLEEPER}/draft/{draft_id}")
    total_teams = int(
        meta.get("metadata", {}).get("teams")
        or meta.get("settings", {}).get("teams")
        or 0
    )
    slot_to_roster = meta.get("slot_to_roster_id") or meta.get("slot_to_roster") or []
    made = len([p for p in picks if p.get("player_id")])

    if not total_teams:
        # Fallback: infer from picks we have
        roster_ids = {p.get("roster_id") for p in picks if isinstance(p.get("roster_id"), int)}
        total_teams = len(roster_ids) or 12

    on_index = made + 1
    cur_round = max(1, math.ceil(on_index / total_teams))
    pick_in_round = ((on_index - 1) % total_teams) + 1
    snake_rev = (cur_round % 2 == 0)

    if slot_to_roster and isinstance(slot_to_roster, list) and len(slot_to_roster) == total_teams:
        # Sleeper slots are 1-based positions in the snake order within the round
        order_index = pick_in_round if not snake_rev else (total_teams - pick_in_round + 1)
        rid_on_clock = slot_to_roster[order_index - 1]
        team_number_on_clock = order_index
    else:
        # Fallback: stable but approximate using sorted roster ids we saw
        unique_rosters = sorted({p.get("roster_id") for p in picks if isinstance(p.get("roster_id"), int)})
        order_index = pick_in_round if not snake_rev else (total_teams - pick_in_round + 1)
        rid_on_clock = unique_rosters[order_index - 1] if 1 <= order_index <= len(unique_rosters) else None
        team_number_on_clock = order_index

    return {
        "draft_id": draft_id,
        "total_teams": total_teams,
        "picks_made": made,
        "current_round": cur_round,
        "pick_in_round": pick_in_round,
        "snake_reversed": snake_rev,
        "team_number_on_clock": team_number_on_clock,
        "roster_id_on_clock": rid_on_clock
    }

# -------------------- helper: join UNDRAFTED with rankings --------------------
def match_players_with_rankings(undrafted_pids: List[str]) -> List[Dict[str, Any]]:
    """
    Join each undrafted Sleeper player to rankings.csv by normalized name.
    Only keep players that have a CSV match (prevents null rank/score spam).
    """
    result: List[Dict[str,Any]] = []
    for pid in undrafted_pids:
        p = PLAYERS_CACHE.get(pid)
        if not p: continue
        nkey = p.get("name_key")
        csv_rows = RANK_IDX.get(nkey, [])
        if not csv_rows:
            continue

        chosen = None
        for r in csv_rows:
            if r.get("pos") == p.get("pos"):
                chosen = r; break
        if not chosen: chosen = csv_rows[0]

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

        score = proj + 0.15*adp_discount
        score *= (0.85 + 0.30*need_boost)
        score *= cap_penalty

        out = {
            "id": p["id"], "name": p.get("name"), "team": p.get("team"),
            "pos": pos, "bye": p.get("bye"),
            "adp": p.get("adp"), "rank_avg": p.get("rank_avg"),
            "proj_ros": p.get("proj_ros"),
            "score": round(score,3),
            "explain": f"{pos} need {need_boost:.2f}, cap {cur}/{cap}, pick {pick_number}"
        }
        scored.append(out)

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored

# -------------------- request models --------------------
class RecommendReq(BaseModel):
    draft_url: str
    roster_id: int
    pick_number: int
    season: int = 2025
    roster_slots: Optional[Dict[str,int]] = None
    limit: int = 10

class InspectReq(BaseModel):
    draft_url: str
    roster_id: int

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
        "rankings_rows": RANK
# server.py — Sleeper live draft → name-join to rankings.csv ("AVG") + roster-aware ranking
# Adds: robust draft_state using slot_to_roster_id + /inspect_draft for debugging
# No pandas dependency

import os, time, asyncio, random, re, math, csv
from typing import Any, Dict, List, Optional, Set, Tuple
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import httpx

API_KEY = os.getenv("API_KEY")  # if set, require x-api-key header
SLEEPER = "https://api.sleeper.app/v1"
RANKINGS_CSV_PATH = os.getenv("RANKINGS_CSV_PATH", "rankings.csv")

ALLOWED_POS = {"QB","RB","WR","TE"}
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
                    max_attempts: int=3, base_backoff: float=0.25,
                    timeout: float=45.0) -> httpx.Response:
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
    parts = s.split(" ")
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
        if k in row and row[k] != "":
            return row[k]
    return None

def _parse_name(row: Dict[str,str]) -> Optional[str]:
    v = _row_get(row, "Player","Name","player","name","Full Name","FullName")
    if v: return re.sub(r"\s*\(.*?\)\s*$","",v).strip()
    return None

def _parse_pos(row: Dict[str,str]) -> Optional[str]:
    v = _row_get(row, "POS","Pos","Position","position","pos")
    if v:
        v = v.strip().upper()
        m = re.match(r"[A-Z]+", v)  # WR1 -> WR
        if m:
            v = m.group(0)
            if v in ALLOWED_POS:
                return v
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

async def draft_state(draft_id: str, picks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Use /draft/{id} to derive teams, round, pick, and who's on the clock.
    Prefers 'slot_to_roster_id' when present (correct Sleeper order).
    """
    meta = await _get_json(f"{SLEEPER}/draft/{draft_id}")
    total_teams = int(
        meta.get("metadata", {}).get("teams")
        or meta.get("settings", {}).get("teams")
        or 0
    )
    slot_to_roster = meta.get("slot_to_roster_id") or meta.get("slot_to_roster") or []
    made = len([p for p in picks if p.get("player_id")])

    if not total_teams:
        # Fallback: infer from picks we have
        roster_ids = {p.get("roster_id") for p in picks if isinstance(p.get("roster_id"), int)}
        total_teams = len(roster_ids) or 12

    on_index = made + 1
    cur_round = max(1, math.ceil(on_index / total_teams))
    pick_in_round = ((on_index - 1) % total_teams) + 1
    snake_rev = (cur_round % 2 == 0)

    if slot_to_roster and isinstance(slot_to_roster, list) and len(slot_to_roster) == total_teams:
        # Sleeper slots are 1-based positions in the snake order within the round
        order_index = pick_in_round if not snake_rev else (total_teams - pick_in_round + 1)
        rid_on_clock = slot_to_roster[order_index - 1]
        team_number_on_clock = order_index
    else:
        # Fallback: stable but approximate using sorted roster ids we saw
        unique_rosters = sorted({p.get("roster_id") for p in picks if isinstance(p.get("roster_id"), int)})
        order_index = pick_in_round if not snake_rev else (total_teams - pick_in_round + 1)
        rid_on_clock = unique_rosters[order_index - 1] if 1 <= order_index <= len(unique_rosters) else None
        team_number_on_clock = order_index

    return {
        "draft_id": draft_id,
        "total_teams": total_teams,
        "picks_made": made,
        "current_round": cur_round,
        "pick_in_round": pick_in_round,
        "snake_reversed": snake_rev,
        "team_number_on_clock": team_number_on_clock,
        "roster_id_on_clock": rid_on_clock
    }

# -------------------- helper: join UNDRAFTED with rankings --------------------
def match_players_with_rankings(undrafted_pids: List[str]) -> List[Dict[str, Any]]:
    """
    Join each undrafted Sleeper player to rankings.csv by normalized name.
    Only keep players that have a CSV match (prevents null rank/score spam).
    """
    result: List[Dict[str,Any]] = []
    for pid in undrafted_pids:
        p = PLAYERS_CACHE.get(pid)
        if not p: continue
        nkey = p.get("name_key")
        csv_rows = RANK_IDX.get(nkey, [])
        if not csv_rows:
            continue

        chosen = None
        for r in csv_rows:
            if r.get("pos") == p.get("pos"):
                chosen = r; break
        if not chosen: chosen = csv_rows[0]

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

        score = proj + 0.15*adp_discount
        score *= (0.85 + 0.30*need_boost)
        score *= cap_penalty

        out = {
            "id": p["id"], "name": p.get("name"), "team": p.get("team"),
            "pos": pos, "bye": p.get("bye"),
            "adp": p.get("adp"), "rank_avg": p.get("rank_avg"),
            "proj_ros": p.get("proj_ros"),
            "score": round(score,3),
            "explain": f"{pos} need {need_boost:.2f}, cap {cur}/{cap}, pick {pick_number}"
        }
        scored.append(out)

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored

# -------------------- request models --------------------
class RecommendReq(BaseModel):
    draft_url: str
    roster_id: int
    pick_number: int
    season: int = 2025
    roster_slots: Optional[Dict[str,int]] = None
    limit: int = 10

class InspectReq(BaseModel):
    draft_url: str
    roster_id: int

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
        "rankings_rows": RANK
