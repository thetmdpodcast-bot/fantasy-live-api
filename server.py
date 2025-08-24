import os, csv, time, json, math, re
from typing import Any, Dict, List, Optional, Tuple
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

API_KEY = os.getenv("API_KEY") or os.getenv("FANTASY_API_KEY") or os.getenv("X_API_KEY")

app = FastAPI(title="Fantasy Live Draft API", version="1.0.6",
              description="Sleeper live draft helper that merges a local rankings.csv with live draft data.")

# ---------------------------
# Rankings cache + loader
# ---------------------------

RANKINGS_PATH = os.getenv("RANKINGS_PATH", "rankings.csv")

_rankings_cache: Dict[str, Any] = {
    "rows": [],               # normalized rows [{name,team,pos,rank_avg,raw:*}]
    "index_by_name": {},      # name_normalized -> row
    "loaded_ts": 0.0,         # file mtime used
    "warnings": [],           # header/parse warnings
}

HEADER_MAP = {
    # CSV synonyms -> canonical keys we use
    "player": "name",
    "name": "name",
    "team": "team",
    "pos": "pos",
    "position": "pos",
    "avg": "rank_avg",
    "rank_avg": "rank_avg",
    "rank": "rank",  # not required, but we keep it if present
    "bye": "bye",
    "yahoo": "yahoo",
    "sleeper": "sleeper",
    "rtsports": "rtsports",
}

def _norm_hdr(h: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", h.strip().strip('"').lower())

def _norm_name(n: str) -> str:
    n = (n or "").strip()
    n = re.sub(r"\s+", " ", n)
    return n

def _to_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip().strip('"').replace(",", "")
    if s == "" or s.lower() in {"na", "n/a", "none"}:
        return None
    try:
        return float(s)
    except:
        # Sometimes "8.7\"" etc—strip trailing non-numeric
        m = re.match(r"^-?\d+(\.\d+)?", s)
        return float(m.group(0)) if m else None

def load_rankings(force: bool = False) -> Dict[str, Any]:
    """Load rankings.csv and normalize columns without requiring you to edit the file."""
    try:
        mtime = os.path.getmtime(RANKINGS_PATH)
    except FileNotFoundError:
        _rankings_cache.update({"rows": [], "index_by_name": {}, "loaded_ts": 0.0,
                                "warnings": [f"{RANKINGS_PATH} not found"]})
        return _rankings_cache

    if not force and _rankings_cache["loaded_ts"] == mtime and _rankings_cache["rows"]:
        return _rankings_cache

    rows: List[Dict[str, Any]] = []
    index: Dict[str, Dict[str, Any]] = []
    warnings: List[str] = []

    with open(RANKINGS_PATH, newline="", encoding="utf-8-sig") as f:
        rdr = csv.reader(f)
        raw_hdr = next(rdr)
        # map headers
        mapped = []
        for h in raw_hdr:
            key = HEADER_MAP.get(_norm_hdr(h), _norm_hdr(h))
            mapped.append(key)

        # we expect to have at least these
        needed = {"name", "team", "pos"}
        if not any(k in mapped for k in ("rank_avg", "avg")):
            warnings.append("No rank_avg/AVG column detected; recommendations may be limited.")

        # build rows
        for r in rdr:
            row = {mapped[i]: r[i].strip() if i < len(r) else "" for i in range(len(mapped))}
            name = _norm_name(row.get("name") or row.get("player") or "")
            if not name:
                continue

            # use AVG if rank_avg missing
            rank_avg = row.get("rank_avg") or row.get("avg")
            rank_val = _to_float(rank_avg)

            norm = {
                "name": name,
                "team": (row.get("team") or "").strip(),
                "pos": (row.get("pos") or "").strip(),
                "rank_avg": rank_val,
                # keep a few useful raw fields if present
                "raw": {
                    "rank": row.get("rank"), "yahoo": row.get("yahoo"),
                    "sleeper": row.get("sleeper"), "rtsports": row.get("rtsports"),
                    "bye": row.get("bye"), "avg": rank_avg,
                },
            }
            rows.append(norm)

    # build index by normalized name for quick match
    index_by_name = {norm["name"].lower(): norm for norm in rows}

    _rankings_cache.update({
        "rows": rows,
        "index_by_name": index_by_name,
        "loaded_ts": mtime,
        "warnings": warnings,
    })
    return _rankings_cache

# ---------------------------
# Models
# ---------------------------

class RecommendLiveRequest(BaseModel):
    draft_url: Optional[str] = None
    league_id: Optional[str] = None
    roster_id: Optional[int] = None
    team_slot: Optional[int] = None
    team_name: Optional[str] = None
    pick_number: Optional[int] = None
    season: Optional[int] = 2025
    limit: int = 10

# ---------------------------
# Small utilities
# ---------------------------

def _auth_check(x_api_key: Optional[str]) -> None:
    if API_KEY and (x_api_key or "") != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

def _top_from_rankings(limit: int, pick_number: Optional[int]) -> Tuple[List[Dict[str, Any]], List[str]]:
    r = load_rankings()
    reasons = []

    if r["warnings"]:
        reasons.extend([f"csv_warning:{w}" for w in r["warnings"]])

    # keep only rows that have a numeric rank_avg
    usable = [row for row in r["rows"] if isinstance(row.get("rank_avg"), (float, int))]
    usable.sort(key=lambda x: (x.get("rank_avg"), x["name"]))

    recommended = []
    for row in usable[: max(limit, 10)]:  # give a bit more headroom
        recommended.append({
            "id": None,
            "name": row["name"],
            "team": row["team"] or None,
            "pos": row["pos"] or None,
            "rank_avg": row["rank_avg"],
            "adp": None,
            "proj_ros": None,
            "score": row["rank_avg"],
            "explain": f"rank={row['rank_avg']}, pick {pick_number}" if pick_number else f"rank={row['rank_avg']}",
        })

    if not recommended:
        reasons.append("fallback1_primary_empty")

    return recommended[:limit], reasons

# ---------------------------
# Endpoints
# ---------------------------

@app.get("/echo_auth")
def echo_auth(x_api_key: Optional[str] = Header(None)):
    got = x_api_key is not None
    got_len = len(x_api_key) if x_api_key else 0
    exp_present = API_KEY is not None
    match = bool(API_KEY and x_api_key == API_KEY)
    return {
        "ok": True,
        "got_present": got,
        "got_len": got_len,
        "exp_present": exp_present,
        "match": match,
        "ts": int(time.time()),
    }

@app.get("/health")
def health():
    r = load_rankings()
    return {
        "ok": True,
        "players_cached": None,  # left for compatibility
        "players_raw": None,
        "players_kept": None,
        "players_ttl_sec": None,
        "rankings_rows": len(r["rows"]),
        "rankings_last_merge": None,
        "rankings_warnings": r["warnings"],
        "ts": int(time.time()),
    }

@app.get("/warmup")
def warmup():
    r = load_rankings(force=True)
    return {
        "ok": True,
        "players_cached": None,
        "players_raw": None,
        "players_kept": None,
        "rankings_rows": len(r["rows"]),
        "rankings_warnings": r["warnings"],
        "ts": int(time.time()),
    }

@app.post("/recommend_live")
def recommend_live(req: RecommendLiveRequest, x_api_key: Optional[str] = Header(None)):
    _auth_check(x_api_key)

    # Try to use CSV-only fallback if live data isn’t available.
    recommended, reasons = _top_from_rankings(limit=req.limit, pick_number=req.pick_number)

    # The rest of this object shape mirrors what you were already seeing in Builder
    return {
        "status": "ok",
        "pick": req.pick_number,
        "season_used": req.season or 2025,
        "recommended": recommended,
        "alternatives": [],
        "my_team": [],  # filled when you integrate the live board / roster glue
        "draft_state": {
            "draft_id": None if not req.draft_url else req.draft_url.split("/")[-1],
            "picks_made": None,
        },
        "effective_roster_id": req.roster_id,
        "drafted_count": None,
        "my_team_count": 0,
        "debug_reason": reasons,
        "ts": int(time.time()),
    }

# If you had other endpoints (inspect_draft, guess_roster, etc.) in your previous file,
# keep them as-is. This file focuses on CSV parsing + recommend_live behavior.
