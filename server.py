import os
import httpx
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Set

# -------------------- Load Rankings --------------------
RANKINGS_FILE = "rankings.csv"
if not os.path.exists(RANKINGS_FILE):
    raise FileNotFoundError("rankings.csv is required but not found.")

rankings_df = pd.read_csv(RANKINGS_FILE)
# Normalize names for safer joins
rankings_df["name_key"] = rankings_df["Player"].str.lower().str.strip()

# -------------------- FastAPI --------------------
app = FastAPI()

# Sleeper base
SLEEPER_BASE = "https://api.sleeper.app/v1"

# -------------------- Models --------------------
class RecommendBody(BaseModel):
    draft_id: str
    roster_id: int  # team slot or roster_id

class InspectBody(BaseModel):
    draft_id: str
    roster_id: int

# -------------------- Cache for players --------------------
PLAYERS_CACHE: Dict[str, Dict[str, Any]] = {}

async def load_players():
    """Preload players cache from Sleeper if empty."""
    global PLAYERS_CACHE
    if PLAYERS_CACHE:
        return
    url = f"{SLEEPER_BASE}/players/nfl"
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        for pid, pdata in data.items():
            key = (pdata.get("full_name") or "").lower().strip()
            PLAYERS_CACHE[pid] = {
                "id": pid,
                "name": pdata.get("full_name"),
                "name_key": key,
                "position": pdata.get("position"),
                "team": pdata.get("team")
            }

# -------------------- Helpers --------------------
def norm_name(name: str) -> str:
    return (name or "").lower().strip()

def _pick_fullname_key(pick: Dict[str,Any]) -> Optional[str]:
    md = pick.get("metadata") or {}
    fn = (md.get("first_name") or "").strip()
    ln = (md.get("last_name") or "").strip()
    if fn or ln:
        return norm_name(f"{fn} {ln}".strip())
    pid = str(pick.get("player_id") or "")
    if pid in PLAYERS_CACHE:
        return PLAYERS_CACHE[pid].get("name_key")
    return None

def resolve_roster_id(roster_id: int, picks: List[Dict[str,Any]], by_roster: Dict[int, Set[str]], total_teams: int) -> int:
    """
    Try to resolve a safe roster_id:
      1. Direct roster_id if valid
      2. Guess by matching drafted players against PLAYERS_CACHE
    """
    if roster_id in by_roster:
        return roster_id

    # fallback: try to guess based on players already drafted
    scores: Dict[int, int] = {}
    for p in picks:
        rid = p.get("roster_id")
        if not isinstance(rid, int): 
            continue
        key = _pick_fullname_key(p)
        if key and key in PLAYERS_CACHE:
            scores[rid] = scores.get(rid, 0) + 1

    if scores:
        best = max(scores.items(), key=lambda kv: kv[1])[0]
        return best

    return roster_id  # fallback to original if no clue

def recommend_from_pool(available_ids: List[str], my_ids: List[str]) -> List[Dict[str,Any]]:
    """
    Recommend best players from rankings.csv, sorted by AVG.
    """
    if not available_ids:
        return []

    available_keys = [PLAYERS_CACHE[pid]["name_key"] for pid in available_ids if pid in PLAYERS_CACHE]
    my_keys = [PLAYERS_CACHE[pid]["name_key"] for pid in my_ids if pid in PLAYERS_CACHE]

    sub = rankings_df[rankings_df["name_key"].isin(available_keys)].copy()
    if sub.empty:
        return []

    # sort ascending by AVG (lower = better draft value)
    sub = sub.sort_values(by="AVG", ascending=True)

    recs = []
    for _, row in sub.head(10).iterrows():
        recs.append({
            "player": row["Player"],
            "team": row.get("Team", ""),
            "pos": row.get("POS", ""),
            "avg_rank": row.get("AVG"),
            "on_my_team": row["name_key"] in my_keys
        })
    return recs

# -------------------- Endpoints --------------------
@app.post("/inspect_draft")
async def inspect_draft(body: InspectBody):
    await load_players()
    draft_url = f"{SLEEPER_BASE}/draft/{body.draft_id}"
    picks_url = f"{SLEEPER_BASE}/draft/{body.draft_id}/picks"

    async with httpx.AsyncClient() as client:
        draft_resp = await client.get(draft_url)
        draft_resp.raise_for_status()
        draft_info = draft_resp.json()

        picks_resp = await client.get(picks_url)
        picks_resp.raise_for_status()
        picks = picks_resp.json()

    # group drafted by roster
    by_roster: Dict[int, Set[str]] = {}
    drafted_ids: Set[str] = set()
    for p in picks:
        rid = p.get("roster_id")
        pid = str(p.get("player_id") or "")
        drafted_ids.add(pid)
        by_roster.setdefault(rid, set()).add(pid)

    eff_roster_id = resolve_roster_id(body.roster_id, picks, by_roster, draft_info.get("total_teams", 12))
    my_ids = list(by_roster.get(eff_roster_id, set()))

    return {
        "draft_id": body.draft_id,
        "effective_roster_id": eff_roster_id,
        "my_team": [PLAYERS_CACHE[pid]["name"] for pid in my_ids if pid in PLAYERS_CACHE],
        "drafted_total": len(drafted_ids)
    }

@app.post("/recommend_live")
async def recommend_live(body: RecommendBody):
    await load_players()
    draft_url = f"{SLEEPER_BASE}/draft/{body.draft_id}"
    picks_url = f"{SLEEPER_BASE}/draft/{body.draft_id}/picks"

    async with httpx.AsyncClient() as client:
        draft_resp = await client.get(draft_url)
        draft_resp.raise_for_status()
        draft_info = draft_resp.json()

        picks_resp = await client.get(picks_url)
        picks_resp.raise_for_status()
        picks = picks_resp.json()

    by_roster: Dict[int, Set[str]] = {}
    drafted_ids: Set[str] = set()
    for p in picks:
        rid = p.get("roster_id")
        pid = str(p.get("player_id") or "")
        drafted_ids.add(pid)
        by_roster.setdefault(rid, set()).add(pid)

    eff_roster_id = resolve_roster_id(body.roster_id, picks, by_roster, draft_info.get("total_teams", 12))
    my_ids = list(by_roster.get(eff_roster_id, set()))

    # available = not yet drafted
    available_ids = [pid for pid in PLAYERS_CACHE if pid not in drafted_ids]

    recs = recommend_from_pool(available_ids, my_ids)

    return {
        "draft_id": body.draft_id,
        "effective_roster_id": eff_roster_id,
        "recommendations": recs[:5],
        "my_team": [PLAYERS_CACHE[pid]["name"] for pid in my_ids if pid in PLAYERS_CACHE]
    }

# -------------------- Root --------------------
@app.get("/")
def root():
    return {"ok": True, "msg": "Fantasy Draft Assistant is running."}
