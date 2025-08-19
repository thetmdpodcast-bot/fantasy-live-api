# FastAPI backend for Live Draft Tracking
# Features:
# - /recommend_live: returns Top 3 undrafted, roster-aware picks
# - /log_pick: optional self-log to reflect your pick immediately
# - /health: liveness
# Provider: Sleeper (no auth required). Add others later.

import os, time, math, asyncio
from typing import List, Dict, Optional
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import httpx

API_KEY = os.getenv("API_KEY")  # simple shared secret for your GPT Action
SLEEPER = "https://api.sleeper.app/v1"

app = FastAPI(title="Fantasy Live Draft API")

# ======== simple caches (in-memory) ========
PLAYERS_CACHE: Dict[str, Dict] = {}   # id -> player dict
PLAYERS_LOADED_AT: float = 0

async def load_players_if_needed() -> None:
    global PLAYERS_CACHE, PLAYERS_LOADED_AT
    # refresh every 3 hours
    if time.time() - PLAYERS_LOADED_AT < 3*3600 and PLAYERS_CACHE:
        return
    async with httpx.AsyncClient(timeout=30) as client:
        # Sleeper: /players/nfl returns a big dict keyed by player_id
        r = await client.get(f"{SLEEPER}/players/nfl")
        r.raise_for_status()
        data = r.json()
    # normalize a light map
    PLAYERS_CACHE = {}
    for pid, p in data.items():
        PLAYERS_CACHE[pid] = {
            "id": pid,
            "name": p.get("full_name") or p.get("first_name","")+" "+p.get("last_name",""),
            "team": p.get("team"),
            "pos": p.get("position"),
            "bye": p.get("bye_week"),
            # you can enrich later from projections/ADP sources
            "adp": None, "proj_ros": None, "proj_week": None
        }
    PLAYERS_LOADED_AT = time.time()

# ======== Sleeper helpers ========
async def sleeper_get(url: str):
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(url)
        r.raise_for_status()
        return r.json()

async def get_drafted_player_ids(draft_id: str) -> List[str]:
    picks = await sleeper_get(f"{SLEEPER}/draft/{draft_id}/picks")
    # Each pick has player_id (Sleeper's internal), e.g., '4034'
    return [p["player_id"] for p in picks if p.get("player_id")]

async def get_my_team_player_ids(league_id: str, team_id: Optional[str], team_name: Optional[str]) -> List[str]:
    rosters = await sleeper_get(f"{SLEEPER}/league/{league_id}/rosters")
    users   = await sleeper_get(f"{SLEEPER}/league/{league_id}/users")
    # Map display_name/team_name to roster_id
    name_to_roster = {}
    for u in users:
        # Sleeper users don't always have the fantasy "team name" here—fallback to display_name
        name_to_roster[u.get("metadata",{}).get("team_name") or u.get("display_name")] = u["user_id"]

    roster_for_user = { r["owner_id"]: r for r in rosters if r.get("owner_id")}
    roster = None

    if team_id:  # Sleeper team_id is actually roster_id in many UIs; accept either
        roster = next((r for r in rosters if str(r.get("roster_id"))==str(team_id) or r.get("owner_id")==team_id), None)

    if not roster and team_name:
        owner_id = name_to_roster.get(team_name)
        if owner_id:
            roster = roster_for_user.get(owner_id)

    if not roster:
        return []

    # 'players' includes all players on roster (drafted or kept)
    return roster.get("players") or []

def need_weights():
    return {"QB":1.1, "RB":1.45, "WR":1.45, "TE":1.25, "DST":0.5, "K":0.4}

def flexable():
    return {"RB","WR","TE"}

def roster_sane_cap(pos: str, roster_slots: Dict[str,int]) -> int:
    if pos in {"RB","WR"}:
        return roster_slots.get(pos,0) + roster_slots.get("FLEX",0) + 2
    if pos in {"TE","QB"}:
        return roster_slots.get(pos,0) + 1
    return roster_slots.get(pos,0) + 1

def compute_needs(my_ids: List[str], roster_slots: Dict[str,int]) -> Dict[str,float]:
    counts = {"QB":0,"RB":0,"WR":0,"TE":0,"DST":0,"K":0}
    for pid in my_ids:
        pos = PLAYERS_CACHE.get(pid,{}).get("pos")
        if pos in counts: counts[pos]+=1
    needs = {}
    for pos, base in need_weights().items():
        gap = max(0, roster_slots.get(pos,0) - counts.get(pos,0))
        needs[pos] = base * (1.0 if gap>0 else 0.35)
    return needs

# ======== Models & endpoints ========
class RecommendReq(BaseModel):
    league_id: str
    draft_id: str
    pick_number: int
    team_id: Optional[str] = None
    team_name: Optional[str] = None
    roster_slots: Optional[Dict[str,int]] = None  # allow override
    limit: int = 10

def auth_or_401(x_api_key: Optional[str]):
    if not API_KEY:
        return  # unsecured (dev)
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

@app.get("/health")
def health(): return {"ok": True, "ts": int(time.time())}

@app.post("/recommend_live")
async def recommend_live(body: RecommendReq, x_api_key: Optional[str] = Header(None)):
    auth_or_401(x_api_key)
    await load_players_if_needed()

    drafted = set(await get_drafted_player_ids(body.draft_id))
    mine    = set(await get_my_team_player_ids(body.league_id, body.team_id, body.team_name))

    # basic roster slots (adjust to your league if you want to fetch from elsewhere)
    roster_slots = body.roster_slots or {"QB":1,"RB":2,"WR":2,"TE":1,"FLEX":2,"DST":1,"K":1}

    # availability
    available_ids = [pid for pid in PLAYERS_CACHE if pid not in drafted and pid not in mine]
    available = [PLAYERS_CACHE[pid] for pid in available_ids if PLAYERS_CACHE[pid].get("pos")]

    # quick ADP baseline if you later enrich players with ADP
    adps = [p.get("adp",999) for p in available if p.get("adp") is not None]
    adp_median = sorted(adps)[len(adps)//2] if adps else 999

    needs = compute_needs(list(mine), roster_slots)

    # count current by pos for cap
    pos_counts = {}
    for pid in mine:
        pos = PLAYERS_CACHE.get(pid,{}).get("pos")
        if pos: pos_counts[pos] = pos_counts.get(pos,0)+1

    scored = []
    for p in available:
        pos = p["pos"]
        proj = float(p.get("proj_ros") or p.get("proj_week") or 0.0)  # plug projections later
        adp_discount = (adp_median - float(p.get("adp", adp_median)))
        need_boost = needs.get(pos,1.0)
        cur = pos_counts.get(pos,0)
        cap = roster_sane_cap(pos, roster_slots)
        overdrafted = max(0, cur - cap)
        cap_penalty = 0.85 ** overdrafted

        score = proj + 0.18*adp_discount
        score *= (0.85 + 0.3*need_boost)
        score *= cap_penalty

        scored.append({
            "id": p["id"], "name": p["name"], "team": p.get("team"),
            "pos": pos, "bye": p.get("bye"), "adp": p.get("adp"),
            "proj_ros": p.get("proj_ros"), "score": round(score,3),
            "explain": f"{pos} need {need_boost:.2f}, cap {cur}/{cap}"
        })

    ranked = sorted(scored, key=lambda x: x["score"], reverse=True)[:body.limit]
    return {
        "pick": body.pick_number,
        "recommended": ranked[:3],
        "alternatives": ranked[3:10],
        "drafted_count": len(drafted),
        "my_team_count": len(mine),
        "ts": int(time.time())
    }

class LogPickReq(BaseModel):
    league_id: str
    draft_id: str
    player_id: str

# optional: store instantly so your GPT reflects your pick even before Sleeper updates
PICKS_LOG = set()

@app.post("/log_pick")
def log_pick(body: LogPickReq, x_api_key: Optional[str] = Header(None)):
    auth_or_401(x_api_key)
    PICKS_LOG.add((body.draft_id, body.player_id))
    return {"ok": True, "stored": list(PICKS_LOG)}
