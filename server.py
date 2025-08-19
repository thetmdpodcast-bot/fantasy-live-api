from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import requests
import os

app = FastAPI()

# ==============================
# Load Rankings
# ==============================
RANKINGS_FILE = "rankings.csv"

def load_rankings():
    if not os.path.exists(RANKINGS_FILE):
        raise FileNotFoundError("rankings.csv not found.")
    df = pd.read_csv(RANKINGS_FILE)
    # Ensure required columns exist
    required_cols = {"player_name", "fantasy_points", "tds"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"rankings.csv missing required columns: {required_cols - set(df.columns)}")
    return df

rankings_df = load_rankings()

# ==============================
# Request Models
# ==============================
class RecommendRequest(BaseModel):
    drafted_players: list[str] = []

# ==============================
# Recommend Endpoint
# ==============================
@app.post("/recommend_live")
def recommend_live(req: RecommendRequest):
    try:
        available = rankings_df[~rankings_df["player_name"].isin(req.drafted_players)]
        if available.empty:
            raise HTTPException(status_code=400, detail="No available players to recommend.")

        # Sort by combination of fantasy points + touchdowns
        available["score"] = available["fantasy_points"] + available["tds"]
        best = available.sort_values("score", ascending=False).head(3)

        return {
            "recommendations": best[["player_name", "fantasy_points", "tds"]].to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==============================
# Draft / Team Monitoring Endpoint
# ==============================
@app.get("/draft/{draft_id}/team/{team_number}")
def get_team_roster(draft_id: str, team_number: int):
    try:
        # 1. Get draft metadata
        draft_url = f"https://api.sleeper.app/v1/draft/{draft_id}"
        draft_data = requests.get(draft_url).json()

        # 2. Get draft picks
        picks_url = f"https://api.sleeper.app/v1/draft/{draft_id}/picks"
        picks_data = requests.get(picks_url).json()

        # Fail safe: if picks_data is empty
        if not picks_data:
            raise HTTPException(status_code=500, detail="Unable to retrieve draft picks.")

        # 3. Map team number to roster_id
        roster_ids = sorted(set([p["roster_id"] for p in picks_data if "roster_id" in p]))
        if team_number < 1 or team_number > len(roster_ids):
            raise HTTPException(status_code=400, detail="Invalid team number")
        roster_id = roster_ids[team_number - 1]

        # 4. Build roster for this team
        team_picks = [p for p in picks_data if p.get("roster_id") == roster_id]
        roster = [
            f"{p['metadata'].get('first_name', '')} {p['metadata'].get('last_name', '')}".strip()
            for p in team_picks
        ]

        return {
            "team_number": team_number,
            "roster_id": roster_id,
            "roster": roster,
            "total_picks": len(team_picks)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==============================
# Health Check
# ==============================
@app.get("/")
def root():
    return {"status": "Fantasy Live API is running!"}
