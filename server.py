from fastapi import FastAPI, Request
import csv

app = FastAPI()

# ------------------------
# Load rankings without pandas
# ------------------------
def load_rankings(filepath="rankings.csv"):
    rankings = []
    try:
        with open(filepath, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rankings.append({
                    "name": row.get("Player", "").strip(),
                    "pos": row.get("Pos", "").strip(),
                    "team": row.get("Team", "").strip(),
                    "fantasy_pts": float(row.get("FantasyPts", 0) or 0),
                    "tds": float(row.get("TDs", 0) or 0),
                })
    except FileNotFoundError:
        print(f"⚠️ rankings.csv not found in path {filepath}")
    return rankings


# Keep rankings in memory
RANKINGS = load_rankings()


# ------------------------
# Health check
# ------------------------
@app.get("/health")
async def health():
    return {"ok": True, "players_cached": len(RANKINGS)}


# ------------------------
# Inspect Draft (basic example)
# ------------------------
@app.post("/inspect_draft")
async def inspect_draft(req: Request):
    body = await req.json()
    draft_url = body.get("draft_url", "")
    roster_id = body.get("roster_id", None)

    # Example response with just top 5 players
    top_preview = sorted(
        RANKINGS,
        key=lambda x: (x["fantasy_pts"], x["tds"]),
        reverse=True
    )[:5]

    return {
        "ok": True,
        "draft_url": draft_url,
        "roster_id": roster_id,
        "csv_top_preview": top_preview
    }


# ------------------------
# Guess Roster (basic example)
# ------------------------
@app.post("/guess_roster")
async def guess_roster(req: Request):
    body = await req.json()
    draft_url = body.get("draft_url", "")
    player_names = body.get("player_names", [])

    team_players = [p for p in RANKINGS if p["name"] in player_names]

    return {
        "ok": True,
        "draft_url": draft_url,
        "players": team_players
    }
