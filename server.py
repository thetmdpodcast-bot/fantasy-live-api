{
  "openapi": "3.1.0",
  "info": {
    "title": "Fantasy Live Draft API",
    "version": "1.0.5",
    "description": "Sleeper live draft helper that merges a local rankings.csv with live draft data."
  },
  "servers": [{ "url": "https://fantasy-live-api.onrender.com" }],
  "security": [{ "ApiKeyAuth": [] }],
  "paths": {
    "/health": {
      "get": {
        "operationId": "health",
        "summary": "Health check",
        "security": [],
        "responses": {
          "200": {
            "description": "Service health",
            "content": {
              "application/json": {
                "schema": { "$ref": "#/components/schemas/HealthResponse" }
              }
            }
          }
        }
      }
    },
    "/warmup": {
      "get": {
        "operationId": "warmup",
        "summary": "Warm caches (players, rankings index)",
        "security": [],
        "responses": {
          "200": {
            "description": "Warmup status",
            "content": {
              "application/json": {
                "schema": { "$ref": "#/components/schemas/WarmupResponse" }
              }
            }
          }
        }
      }
    },
    "/echo_auth": {
      "get": {
        "operationId": "echo_auth",
        "summary": "Debug: verify API key presence/match",
        "security": [],
        "responses": {
          "200": {
            "description": "Echo auth",
            "content": {
              "application/json": {
                "schema": { "$ref": "#/components/schemas/EchoAuthResponse" }
              }
            }
          }
        }
      }
    },
    "/inspect_draft": {
      "post": {
        "operationId": "inspect_draft",
        "summary": "Inspect a draft and resolve the effective roster/team",
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": { "$ref": "#/components/schemas/InspectDraftRequest" }
            }
          }
        },
        "responses": {
          "200": {
            "description": "Draft inspection",
            "content": {
              "application/json": {
                "schema": { "$ref": "#/components/schemas/InspectDraftResponse" }
              }
            }
          }
        }
      }
    },
    "/guess_roster": {
      "post": {
        "operationId": "guess_roster",
        "summary": "Guess roster_id from a few drafted player names",
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": { "$ref": "#/components/schemas/GuessRosterRequest" }
            }
          }
        },
        "responses": {
          "200": {
            "description": "Roster guess results",
            "content": {
              "application/json": {
                "schema": { "$ref": "#/components/schemas/GuessRosterResponse" }
              }
            }
          }
        }
      }
    },
    "/recommend_live": {
      "post": {
        "operationId": "recommend_live",
        "summary": "Recommend the best available players during a live draft",
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": { "$ref": "#/components/schemas/RecommendLiveRequest" }
            }
          }
        },
        "responses": {
          "200": {
            "description": "Live recommendations",
            "content": {
              "application/json": {
                "schema": { "$ref": "#/components/schemas/RecommendLiveResponse" }
              }
            }
          }
        }
      }
    }
  },
  "components": {
    "securitySchemes": {
      "ApiKeyAuth": { "type": "apiKey", "in": "header", "name": "x-api-key" }
    },
    "schemas": {
      "EchoAuthResponse": {
        "type": "object",
        "properties": {
          "ok": { "type": "boolean" },
          "got_present": { "type": "boolean" },
          "got_len": { "type": "integer" },
          "exp_present": { "type": "boolean" },
          "match": { "type": "boolean" }
        },
        "required": ["ok", "got_present", "got_len", "exp_present", "match"]
      },
      "HealthResponse": {
        "type": "object",
        "properties": {
          "ok": { "type": "boolean" },
          "players_cached": { "type": "integer" },
          "players_raw": { "type": "integer" },
          "players_kept": { "type": "integer" },
          "players_ttl_sec": { "type": ["integer", "null"] },
          "rankings_rows": { "type": "integer" },
          "rankings_last_merge": { "type": ["integer", "null"] },
          "rankings_warnings": { "type": "array", "items": { "type": "string" } },
          "ts": { "type": "integer" }
        },
        "additionalProperties": true
      },
      "WarmupResponse": {
        "type": "object",
        "properties": {
          "ok": { "type": "boolean" },
          "players_cached": { "type": "integer" },
          "players_raw": { "type": "integer" },
          "players_kept": { "type": "integer" },
          "rankings_rows": { "type": "integer" },
          "rankings_warnings": { "type": "array", "items": { "type": "string" } },
          "ts": { "type": "integer" }
        },
        "additionalProperties": true
      },
      "InspectDraftRequest": {
        "type": "object",
        "properties": {
          "draft_url": { "type": "string" },
          "league_id": { "type": "string" },
          "roster_id": { "type": "integer" },
          "team_slot": { "type": "integer" },
          "team_name": { "type": "string" }
        },
        "oneOf": [ { "required": ["draft_url"] }, { "required": ["league_id"] } ],
        "additionalProperties": false
      },
      "InspectDraftResponse": {
        "type": "object",
        "properties": {
          "status": { "type": "string" },
          "draft_state": { "type": "object", "additionalProperties": true },
          "slot_to_roster_raw": { "type": ["object", "null"], "additionalProperties": true },
          "slot_to_roster_normalized": { "type": ["array", "null"], "items": { "type": ["integer", "null"] } },
          "observed_roster_ids": { "type": "array", "items": { "type": "integer" } },
          "by_roster_counts": { "type": "object", "additionalProperties": { "type": "integer" } },
          "input": { "type": "object", "additionalProperties": true },
          "effective_roster_id": { "type": ["integer", "null"] },
          "effective_team_slot": { "type": ["integer", "null"] },
          "my_team": { "type": "array", "items": { "type": "object", "additionalProperties": true } },
          "drafted_count": { "type": "integer" },
          "my_team_count": { "type": "integer" },
          "undrafted_count": { "type": "integer" },
          "csv_matched_count": { "type": "integer" },
          "csv_top_preview": { "type": "array", "items": { "type": "object", "additionalProperties": true } },
          "ts": { "type": "integer" }
        },
        "additionalProperties": true
      },
      "GuessRosterRequest": {
        "type": "object",
        "required": ["draft_url", "player_names"],
        "properties": {
          "draft_url": { "type": "string" },
          "player_names": { "type": "array", "items": { "type": "string" }, "minItems": 1 }
        },
        "additionalProperties": false
      },
      "GuessRosterResponse": {
        "type": "object",
        "properties": {
          "status": { "type": "string" },
          "draft_id": { "type": "string" },
          "candidates": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "roster_id": { "type": "integer" },
                "matches": { "type": "integer" },
                "players": { "type": "array", "items": { "type": "string" } }
              },
              "additionalProperties": false
            }
          },
          "guessed_roster_id": { "type": ["integer", "null"] },
          "note": { "type": "string" },
          "ts": { "type": "integer" }
        },
        "additionalProperties": true
      },
      "RecommendLiveRequest": {
        "type": "object",
        "properties": {
          "draft_url": { "type": "string" },
          "league_id": { "type": "string" },
          "roster_id": { "type": "integer" },
          "team_slot": { "type": "integer" },
          "team_name": { "type": "string" },
          "pick_number": { "type": ["integer", "null"], "description": "Overall pick number (optional)" },
          "season": { "type": "integer", "default": 2025 },
          "roster_slots": {
            "type": "object",
            "description": "Optional override caps, e.g. {\"QB\":1,\"RB\":2,\"WR\":2,\"TE\":1,\"FLEX\":2}",
            "additionalProperties": { "type": "integer" },
            "default": { "QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 2 }
          },
          "limit": { "type": "integer", "default": 10 }
        },
        "oneOf": [ { "required": ["draft_url"] }, { "required": ["league_id"] } ],
        "anyOf": [ { "required": ["roster_id"] }, { "required": ["team_slot"] }, { "required": ["team_name"] } ],
        "additionalProperties": false
      },
      "RecommendLiveResponse": {
        "type": "object",
        "properties": {
          "status": { "type": "string" },
          "pick": { "type": ["integer", "null"] },
          "season_used": { "type": "integer" },
          "recommended": { "type": "array", "items": { "type": "object", "additionalProperties": true } },
          "alternatives": { "type": "array", "items": { "type": "object", "additionalProperties": true } },
          "my_team": { "type": "array", "items": { "type": "object", "additionalProperties": true } },
          "draft_state": { "type": "object", "additionalProperties": true },
          "effective_roster_id": { "type": ["integer", "null"] },
          "drafted_count": { "type": "integer" },
          "my_team_count": { "type": "integer" },
          "ts": { "type": "integer" }
        },
        "additionalProperties": true
      }
    }
  }
}
