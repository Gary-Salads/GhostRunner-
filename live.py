# ── GhostRunner — live.py ─────────────────────────────────────
# Polls the MLB Stats API every 15 seconds during a live game,
# makes run probability predictions on every pitch, tracks
# confident predictions that were wrong (and right), and saves
# a full game log + per-player summary at the end.
# ─────────────────────────────────────────────────────────────

import requests
import time
import json
import os
import pickle
from datetime import datetime, date
from typing import Optional

import pandas as pd

import review

from train import (
    predict,
    BASE_OUT_STATES,
    THRESHOLD_HIGH,
    THRESHOLD_LOW,
    LEAGUE_AVG_XWOBA,
)

# ── Constants ─────────────────────────────────────────────────
MLB_API_BASE    = "https://statsapi.mlb.com/api/v1"
POLL_INTERVAL   = 15      # seconds between API polls
MODELS_DIR      = "models"
LOGS_DIR        = "logs"


# ══════════════════════════════════════════════════════════════
# 1. MODEL LOADING
# ══════════════════════════════════════════════════════════════

def load_models(season_str: str) -> tuple:
    """
    Loads the trained pipeline and both lookup tables from disk.
    season_str can be "2026", "2025_2026", etc — matches the
    filename suffix that save_all() produced in train.py.
    Returns (pipeline, batter_table, pitcher_table).
    Exits with a clear message if files are missing.
    """
    pipeline_path = f"{MODELS_DIR}/pipeline_{season_str}.pkl"
    batter_path   = f"{MODELS_DIR}/batter_xwoba_{season_str}.csv"
    pitcher_path  = f"{MODELS_DIR}/pitcher_xwoba_{season_str}.csv"

    missing = [p for p in [pipeline_path, batter_path, pitcher_path]
               if not os.path.exists(p)]

    if missing:
        print("\nMissing model files:")
        for m in missing:
            print(f"  {m}")
        print("\nRun train.py first to generate these files.")
        raise FileNotFoundError("Model files not found.")

    print(f"Loading model ({season_str})...")

    with open(pipeline_path, "rb") as f:
        pipeline = pickle.load(f)

    batter_table  = pd.read_csv(batter_path)
    pitcher_table = pd.read_csv(pitcher_path)

    # empty string markers get read back as NaN by pandas
    # fill them so Tier 1 players show no marker instead of "nan"
    batter_table["marker"]  = batter_table["marker"].fillna("")
    pitcher_table["marker"] = pitcher_table["marker"].fillna("")

    print(f"  Pipeline     : {pipeline_path}")
    print(f"  Batters      : {len(batter_table):,} players")
    print(f"  Pitchers     : {len(pitcher_table):,} players")

    return pipeline, batter_table, pitcher_table


# ══════════════════════════════════════════════════════════════
# 2. MLB STATS API — GAME DISCOVERY
# ══════════════════════════════════════════════════════════════

def get_todays_games() -> list:
    """
    Returns all games scheduled for today from the MLB Stats API.
    Each entry: {game_pk, home_team, away_team, status, time}
    hydrate=team pulls full team details including abbreviation.
    """
    today = date.today().strftime("%Y-%m-%d")
    url   = (f"{MLB_API_BASE}/schedule?sportId=1"
             f"&date={today}&hydrate=team")

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"Error fetching schedule: {e}")
        return []

    games = []
    for game_date in data.get("dates", []):
        for g in game_date.get("games", []):
            home = g["teams"]["home"]["team"]
            away = g["teams"]["away"]["team"]

            # abbreviation may be under "abbreviation" or "teamCode"
            # fallback: first 3 chars of team name uppercased
            home_abbr = (home.get("abbreviation")
                         or home.get("teamCode", "")
                         or home.get("name", "HOM")[:3].upper())
            away_abbr = (away.get("abbreviation")
                         or away.get("teamCode", "")
                         or away.get("name", "AWY")[:3].upper())

            games.append({
                "game_pk":   g["gamePk"],
                "home_team": home["name"],
                "away_team": away["name"],
                "home_abbr": home_abbr.replace(" ", ""),
                "away_abbr": away_abbr.replace(" ", ""),
                "status":    g["status"]["detailedState"],
                "time":      g.get("gameDate", ""),
            })
    return games


def select_game(games: list) -> Optional[dict]:
    """
    Presents today's games to the user and returns the selected one.
    """
    if not games:
        print("No games found for today.")
        return None

    print("\nToday's games:")
    for i, g in enumerate(games, 1):
        print(f"  [{i}] {g['away_team']} @ {g['home_team']}"
              f"  —  {g['status']}")

    print("  [0] Exit")

    while True:
        choice = input("\nSelect a game: ").strip()
        if choice == "0":
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(games):
            return games[int(choice) - 1]
        print("  Invalid selection.")


# ══════════════════════════════════════════════════════════════
# 3. MLB STATS API — LIVE GAME STATE
# ══════════════════════════════════════════════════════════════

def fetch_live_state(game_pk: int) -> Optional[dict]:
    """
    Fetches the current live game state from the MLB Stats API.
    Returns a simplified dict of everything we need for prediction.
    Returns None if the feed is unavailable.
    """
    url = (f"https://statsapi.mlb.com/api/v1.1/game/"
           f"{game_pk}/feed/live")

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  API error: {e}")
        return None

    game_data  = data.get("gameData", {})
    live_data  = data.get("liveData", {})
    play_data  = live_data.get("plays", {})
    linescore  = live_data.get("linescore", {})
    current    = play_data.get("currentPlay", {})
    matchup    = current.get("matchup", {})
    offense    = linescore.get("offense", {})

    status = game_data.get("status", {}).get("detailedState", "Unknown")

    # batter and pitcher — confirmed paths from API debug
    batter_id    = matchup.get("batter",  {}).get("id")
    pitcher_id   = matchup.get("pitcher", {}).get("id")
    batter_name  = matchup.get("batter",  {}).get("fullName", "Unknown")
    pitcher_name = matchup.get("pitcher", {}).get("fullName", "Unknown")

    # handedness
    batter_hand  = matchup.get("batSide",   {}).get("code", "R")
    pitcher_hand = matchup.get("pitchHand", {}).get("code", "R")

    # count — confirmed: balls/strikes/outs live on linescore directly
    # NOT inside currentPlay.count (that reflects end-of-play state)
    balls   = linescore.get("balls",   0)
    strikes = linescore.get("strikes", 0)
    outs    = linescore.get("outs",    0)

    # inning — confirmed paths from debug
    inning      = linescore.get("currentInning", 1)
    inning_half = linescore.get("inningHalf", "Top")

    # base runners — confirmed: offense has "first"/"second"/"third"
    # keys containing player dicts when occupied, absent when empty
    on_1b = "first"  in offense and offense["first"]  is not None
    on_2b = "second" in offense and offense["second"] is not None
    on_3b = "third"  in offense and offense["third"]  is not None

    runner_1b = offense.get("first",  {}).get("fullName") if on_1b else None
    runner_2b = offense.get("second", {}).get("fullName") if on_2b else None
    runner_3b = offense.get("third",  {}).get("fullName") if on_3b else None

    batting_team_name = offense.get("team", {}).get("name", "")

    # at-bat index — confirmed: lives directly on currentPlay
    at_bat_index = current.get("atBatIndex", -1)

    # composite key backup
    at_bat_key = f"{batter_id}_{inning}_{inning_half}_{outs}"

    # scores
    home_score = linescore.get("teams", {}).get("home", {}).get("runs", 0)
    away_score = linescore.get("teams", {}).get("away", {}).get("runs", 0)

    home_team = game_data.get("teams", {}).get("home", {}).get("name", "")
    away_team = game_data.get("teams", {}).get("away", {}).get("name", "")

    return {
        "status":           status,
        "game_pk":          game_pk,
        "home_team":        home_team,
        "away_team":        away_team,
        "home_score":       home_score,
        "away_score":       away_score,
        "inning":           inning,
        "inning_half":      inning_half,
        "outs":             outs,
        "batter_id":        batter_id,
        "batter_name":      batter_name,
        "batter_hand":      batter_hand,
        "pitcher_id":       pitcher_id,
        "pitcher_name":     pitcher_name,
        "pitcher_hand":     pitcher_hand,
        "balls":            balls,
        "strikes":          strikes,
        "on_1b":            on_1b,
        "on_2b":            on_2b,
        "on_3b":            on_3b,
        "runner_1b":        runner_1b,
        "runner_2b":        runner_2b,
        "runner_3b":        runner_3b,
        "batting_team":     batting_team_name,
        "batting_team_abbr": offense.get("team", {}).get("abbreviation", ""),
        "at_bat_index":     at_bat_index,
        "at_bat_key":       at_bat_key,
        "pitch_count":      len(current.get("pitchIndex", [])),
    }


# ══════════════════════════════════════════════════════════════
# 4. DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════

def format_runners(state: dict) -> str:
    """Returns human-readable base runner string."""
    occupied = []
    if state["on_1b"]: occupied.append("1st")
    if state["on_2b"]: occupied.append("2nd")
    if state["on_3b"]: occupied.append("3rd")
    return ", ".join(occupied) if occupied else "bases empty"


def format_confidence(confidence: str, prob: float) -> str:
    if confidence == "confident_score":
        return f"▲ CONFIDENT SCORE  ({prob:.0%})"
    elif confidence == "confident_no_score":
        return f"▼ CONFIDENT NO SCORE  ({prob:.0%})"
    else:
        return f"~ uncertain  ({prob:.0%})"


def print_pitch_update(state: dict, result: dict, pitch_num: int):
    """Prints a formatted live prediction to the terminal."""
    runners  = format_runners(state)
    conf     = format_confidence(result["confidence"], result["probability"])
    b_marker = result["batter_marker"]
    p_marker = result["pitcher_marker"]

    # pitch label — within at-bat pitch count
    pitch_in_ab = state.get("pitch_count", 0)
    pitch_label = (f"Pitch {pitch_in_ab} of AB"
                   if pitch_in_ab > 0 else "At-bat start")

    print(f"\n{'─'*58}")
    print(f"  {state['inning_half']} {state['inning']}  "
          f"|  {state['outs']} out(s)  "
          f"|  {state['balls']}-{state['strikes']} count"
          f"  [{pitch_label}]")
    print(f"  Runners: {runners}")
    print(f"  Batter : {state['batter_name']}{b_marker}"
          f"  xwOBA {result['batter_xwoba']:.3f}{b_marker}")
    print(f"  Pitcher: {state['pitcher_name']}{p_marker}"
          f"  xwOBA-allowed {result['pitcher_xwoba']:.3f}{p_marker}")

    if result["batter_note"]:
        print(f"  ⚠ Batter  : {result['batter_note']}")
    if result["pitcher_note"]:
        print(f"  ⚠ Pitcher : {result['pitcher_note']}")

    print(f"\n  Run probability: {conf}")
    print(f"  Data quality   : {result['data_quality']}")

    if result["data_quality"] in ("low", "blind"):
        print(f"  ⚠ Low confidence inputs — treat prediction with caution")


# ══════════════════════════════════════════════════════════════
# 5. PITCH LOG & CONFIDENCE TRACKING
# ══════════════════════════════════════════════════════════════

def make_pitch_entry(state: dict, result: dict) -> dict:
    """
    Creates a log entry for one pitch.
    actual_outcome and failure_type are filled in at inning end.
    """
    return {
        # situation
        "game_pk":          state["game_pk"],
        "inning":           state["inning"],
        "inning_half":      state["inning_half"],
        "at_bat_index":     state["at_bat_index"],
        "outs":             state["outs"],
        "balls":            state["balls"],
        "strikes":          state["strikes"],
        "runners":          format_runners(state),
        "on_1b":            state["on_1b"],
        "on_2b":            state["on_2b"],
        "on_3b":            state["on_3b"],
        "batting_team":     state.get("batting_team", ""),
        "batting_team_abbr":state.get("batting_team_abbr", ""),
        # players
        "batter_id":      state["batter_id"],
        "batter_name":    state["batter_name"],
        "batter_xwoba":   result["batter_xwoba"],
        "batter_marker":  result["batter_marker"],
        "batter_tier":    result["batter_tier"],
        "pitcher_id":     state["pitcher_id"],
        "pitcher_name":   state["pitcher_name"],
        "pitcher_xwoba":  result["pitcher_xwoba"],
        "pitcher_marker": result["pitcher_marker"],
        "pitcher_tier":   result["pitcher_tier"],
        # prediction
        "probability":    result["probability"],
        "confidence":     result["confidence"],
        "data_quality":   result["data_quality"],
        # score at the moment of this pitch — used for forward-looking resolution
        "bat_score_at_pitch": (
            state["home_score"]
            if state["inning_half"] == "Bottom"
            else state["away_score"]
        ),
        # resolved after inning ends
        "actual_outcome": None,
        "failure_type":   None,
    }


def resolve_inning(inning_log: list,
                   final_bat_score: int) -> list:
    """
    At the end of each half-inning, resolves actual_outcome
    and failure_type for every logged pitch.

    Forward-looking: each pitch is only marked actual_outcome=1
    if at least one run scored AFTER that specific pitch in the
    inning — not if a run scored before it or in a prior at-bat.

    final_bat_score: batting team score at end of half-inning.
    Each entry must have bat_score_at_pitch set at log time.
    """
    for entry in inning_log:
        score_at_pitch = entry.get("bat_score_at_pitch", 0)

        # runs that scored AFTER this pitch in the same inning
        runs_after = max(0, final_bat_score - score_at_pitch)
        actual     = 1 if runs_after > 0 else 0

        entry["actual_outcome"]  = actual
        entry["runs_after_pitch"] = runs_after

        conf = entry["confidence"]

        if conf == "confident_score":
            entry["failure_type"] = (
                "correct_confident_score"
                if actual == 1
                else "false_confidence_score"
            )
        elif conf == "confident_no_score":
            entry["failure_type"] = (
                "correct_confident_no_score"
                if actual == 0
                else "false_confidence_no_score"
            )
        else:
            entry["failure_type"] = "uncertain"

    return inning_log


# ══════════════════════════════════════════════════════════════
# 6. END-OF-GAME SUMMARY
# ══════════════════════════════════════════════════════════════

def build_player_summary(game_log: list,
                          game_info: dict = None) -> dict:
    """
    Rolls up the full game log into per-batter and per-pitcher
    summary dicts split by team.

    Batters are split by batting_team (who was up).
    Pitchers are split by which team was pitching — derived
    from game_info home_team/away_team names.

    Structure:
      summary["batters"][team_name][batter_id] = {...}
      summary["pitchers"][team_name][pitcher_id] = {...}
    """
    if game_info is None:
        game_info = {}
    home_team = game_info.get("home_team", "Home")
    away_team = game_info.get("away_team", "Away")

    batters  = {}   # {team_name: {player_id: stats}}
    pitchers = {}

    for entry in game_log:
        ft = entry["failure_type"]
        if ft is None or ft == "uncertain":
            continue

        # ── batter team from batting_team field ────────────────
        batter_team = entry.get("batting_team", "Unknown")
        if batter_team not in batters:
            batters[batter_team] = {}

        bid = entry["batter_id"]
        if bid not in batters[batter_team]:
            batters[batter_team][bid] = {
                "name":   entry["batter_name"],
                "xwoba":  entry["batter_xwoba"],
                "marker": entry.get("batter_marker", ""),
                "tier":   entry["batter_tier"],
                "correct_confident_score":    0,
                "correct_confident_no_score": 0,
                "false_confidence_score":     0,
                "false_confidence_no_score":  0,
            }
        batters[batter_team][bid][ft] += 1

        # ── pitcher team — opposite of batting team ────────────
        # Top inning    = away team batting → home team pitching
        # Bottom inning = home team batting → away team pitching
        inning_half  = entry.get("inning_half", "Top")
        pitcher_team = (
            home_team if inning_half == "Top"
            else away_team
        )

        if pitcher_team not in pitchers:
            pitchers[pitcher_team] = {}

        pid = entry["pitcher_id"]
        if pid not in pitchers[pitcher_team]:
            pitchers[pitcher_team][pid] = {
                "name":   entry["pitcher_name"],
                "xwoba":  entry["pitcher_xwoba"],
                "marker": entry.get("pitcher_marker", ""),
                "tier":   entry["pitcher_tier"],
                "correct_confident_score":    0,
                "correct_confident_no_score": 0,
                "false_confidence_score":     0,
                "false_confidence_no_score":  0,
            }
        pitchers[pitcher_team][pid][ft] += 1

    return {"batters": batters, "pitchers": pitchers}


def print_game_summary(game_info: dict,
                       game_log: list,
                       summary: dict):
    """
    Prints the end-of-game report to the terminal.
    """
    total     = len(game9_log)
    confident = [e for e in game_log
                 if e["confidence"] != "uncertain"]
    false_score    = [e for e in game_log
                      if e["failure_type"] == "false_confidence_score"]
    false_no_score = [e for e in game_log
                      if e["failure_type"] == "false_confidence_no_score"]

    print(f"\n{'═'*58}")
    print(f"  GAME SUMMARY")
    print(f"  {game_info['away_team']} @ {game_info['home_team']}")
    print(f"  Final: {game_info['away_score']} – {game_info['home_score']}")
    print(f"{'═'*58}")

    print(f"\n  Total pitches tracked : {total:,}")
    print(f"  Confident predictions : {len(confident)}")
    print(f"  False confidence — said run, none scored  : "
          f"{len(false_score)}")
    print(f"  False confidence — said no run, run scored: "
          f"{len(false_no_score)}")

    if confident:
        correct = len([e for e in confident
                       if e["failure_type"] in
                       ("correct_confident_score",
                        "correct_confident_no_score")])
        acc = correct / len(confident)
        print(f"  Confident accuracy    : {acc:.1%} "
              f"({correct}/{len(confident)})")

    col_header = (f"  {'Player':<26} {'xwOBA':>6}  "
                  f"{'✓Run':>5} {'✗Run':>5} "
                  f"{'✓None':>6} {'✗None':>6}")

    def print_player_block(player_dict: dict):
        for pid, p in sorted(player_dict.items(),
                              key=lambda x: x[1]["xwoba"],
                              reverse=True):
            marker    = p.get("marker", "")
            name_disp = f"{p['name']}{marker}"
            xw_disp   = f"{p['xwoba']:.3f}{marker}"
            cs  = p.get("correct_confident_score",    0)
            fs  = p.get("false_confidence_score",     0)
            cn  = p.get("correct_confident_no_score", 0)
            fn  = p.get("false_confidence_no_score",  0)
            flag = ("← overconfident"  if fs >= 2
                    else "← underestimated" if fn >= 2
                    else "")
            print(f"  {name_disp:<26} {xw_disp:>7}  "
                  f"{cs:>5} {fs:>5} {cn:>6} {fn:>6}  {flag}")

    # ── per-batter tables split by team ────────────────────────
    for team_name, team_batters in sorted(summary["batters"].items()):
        print(f"\n{'─'*58}")
        print(f"  BATTERS — {team_name}")
        print(f"{'─'*58}")
        print(col_header)
        print_player_block(team_batters)

    # ── per-pitcher tables split by team ───────────────────────
    for team_name, team_pitchers in sorted(summary["pitchers"].items()):
        print(f"\n{'─'*58}")
        print(f"  PITCHERS — {team_name}")
        print(f"{'─'*58}")
        print(col_header)
        print_player_block(team_pitchers)

    # ── data quality breakdown ─────────────────────────────────
    by_quality = {}
    for e in false_score + false_no_score:
        q = e["data_quality"]
        by_quality[q] = by_quality.get(q, 0) + 1

    if by_quality:
        print(f"\n{'─'*58}")
        print(f"  WRONG PREDICTIONS BY DATA QUALITY")
        print(f"{'─'*58}")
        for quality in ["high", "medium", "low", "blind"]:
            count = by_quality.get(quality, 0)
            if count > 0:
                note = ("← genuine model failures" if quality == "high"
                        else "← treat with caution"  if quality == "medium"
                        else "← mostly noise"         if quality == "low"
                        else "← disregard")
                print(f"  {quality:<8}: {count:>3}  {note}")

    # marker key
    print(f"\n{'─'*58}")
    print(f"  DATA QUALITY KEY")
    print(f"  (no marker) Full data — both splits reliable")
    print(f"  *           Partial — split sparse, regressed to avg")
    print(f"  **          Very sparse — estimate unreliable")
    print(f"  ***         Unknown — not in training data, using .320")
    print(f"{'═'*58}\n")


# ══════════════════════════════════════════════════════════════
# 7. GAME LOG SAVE
# ══════════════════════════════════════════════════════════════

def save_game_log(game_info: dict,
                  game_log: list,
                  summary: dict):
    """
    Saves the full game log and summary to a JSON file in logs/.
    Filename: logs/YYYY-MM-DD_AWAY_vs_HOME.json
    """
    os.makedirs(LOGS_DIR, exist_ok=True)

    today     = date.today().strftime("%Y-%m-%d")
    away_abbr = game_info.get("away_abbr", "AWAY")
    home_abbr = game_info.get("home_abbr", "HOME")
    filename  = f"{LOGS_DIR}/{today}_{away_abbr}_vs_{home_abbr}.json"

    output = {
        "game_info":  game_info,
        "date":       today,
        "summary":    summary,
        "pitch_log":  game_log,
    }

    with open(filename, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"  Game log saved: {filename}")
    return filename


# ══════════════════════════════════════════════════════════════
# 8. MAIN POLLING LOOP
# ══════════════════════════════════════════════════════════════

def run_live(pipeline, batter_table, pitcher_table, game: dict):
    """
    Main polling loop. Runs until the game ends or the user
    presses Ctrl+C.

    Tracks:
      - last_at_bat_index: detects new at-bats
      - last_inning / last_inning_half: detects half-inning changes
      - current_inning_log: pitches in the current half-inning
      - last_score: detects run scoring for inning resolution
      - game_log: all resolved pitch entries for the full game
    """
    game_pk = game["game_pk"]

    print(f"\n{'═'*58}")
    print(f"  GhostRunner — LIVE")
    print(f"  {game['away_team']} @ {game['home_team']}")
    print(f"  Polling every {POLL_INTERVAL}s | Ctrl+C to stop")
    print(f"{'═'*58}")

    last_at_bat_index  = -1
    last_at_bat_key    = ""
    last_batter_id     = None   # primary change detector
    last_pitcher_id    = None   # catches pitcher changes mid-inning
    last_outs          = -1     # catches outs changes within inning
    last_pitch_count   = -1     # catches each new pitch within at-bat
    last_inning        = -1
    last_inning_half   = ""
    last_score         = 0      # batting team's score, tracks run scoring
    current_inning_log = []
    game_log           = []
    game_info          = {
        "game_pk":    game_pk,
        "home_team":  game["home_team"],
        "away_team":  game["away_team"],
        "home_abbr":  game.get("home_abbr", ""),
        "away_abbr":  game.get("away_abbr", ""),
        "home_score": 0,
        "away_score": 0,
    }

    try:
        while True:
            state = fetch_live_state(game_pk)

            if state is None:
                print("  Could not reach API — retrying...")
                time.sleep(POLL_INTERVAL)
                continue

            status = state["status"]

            # ── game not started yet ───────────────────────────
            if status in ("Preview", "Warmup", "Pre-Game",
                          "Scheduled"):
                print(f"  Waiting for first pitch... ({status})")
                time.sleep(POLL_INTERVAL)
                continue

            # ── game is over ───────────────────────────────────
            # catch all MLB API end-state strings including
            # doubleheader games which use "Final" + game number
            game_over = any(s in status for s in
                           ("Final", "Game Over", "Completed",
                            "Postponed", "Suspended"))

            if game_over:
                print(f"\n  Game status: {status}")

                # make one final API call to get the definitive
                # final score — the cached state may be one poll behind
                final_state = fetch_live_state(game_pk)
                if final_state:
                    game_info["home_score"] = final_state["home_score"]
                    game_info["away_score"] = final_state["away_score"]
                    state = final_state

                # resolve any remaining inning log
                if current_inning_log:
                    final_batting_score = (
                        state["home_score"]
                        if state["inning_half"] == "Bottom"
                        else state["away_score"]
                    )
                    current_inning_log = resolve_inning(
                        current_inning_log, final_batting_score
                    )
                    game_log.extend(current_inning_log)

                break

            # ── update game info scores ────────────────────────
            game_info["home_score"] = state["home_score"]
            game_info["away_score"] = state["away_score"]



            # ── detect half-inning change ──────────────────────
            inning_changed = (
                state["inning"]      != last_inning or
                state["inning_half"] != last_inning_half
            )

            if inning_changed and current_inning_log:
                # resolve the inning that just ended
                # pass final batting score so resolve_inning can
                # compute runs_after_pitch for each logged pitch
                current_inning_log = resolve_inning(
                    current_inning_log, last_score_end
                )
                game_log.extend(current_inning_log)
                current_inning_log = []

                print(f"\n  ── Inning change → "
                      f"{state['inning_half']} {state['inning']} ──")

            if inning_changed:
                # reset score tracker for new half-inning
                batting_score = (
                    state["home_score"]
                    if state["inning_half"] == "Bottom"
                    else state["away_score"]
                )
                last_score     = batting_score
                last_score_end = batting_score
                last_inning      = state["inning"]
                last_inning_half = state["inning_half"]

            # track current batting team score for run detection
            batting_score_now = (
                state["home_score"]
                if state["inning_half"] == "Bottom"
                else state["away_score"]
            )
            last_score_end = batting_score_now

            # ── detect new pitch or new at-bat ────────────────
            # fires on every new pitch thrown within an at-bat
            # as well as every new at-bat and situation change
            new_at_bat = (
                last_batter_id  is None                    or
                state["batter_id"]   != last_batter_id     or
                state["pitcher_id"]  != last_pitcher_id    or
                state["outs"]        != last_outs           or
                state["at_bat_index"]!= last_at_bat_index  or
                state["pitch_count"] != last_pitch_count
            )

            if new_at_bat:
                if (state["batter_id"] is None or
                        state["pitcher_id"] is None):
                    print("  ⚠ batter_id or pitcher_id is None — skipping")
                    time.sleep(POLL_INTERVAL)
                    continue

                try:
                    # run prediction
                    result = predict(
                        pipeline      = pipeline,
                        batter_table  = batter_table,
                        pitcher_table = pitcher_table,
                        batter_id     = state["batter_id"],
                        pitcher_id    = state["pitcher_id"],
                        batter_hand   = state["batter_hand"],
                        pitcher_hand  = state["pitcher_hand"],
                        balls         = state["balls"],
                        strikes       = state["strikes"],
                        outs          = state["outs"],
                        on_1b         = state["on_1b"],
                        on_2b         = state["on_2b"],
                        on_3b         = state["on_3b"],
                    )

                    # print live update
                    print_pitch_update(state, result,
                                       len(current_inning_log) + 1)

                    # log the pitch
                    entry = make_pitch_entry(state, result)
                    current_inning_log.append(entry)

                    last_at_bat_index = state["at_bat_index"]
                    last_at_bat_key   = state["at_bat_key"]
                    last_batter_id    = state["batter_id"]
                    last_pitcher_id   = state["pitcher_id"]
                    last_outs         = state["outs"]
                    last_pitch_count  = state["pitch_count"]

                except Exception as e:
                    import traceback
                    print(f"✗ Prediction error: {e}")
                    traceback.print_exc()
                    # still update trackers so we dont loop on same error
                    last_batter_id   = state["batter_id"]
                    last_pitcher_id  = state["pitcher_id"]
                    last_outs        = state["outs"]
                    last_pitch_count = state["pitch_count"]

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print(f"\n\n  Stopped by user.")
        save = input("  Save partial game log? (y/n): ").strip().lower()
        if save == "y" and game_log:
            summary = build_player_summary(game_log, game_info)
            save_game_log(game_info, game_log, summary)
            print_game_summary(game_info, game_log, summary)
        return

    # ── end of game ────────────────────────────────────────────
    summary = build_player_summary(game_log, game_info)
    print_game_summary(game_info, game_log, summary)
    log_file = save_game_log(game_info, game_log, summary)

    print(f"\n  Review this game with: python review.py")
    print(f"  Log file: {log_file}")
    try:
        from review import plot_game
        print(f"\n  Generating charts...")
        plot_game(log_file)
    except ImportError as e:
        print(f"  Could not load review_viz: {e}")
        print(f"  Run manually: python review_viz.py {log_file}")


# ══════════════════════════════════════════════════════════════
# 9. ENTRY POINT
# ══════════════════════════════════════════════════════════════

def start_live(season: int = None):
    """
    Called by main.py or run directly.
    Loads the model, selects a game, and starts polling.
    Season can be a single year or None to use the combined model.
    """
    # default to combined 2025+2026 model
    if season is None:
        season_str = "2025_2026"
    else:
        season_str = str(season)

    try:
        pipeline, batter_table, pitcher_table = load_models(season_str)
    except FileNotFoundError:
        return

    games = get_todays_games()

    if not games:
        print("No games scheduled today.")
        return

    game = select_game(games)
    if game is None:
        return

    # warn if game is already final
    if game["status"] in ("Final", "Game Over"):
        print(f"\n  Today's game is already final.")
        print(f"  Run review.py to see the log if one was saved.")
        return

    run_live(pipeline, batter_table, pitcher_table, game)


if __name__ == "__main__":
    start_live()