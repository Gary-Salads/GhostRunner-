# ── GhostRunner — train.py ────────────────────────────────────
# Pulls full-league Statcast data, builds regressed split xwOBA
# lookup tables for every batter and pitcher, engineers features
# at the pitch level, trains a logistic regression model, and
# saves everything to disk for use by live.py and review.py
# ─────────────────────────────────────────────────────────────

from pybaseball import statcast
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from sklearn.calibration import calibration_curve
import pickle
import os

# ── Constants ─────────────────────────────────────────────────
LEAGUE_AVG_XWOBA   = 0.320   # fallback for unknown players
K_STABILIZATION    = 200     # PA needed for split to be fully trusted
MIN_PA_SPARSE      = 50      # below this = Tier 3 (sparse)
MIN_PA_OVERALL     = 1       # below this = Tier 4 (unknown)

# Confidence thresholds (from calibration analysis)
THRESHOLD_HIGH     = 0.60    # confident a run will score
THRESHOLD_LOW      = 0.29    # confident no run will score

# ── Base-out encoding ─────────────────────────────────────────
BASE_OUT_STATES = {
    (0,0,0,0): 0,  (1,0,0,0): 1,  (0,1,0,0): 2,  (0,0,1,0): 3,
    (1,1,0,0): 4,  (0,1,1,0): 5,  (1,0,1,0): 6,  (1,1,1,0): 7,
    (0,0,0,1): 8,  (1,0,0,1): 9,  (0,1,0,1):10,  (0,0,1,1):11,
    (1,1,0,1):12,  (0,1,1,1):13,  (1,0,1,1):14,  (1,1,1,1):15,
    (0,0,0,2):16,  (1,0,0,2):17,  (0,1,0,2):18,  (0,0,1,2):19,
    (1,1,0,2):20,  (0,1,1,2):21,  (1,0,1,2):22,  (1,1,1,2):23,
}

def encode_base_out(row) -> int:
    try:
        key = (
            int(pd.notna(row["on_1b"])),
            int(pd.notna(row["on_2b"])),
            int(pd.notna(row["on_3b"])),
            int(row["outs_when_up"])
        )
        return BASE_OUT_STATES[key]
    except (KeyError, ValueError, TypeError):
        return -1


# ── 1. Data ingestion — full league ───────────────────────────
def load_statcast_all(seasons: list) -> pd.DataFrame:
    """
    Pull full-league Statcast for every season in the list.
    No team filter — we need every batter and pitcher in MLB
    so our lookup tables generalize to any live matchup.
    """
    frames = []
    for yr in seasons:
        print(f"Pulling full-league Statcast {yr}...")
        df = statcast(start_dt=f"{yr}-04-01", end_dt=f"{yr}-10-05")
        print(f"  {yr}: {len(df):,} pitches")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    print(f"\nTotal pitches loaded: {len(combined):,}")
    return combined


# ── 2. Data quality tier assignment ───────────────────────────
def assign_tier(total_pa: int, pa_vs_R: int, pa_vs_L: int,
                K: int = K_STABILIZATION,
                min_sparse: int = MIN_PA_SPARSE) -> tuple:
    """
    Returns (tier, marker) based on how much data backs
    the player's xwOBA estimates.

    Tier 1 — ""    full data, both splits reliable
    Tier 2 — "*"   partial data, splits exist but sparse
    Tier 3 — "**"  very sparse, overall xwOBA unreliable
    Tier 4 — "***" unknown, not in training data at all
    """
    if total_pa < min_sparse:
        return 3, "**"
    if pa_vs_R < K or pa_vs_L < K:
        return 2, "*"
    return 1, ""


# ── 3. Regressed split xwOBA lookup tables ────────────────────
def compute_regressed_splits(df: pd.DataFrame,
                              player_type: str,
                              K: int = K_STABILIZATION) -> pd.DataFrame:
    """
    Builds a lookup table of regression-adjusted split xwOBA
    for every batter or pitcher in the dataset.

    player_type = "batter"  → splits by pitcher throwing hand (p_throws)
    player_type = "pitcher" → splits by batter stance (stand)

    Regression formula per split:
      adjusted = (split_pa / (split_pa + K)) * split_xwoba
               + (K       / (split_pa + K)) * overall_xwoba

    As split_pa grows, adjusted leans toward the real split.
    As split_pa shrinks, adjusted leans toward the overall average.
    """
    assert player_type in ("batter", "pitcher"), \
        "player_type must be 'batter' or 'pitcher'"

    id_col   = player_type                    # "batter" or "pitcher"
    hand_col = "p_throws" if player_type == "batter" else "stand"
    val_col  = "estimated_woba_using_speedangle"

    # filter to rows with a valid xwOBA value
    df_valid = df[df[val_col].notna()].copy()

    print(f"\nBuilding {player_type} lookup table...")
    print(f"  Valid {val_col} rows: {len(df_valid):,}")

    # ── Step 1: overall xwOBA per player ──────────────────────
    overall = (
        df_valid
        .groupby(id_col)[val_col]
        .agg(overall_xwoba="mean", total_pa="count")
        .reset_index()
    )

    # ── Step 2: raw split xwOBA per player per hand ───────────
    splits = (
        df_valid
        .groupby([id_col, hand_col])[val_col]
        .agg(split_xwoba="mean", split_pa="count")
        .reset_index()
    )

    # pivot so each player has one row with R and L columns
    splits_pivot = splits.pivot(
        index=id_col,
        columns=hand_col,
        values=["split_xwoba", "split_pa"]
    )

    # flatten multi-level column names
    splits_pivot.columns = [
        f"{stat}_vs_{hand}"
        for stat, hand in splits_pivot.columns
    ]
    splits_pivot = splits_pivot.reset_index()

    # ensure R and L columns always exist even if one is missing
    for col in ["split_xwoba_vs_R", "split_pa_vs_R",
                "split_xwoba_vs_L", "split_pa_vs_L"]:
        if col not in splits_pivot.columns:
            splits_pivot[col] = np.nan

    # ── Step 3: join overall + splits ─────────────────────────
    table = overall.merge(splits_pivot, on=id_col, how="left")

    # fill missing splits with 0 PA and overall xwOBA as placeholder
    table["split_pa_vs_R"]    = table["split_pa_vs_R"].fillna(0).astype(int)
    table["split_pa_vs_L"]    = table["split_pa_vs_L"].fillna(0).astype(int)
    table["split_xwoba_vs_R"] = table["split_xwoba_vs_R"].fillna(
        table["overall_xwoba"]
    )
    table["split_xwoba_vs_L"] = table["split_xwoba_vs_L"].fillna(
        table["overall_xwoba"]
    )

    # ── Step 4: apply regression to the mean formula ──────────
    for hand in ["R", "L"]:
        pa_col  = f"split_pa_vs_{hand}"
        raw_col = f"split_xwoba_vs_{hand}"
        adj_col = f"adj_xwoba_vs_{hand}"

        weight_split   = table[pa_col] / (table[pa_col] + K)
        weight_overall = K             / (table[pa_col] + K)

        table[adj_col] = (
            weight_split   * table[raw_col] +
            weight_overall * table["overall_xwoba"]
        ).round(4)

    # ── Step 5: assign data quality tiers ─────────────────────
    tier_results = table.apply(
        lambda row: assign_tier(
            int(row["total_pa"]),
            int(row["split_pa_vs_R"]),
            int(row["split_pa_vs_L"])
        ),
        axis=1
    )
    table["tier"]   = tier_results.apply(lambda x: x[0])
    table["marker"] = tier_results.apply(lambda x: x[1])

    # ── Step 6: clean up — keep only what we need ─────────────
    keep_cols = [
        id_col,
        "overall_xwoba",
        "adj_xwoba_vs_R",
        "adj_xwoba_vs_L",
        "split_pa_vs_R",
        "split_pa_vs_L",
        "total_pa",
        "tier",
        "marker"
    ]
    table = table[keep_cols].copy()

    # ── Summary ───────────────────────────────────────────────
    tier_counts = table["tier"].value_counts().sort_index()
    print(f"  Total {player_type}s: {len(table):,}")
    print(f"  Tier 1 (full data)   : {tier_counts.get(1, 0):,}")
    print(f"  Tier 2 (partial)  *  : {tier_counts.get(2, 0):,}")
    print(f"  Tier 3 (sparse)   ** : {tier_counts.get(3, 0):,}")

    return table


# ── 4. Lookup functions ───────────────────────────────────────
def lookup_xwoba(player_id: int,
                 opposing_hand: str,
                 table: pd.DataFrame,
                 id_col: str) -> dict:
    """
    Returns xwOBA value, marker, tier, and a human-readable note
    for a given player against a specific opposing hand.

    opposing_hand: "R" or "L"
    id_col:        "batter" or "pitcher"
    """
    row = table[table[id_col] == player_id]

    # Tier 4 — completely unknown player
    if row.empty:
        return {
            "xwoba":  LEAGUE_AVG_XWOBA,
            "marker": "***",
            "tier":   4,
            "note":   "not in training data — using league avg .320"
        }

    row = row.iloc[0]

    # select the right split based on opposing hand
    if opposing_hand == "R":
        xwoba_val = row["adj_xwoba_vs_R"]
        split_pa  = int(row["split_pa_vs_R"])
    elif opposing_hand == "L":
        xwoba_val = row["adj_xwoba_vs_L"]
        split_pa  = int(row["split_pa_vs_L"])
    else:
        # hand unknown — fall back to overall
        xwoba_val = row["overall_xwoba"]
        split_pa  = int(row["total_pa"])

    marker = row["marker"]
    tier   = int(row["tier"])

    # build human-readable note
    if tier == 1:
        note = ""
    elif tier == 2:
        note = (f"split based on {split_pa} PA "
                f"— heavily regressed to season avg")
    elif tier == 3:
        note = (f"only {int(row['total_pa'])} total PA "
                f"— estimate unreliable")
    else:
        note = "not in training data — using league avg .320"

    return {
        "xwoba":  round(float(xwoba_val), 4),
        "marker": marker,
        "tier":   tier,
        "note":   note
    }


def compute_data_quality(batter_tier: int, pitcher_tier: int) -> str:
    """
    Combines batter and pitcher tier into one data quality label.
    Worst tier of the two dominates.
    """
    worst = max(batter_tier, pitcher_tier)
    return {1: "high", 2: "medium", 3: "low", 4: "blind"}[worst]


# ── 5. Feature engineering ────────────────────────────────────
def build_features(df: pd.DataFrame,
                   batter_table: pd.DataFrame,
                   pitcher_table: pd.DataFrame) -> pd.DataFrame:
    """
    Builds the feature matrix at the PITCH level.

    Key design: target label is computed at the AT-BAT level first
    (using the last pitch of each at-bat where post_bat_score is
    resolved), then joined back to every pitch in that at-bat.
    Computing the label directly at pitch level causes post_bat_score
    to equal bat_score on non-final pitches, giving a 0% run rate.

    Features per pitch:
      - batter's regressed split xwOBA vs this pitcher's hand
      - pitcher's regressed split xwOBA-allowed vs this batter's hand
      - balls, strikes, outs
      - base-out state (one-hot encoded later in pipeline)
      - target: did any run score rest of this half-inning
    """
    print("\nBuilding feature matrix...")

    needed_cols = [
        "game_pk", "inning", "inning_topbot",
        "at_bat_number", "pitch_number",
        "batter", "pitcher",
        "stand",       # batter handedness
        "p_throws",    # pitcher handedness
        "balls", "strikes", "outs_when_up",
        "on_1b", "on_2b", "on_3b",
        "bat_score", "post_bat_score",
        "estimated_woba_using_speedangle"
    ]
    df_clean = df[needed_cols].dropna(
        subset=["batter", "pitcher", "stand", "p_throws",
                "balls", "strikes", "outs_when_up"]
    ).copy()

    print(f"  Pitches after cleaning: {len(df_clean):,}")

    # ── Base-out state (from raw pitch rows) ──────────────────
    df_clean["base_out_state"] = df_clean.apply(encode_base_out, axis=1)
    df_clean = df_clean[df_clean["base_out_state"] >= 0]

    # ── Step 1: target label at AT-BAT level ──────────────────
    # Use the LAST pitch of each at-bat — only here is
    # post_bat_score fully resolved after the play
    print("  Computing target labels at at-bat level...")

    ab_level = (
        df_clean
        .sort_values(["game_pk", "at_bat_number", "pitch_number"])
        .groupby(["game_pk", "at_bat_number"], as_index=False)
        .last()
    )

    # forward-looking run label — use merge instead of groupby.apply
    # to avoid pandas index absorption across different versions

    # total runs scored in each half-inning
    inning_totals = (
        ab_level
        .groupby(["game_pk", "inning", "inning_topbot"])
        .agg(
            inning_end_score=("post_bat_score", "last"),
            inning_start_score=("bat_score", "first")
        )
        .reset_index()
    )
    inning_totals["inning_total_runs"] = (
        inning_totals["inning_end_score"] -
        inning_totals["inning_start_score"]
    )

    # join totals back onto ab_level
    ab_level = ab_level.merge(
        inning_totals[["game_pk", "inning", "inning_topbot",
                        "inning_total_runs", "inning_start_score"]],
        on=["game_pk", "inning", "inning_topbot"],
        how="left"
    )

    # runs scored before this at-bat started
    ab_level["cum_runs_before"] = (
        ab_level["bat_score"] - ab_level["inning_start_score"]
    )

    ab_level["run_scored"] = (
        (ab_level["inning_total_runs"] - ab_level["cum_runs_before"]) > 0
    ).astype(int)

    ab_labels = ab_level[["game_pk", "at_bat_number", "run_scored"]].copy()

    print(f"  At-bat level run rate: "
          f"{ab_labels['run_scored'].mean():.1%}")

    # ── Step 2: join label back to every pitch ─────────────────
    # Every pitch within an at-bat inherits the same resolved label
    df_clean = df_clean.merge(
        ab_labels,
        on=["game_pk", "at_bat_number"],
        how="left"
    )

    # ── Step 3: batter xwOBA lookup ───────────────────────────
    print("  Looking up batter split xwOBA...")
    batter_lookup = batter_table.set_index("batter")

    def get_batter_xwoba(row):
        pid  = row["batter"]
        hand = row["p_throws"]
        if pid in batter_lookup.index:
            r = batter_lookup.loc[pid]
            if hand == "R":   return r["adj_xwoba_vs_R"]
            elif hand == "L": return r["adj_xwoba_vs_L"]
            else:             return r["overall_xwoba"]
        return LEAGUE_AVG_XWOBA

    df_clean["batter_xwoba"] = df_clean.apply(get_batter_xwoba, axis=1)

    # ── Step 4: pitcher xwOBA-allowed lookup ──────────────────
    print("  Looking up pitcher split xwOBA-allowed...")
    pitcher_lookup = pitcher_table.set_index("pitcher")

    def get_pitcher_xwoba(row):
        pid  = row["pitcher"]
        hand = row["stand"]
        if pid in pitcher_lookup.index:
            r = pitcher_lookup.loc[pid]
            if hand == "R":   return r["adj_xwoba_vs_R"]
            elif hand == "L": return r["adj_xwoba_vs_L"]
            else:             return r["overall_xwoba"]
        return LEAGUE_AVG_XWOBA

    df_clean["pitcher_xwoba"] = df_clean.apply(get_pitcher_xwoba, axis=1)

    # ── Final feature columns ──────────────────────────────────
    feature_cols = [
        "base_out_state",
        "batter_xwoba",
        "pitcher_xwoba",
        "balls",
        "strikes",
        "outs_when_up",
    ]

    df_model = df_clean[
        feature_cols + ["run_scored", "batter", "pitcher",
                        "stand", "p_throws"]
    ].dropna()

    print(f"\nFeature matrix: {len(df_model):,} pitches")
    print(f"Run rate: {df_model['run_scored'].mean():.1%}")
    print(f"Unique batters:  {df_model['batter'].nunique():,}")
    print(f"Unique pitchers: {df_model['pitcher'].nunique():,}")

    return df_model


# ── 6. Pipeline ───────────────────────────────────────────────
def build_pipeline() -> Pipeline:
    """
    Features:
      base_out_state → one-hot encoded (24 binary columns)
      batter_xwoba   → passthrough float
      pitcher_xwoba  → passthrough float
      balls          → passthrough int (0-3)
      strikes        → passthrough int (0-2)
      outs_when_up   → passthrough int (0-2)
    """
    preprocessor = ColumnTransformer(transformers=[
        (
            "base_out",
            OneHotEncoder(
                categories=[list(range(24))],
                sparse_output=False,
                handle_unknown="ignore"
            ),
            ["base_out_state"]
        ),
        (
            "numeric",
            "passthrough",
            ["batter_xwoba", "pitcher_xwoba",
             "balls", "strikes", "outs_when_up"]
        ),
    ])

    return Pipeline([
        ("prep",  preprocessor),
        ("model", LogisticRegression(max_iter=1000, C=1.0)),
    ])


# ── 7. Train ──────────────────────────────────────────────────
def train(df: pd.DataFrame):
    feature_cols = [
        "base_out_state", "batter_xwoba", "pitcher_xwoba",
        "balls", "strikes", "outs_when_up"
    ]
    X = df[feature_cols]
    y = df["run_scored"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTraining on {len(X_train):,} pitches...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    print("  Done.")
    return pipeline, X_test, y_test


# ── 8. Evaluate ───────────────────────────────────────────────
def evaluate(pipeline, X_test, y_test):
    probs = pipeline.predict_proba(X_test)[:, 1]

    print(f"\nEvaluation:")
    print(f"  Log-loss : {log_loss(y_test, probs):.4f}")
    print(f"  Brier    : {brier_score_loss(y_test, probs):.4f}")
    print(f"  ROC-AUC  : {roc_auc_score(y_test, probs):.4f}")

    print(f"\nCalibration (predicted → actual):")
    frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10)
    for mp, fp in zip(mean_pred, frac_pos):
        bar = "█" * int(fp * 20)
        diff = fp - mp
        flag = " ← overconfident" if diff < -0.05 else \
               " ← underconfident" if diff > 0.05 else ""
        print(f"  predicted {mp:.2f} → actual {fp:.2f}  {bar}{flag}")


# ── 9. Coefficients ───────────────────────────────────────────
def print_coefficients(pipeline):
    model  = pipeline.named_steps["model"]
    coeffs = model.coef_[0]

    state_names = [
        "Empty 0 out",   "1st 0 out",     "2nd 0 out",     "3rd 0 out",
        "1st+2nd 0 out", "2nd+3rd 0 out", "1st+3rd 0 out", "Loaded 0 out",
        "Empty 1 out",   "1st 1 out",     "2nd 1 out",     "3rd 1 out",
        "1st+2nd 1 out", "2nd+3rd 1 out", "1st+3rd 1 out", "Loaded 1 out",
        "Empty 2 out",   "1st 2 out",     "2nd 2 out",     "3rd 2 out",
        "1st+2nd 2 out", "2nd+3rd 2 out", "1st+3rd 2 out", "Loaded 2 out",
    ]

    print("\nCoefficients — base-out states:")
    for name, coef in sorted(zip(state_names, coeffs[:24]),
                             key=lambda x: x[1], reverse=True):
        bar = "+" * int(abs(coef) * 5) if coef > 0 else \
              "-" * int(abs(coef) * 5)
        print(f"  {name:<20} {coef:+.3f}  {bar}")

    # numeric features start at index 24
    numeric_names = [
        "batter_xwoba", "pitcher_xwoba",
        "balls", "strikes", "outs_when_up"
    ]
    print()
    for i, name in enumerate(numeric_names):
        coef = coeffs[24 + i]
        direction = "↑ run more likely" if coef > 0 else "↓ run less likely"
        print(f"  {name:<22} {coef:+.3f}  {direction}")


# ── 10. Save ──────────────────────────────────────────────────
def save_all(pipeline, batter_table, pitcher_table, seasons):
    os.makedirs("models", exist_ok=True)
    season_str = "_".join(str(s) for s in seasons)

    pipeline_path = f"models/pipeline_{season_str}.pkl"
    batter_path   = f"models/batter_xwoba_{season_str}.csv"
    pitcher_path  = f"models/pitcher_xwoba_{season_str}.csv"

    with open(pipeline_path, "wb") as f:
        pickle.dump(pipeline, f)

    batter_table.to_csv(batter_path, index=False)
    pitcher_table.to_csv(pitcher_path, index=False)

    print(f"\nSaved:")
    print(f"  {pipeline_path}")
    print(f"  {batter_path}")
    print(f"  {pitcher_path}")

    return pipeline_path, batter_path, pitcher_path


# ── 11. Predict (single pitch) ────────────────────────────────
def predict(pipeline,
            batter_table: pd.DataFrame,
            pitcher_table: pd.DataFrame,
            batter_id: int,
            pitcher_id: int,
            batter_hand: str,
            pitcher_hand: str,
            balls: int,
            strikes: int,
            outs: int,
            on_1b: bool,
            on_2b: bool,
            on_3b: bool) -> dict:
    """
    Returns probability, marker info, and data quality
    for a single pitch situation.
    """
    # look up batter xwOBA vs pitcher's throwing hand
    batter_info  = lookup_xwoba(batter_id,  pitcher_hand,
                                batter_table,  "batter")

    # look up pitcher xwOBA-allowed vs batter's stance
    pitcher_info = lookup_xwoba(pitcher_id, batter_hand,
                                pitcher_table, "pitcher")

    # encode base-out state
    key = (int(on_1b), int(on_2b), int(on_3b), outs)
    base_out = BASE_OUT_STATES.get(key, 0)

    X = pd.DataFrame([{
        "base_out_state": base_out,
        "batter_xwoba":  batter_info["xwoba"],
        "pitcher_xwoba": pitcher_info["xwoba"],
        "balls":         balls,
        "strikes":       strikes,
        "outs_when_up":  outs,
    }])

    prob = round(float(pipeline.predict_proba(X)[0][1]), 4)

    # classify confidence
    if prob >= THRESHOLD_HIGH:
        confidence = "confident_score"
    elif prob <= THRESHOLD_LOW:
        confidence = "confident_no_score"
    else:
        confidence = "uncertain"

    data_quality = compute_data_quality(
        batter_info["tier"], pitcher_info["tier"]
    )

    return {
        "probability":   prob,
        "confidence":    confidence,
        "data_quality":  data_quality,
        "batter_xwoba":  batter_info["xwoba"],
        "batter_marker": batter_info["marker"],
        "batter_tier":   batter_info["tier"],
        "batter_note":   batter_info["note"],
        "pitcher_xwoba": pitcher_info["xwoba"],
        "pitcher_marker": pitcher_info["marker"],
        "pitcher_tier":  pitcher_info["tier"],
        "pitcher_note":  pitcher_info["note"],
    }


# ── Run ───────────────────────────────────────────────────────
if __name__ == "__main__":
    SEASONS = [2024, 2025, 2026]

    raw           = load_statcast_all(SEASONS)
    batter_table  = compute_regressed_splits(raw, "batter")
    pitcher_table = compute_regressed_splits(raw, "pitcher")
    df_model      = build_features(raw, batter_table, pitcher_table)

    pipeline, X_test, y_test = train(df_model)

    evaluate(pipeline, X_test, y_test)
    print_coefficients(pipeline)

    save_all(pipeline, batter_table, pitcher_table, SEASONS)