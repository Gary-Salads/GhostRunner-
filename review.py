# ── GhostRunner — review_viz.py ───────────────────────────────
# Matplotlib visualization layer for GhostRunner review.
# Each chart type is its own figure window for readability.
# ─────────────────────────────────────────────────────────────

import json
import os
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from typing import Optional

# ── Style ──────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "figure.facecolor":  "#1a1a2e",
    "axes.facecolor":    "#16213e",
    "axes.edgecolor":    "#0f3460",
    "axes.labelcolor":   "#e0e0e0",
    "axes.titlecolor":   "#e0e0e0",
    "xtick.color":       "#a0a0a0",
    "ytick.color":       "#a0a0a0",
    "text.color":        "#e0e0e0",
    "grid.color":        "#0f3460",
    "grid.linestyle":    "--",
    "grid.alpha":        0.4,
    "font.family":       "monospace",
    "figure.dpi":        100,
})

CLR_CORRECT_SCORE    = "#2ecc71"
CLR_FALSE_SCORE      = "#e74c3c"
CLR_CORRECT_NO_SCORE = "#3498db"
CLR_FALSE_NO_SCORE   = "#f39c12"
CLR_ACCENT           = "#4cc9f0"
CLR_PINK             = "#f72585"


# ══════════════════════════════════════════════════════════════
# 0. GAME SELECTION MENU
# ══════════════════════════════════════════════════════════════

def select_game_log(logs_dir: str = "logs") -> Optional[str]:
    """
    Presents a numbered menu of available game logs and returns
    the path of the selected one. Returns None if none found or
    user cancels.
    """
    if not os.path.exists(logs_dir):
        print(f"  logs/ directory not found.")
        return None

    files = sorted(
        [f for f in os.listdir(logs_dir) if f.endswith(".json")],
        reverse=True
    )

    if not files:
        print("  No game logs found. Run a live game first.")
        return None

    print("\n  Available game logs:")
    for i, fname in enumerate(files, 1):
        # pretty-print filename: 2026-07-11_COL_vs_SF → Jul 11 2026 COL vs SF
        label = fname.replace(".json", "").replace("_", " ")
        print(f"  [{i}] {label}")
    print("  [0] Cancel")

    while True:
        choice = input("\n  Select a game: ").strip()
        if choice == "0":
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(files):
            return os.path.join(logs_dir, files[int(choice) - 1])
        print("  Invalid selection — enter a number from the list.")


# ══════════════════════════════════════════════════════════════
# 1. TRAINING MODEL VISUALIZATIONS  (3 separate figures)
# ══════════════════════════════════════════════════════════════

def plot_training(pipeline_path: str,
                  calibration_data: Optional[dict] = None,
                  save_path: str = None):
    """
    Three separate figures for the training model:
      Fig 1 — Calibration curve
      Fig 2 — Base-out state coefficients (full height, readable)
      Fig 3 — Numeric feature coefficients
    """
    if not os.path.exists(pipeline_path):
        print(f"  Pipeline not found: {pipeline_path}")
        return

    with open(pipeline_path, "rb") as f:
        pipeline = pickle.load(f)

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
    state_coeffs = coeffs[:24]
    sorted_pairs  = sorted(zip(state_names, state_coeffs), key=lambda x: x[1])
    sorted_names  = [p[0] for p in sorted_pairs]
    sorted_coeffs = [p[1] for p in sorted_pairs]

    feat_names  = ["batter xwOBA", "pitcher xwOBA",
                   "balls", "strikes", "outs"]
    feat_coeffs = list(coeffs[24:29])

    # ── Fig 1: Calibration ─────────────────────────────────────
    fig1, ax = plt.subplots(figsize=(7, 6))
    fig1.suptitle("GhostRunner — Calibration Curve",
                  fontsize=13, fontweight="bold",
                  color=CLR_ACCENT, y=0.98)

    if calibration_data:
        mp = calibration_data["mean_pred"]
        fp = calibration_data["frac_pos"]
        ax.plot(mp, fp, "o-", color=CLR_ACCENT,
                linewidth=2.5, markersize=0, zorder=3,
                label="Model calibration")
        for x, y in zip(mp, fp):
            color = CLR_CORRECT_SCORE if abs(x-y) < 0.03 else \
                    CLR_FALSE_NO_SCORE if abs(x-y) < 0.07 else CLR_FALSE_SCORE
            ax.plot(x, y, "o", color=color, markersize=11, zorder=4)
            ax.annotate(f"{y:.2f}", (x, y),
                        textcoords="offset points",
                        xytext=(6, 3), fontsize=8,
                        color="#e0e0e0")
    else:
        ax.text(0.5, 0.5,
                "Pass calibration_data\nto see this chart",
                ha="center", va="center",
                color="#a0a0a0", fontsize=10,
                transform=ax.transAxes)

    ax.plot([0,1],[0,1], "--", color="#a0a0a0",
            linewidth=1.2, label="Perfect calibration", zorder=2)
    ax.fill_between([0,1],[0,1],[0,1], alpha=0.04, color="#a0a0a0")
    ax.set_xlabel("Predicted probability", fontsize=11)
    ax.set_ylabel("Actual run rate",       fontsize=11)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=9)
    ax.grid(True)

    _legend_patches(ax, [
        (CLR_CORRECT_SCORE,  "< 0.03 gap — excellent"),
        (CLR_FALSE_NO_SCORE, "0.03–0.07 gap — acceptable"),
        (CLR_FALSE_SCORE,    "> 0.07 gap — calibration issue"),
    ])

    fig1.subplots_adjust(left=0.10, right=0.95,
                         top=0.90, bottom=0.12)
    _save_or_show(fig1, save_path, "calibration")

    # ── Fig 2: Base-out state coefficients ─────────────────────
    fig2, ax2 = plt.subplots(figsize=(9, 10))
    fig2.suptitle("GhostRunner — Base-Out State Coefficients",
                  fontsize=13, fontweight="bold",
                  color=CLR_ACCENT, y=0.99)

    bar_colors = [CLR_FALSE_SCORE if c < 0 else CLR_CORRECT_SCORE
                  for c in sorted_coeffs]

    bars = ax2.barh(range(24), sorted_coeffs,
                    color=bar_colors, alpha=0.88,
                    edgecolor="#0f3460", linewidth=0.6,
                    height=0.65)

    ax2.set_yticks(range(24))
    ax2.set_yticklabels(sorted_names, fontsize=10)
    ax2.axvline(0, color="#a0a0a0", linewidth=1.0)
    ax2.set_xlabel("Log-odds contribution to run probability", fontsize=10)
    ax2.grid(True, axis="x", alpha=0.4)

    for bar, val in zip(bars, sorted_coeffs):
        pad  = 0.03 if val >= 0 else -0.03
        ha   = "left" if val >= 0 else "right"
        ax2.text(val + pad,
                 bar.get_y() + bar.get_height() / 2,
                 f"{val:+.3f}",
                 va="center", ha=ha,
                 fontsize=8.5, color="#e0e0e0",
                 fontweight="bold")

    _legend_patches(ax2, [
        (CLR_CORRECT_SCORE, "Increases run probability"),
        (CLR_FALSE_SCORE,   "Decreases run probability"),
    ], loc="lower right")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    _save_or_show(fig2, save_path, "base_out_coefficients")

    # ── Fig 3: Numeric feature coefficients ────────────────────
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    fig3.suptitle("GhostRunner — Feature Coefficients",
                  fontsize=13, fontweight="bold",
                  color=CLR_ACCENT, y=0.99)

    feat_colors = [CLR_CORRECT_SCORE if c > 0 else CLR_FALSE_SCORE
                   for c in feat_coeffs]

    bars3 = ax3.barh(range(len(feat_names)), feat_coeffs,
                     color=feat_colors, alpha=0.88,
                     edgecolor="#0f3460", linewidth=0.6,
                     height=0.55)

    ax3.set_yticks(range(len(feat_names)))
    ax3.set_yticklabels(feat_names, fontsize=12)
    ax3.axvline(0, color="#a0a0a0", linewidth=1.0)
    ax3.set_xlabel("Log-odds contribution", fontsize=11)
    ax3.grid(True, axis="x", alpha=0.4)

    for bar, val, name in zip(bars3, feat_coeffs, feat_names):
        pad = 0.06 if val >= 0 else -0.06
        ha  = "left" if val >= 0 else "right"
        ax3.text(val + pad,
                 bar.get_y() + bar.get_height() / 2,
                 f"{val:+.3f}",
                 va="center", ha=ha,
                 fontsize=10, color="#e0e0e0",
                 fontweight="bold")
        if name == "pitcher xwOBA":
            ax3.text(val + pad * 6,
                     bar.get_y() + bar.get_height() / 2,
                     "⚠ sign issue — see notes",
                     va="center", ha=ha,
                     fontsize=8, color=CLR_FALSE_NO_SCORE)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _save_or_show(fig3, save_path, "feature_coefficients")


# ══════════════════════════════════════════════════════════════
# 2. GAME LOG VISUALIZATIONS  (3 separate figures)
# ══════════════════════════════════════════════════════════════

def plot_game(log_path: str, save_path: str = None):
    """
    Four separate figures for a game log:
      Fig 1 — Probability timeline + inning calibration
      Fig 2 — Team performance: actual runs per half-inning
      Fig 3 — Wrong prediction gap distribution + inning breakdown
      Fig 4 — Player confidence bars (batters + pitchers)
    """
    if not os.path.exists(log_path):
        print(f"  Log not found: {log_path}")
        return

    with open(log_path) as f:
        log = json.load(f)

    _clean_nan_markers(log)

    game_info = log.get("game_info", {})
    pitch_log = log.get("pitch_log", [])
    summary   = log.get("summary", {})

    if not pitch_log:
        print("  No pitch data in log.")
        return

    away = game_info.get("away_team", "Away")
    home = game_info.get("home_team", "Home")
    a_sc = game_info.get("away_score", "?")
    h_sc = game_info.get("home_score", "?")
    title_base = f"{away} @ {home}  |  Final: {a_sc}–{h_sc}"

    # ── Fig 1: Timeline + Inning calibration ───────────────────
    fig1, (ax_time, ax_cal) = plt.subplots(
        2, 1, figsize=(16, 10),
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.40}
    )
    fig1.suptitle(f"GhostRunner — Probability Timeline\n{title_base}",
                  fontsize=13, fontweight="bold",
                  color=CLR_ACCENT, y=0.99)

    _plot_probability_timeline(ax_time, pitch_log, away, home)
    _plot_inning_calibration(ax_cal, pitch_log)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _save_or_show(fig1, save_path, "timeline")

    # ── Fig 2: Team performance — actual runs per half-inning ──
    fig2, ax_runs = plt.subplots(figsize=(13, 6.5))
    fig2.suptitle(f"GhostRunner — Team Performance vs Model Expectation\n{title_base}",
                  fontsize=13, fontweight="bold",
                  color=CLR_ACCENT, y=0.98)

    _plot_runs_per_half_inning(ax_runs, pitch_log, away, home)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    _save_or_show(fig2, save_path, "team_performance")

    # ── Fig 3: Wrong predictions + inning table ─────────────────
    fig3, (ax_wrong, ax_ing) = plt.subplots(
        1, 2, figsize=(14, 6)
    )
    fig3.suptitle(f"GhostRunner — Prediction Errors\n{title_base}",
                  fontsize=13, fontweight="bold",
                  color=CLR_ACCENT, y=1.01)

    _plot_wrong_distribution(ax_wrong, pitch_log)
    _plot_inning_table(ax_ing, pitch_log)

    fig3.subplots_adjust(left=0.08, right=0.97,
                         top=0.88, bottom=0.10,
                         wspace=0.32)
    _save_or_show(fig3, save_path, "errors")

    # ── Fig 4: Player bars ──────────────────────────────────────
    plot_player_bars(log, title_base, save_path)


def _plot_probability_timeline(ax, pitch_log: list,
                                away: str, home: str):
    """Full-width probability line colored by batting team."""
    ax.set_title("Run Probability — Pitch by Pitch",
                 fontsize=11, pad=8)
    ax.set_xlabel("Pitch number", fontsize=9)
    ax.set_ylabel("P(run scores)",  fontsize=9)
    ax.set_ylim(-0.08, 1.10)

    ax.axhline(0.65, color=CLR_CORRECT_SCORE, linewidth=0.9,
               linestyle="--", alpha=0.55, label="High confidence (0.65)")
    ax.axhline(0.25, color=CLR_FALSE_SCORE,   linewidth=0.9,
               linestyle="--", alpha=0.55, label="Low confidence (0.25)")

    top_x, top_y, bot_x, bot_y = [], [], [], []
    inning_firsts = {}
    last_half = None

    for i, e in enumerate(pitch_log):
        prob = e.get("probability", 0.5)
        half = e.get("inning_half", "Top")
        ing  = e.get("inning", 0)
        key  = (ing, half)

        if key not in inning_firsts:
            inning_firsts[key] = i

        if half != last_half and last_half is not None:
            ax.axvline(i, color="#2a2a4e",
                       linewidth=1.0, alpha=0.8)
        last_half = half

        if half == "Top":
            top_x.append(i); top_y.append(prob)
        else:
            bot_x.append(i); bot_y.append(prob)

    if top_x:
        ax.plot(top_x, top_y, color=CLR_ACCENT,
                linewidth=1.4, alpha=0.85,
                label=f"{away} batting")
    if bot_x:
        ax.plot(bot_x, bot_y, color=CLR_PINK,
                linewidth=1.4, alpha=0.85,
                label=f"{home} batting")

    # wrong prediction markers — only confident ones
    for i, e in enumerate(pitch_log):
        ft = e.get("failure_type", "")
        p  = e.get("probability", 0.5)
        if ft == "false_confidence_score":
            ax.scatter(i, p, color=CLR_FALSE_SCORE,
                       s=55, zorder=6, marker="v",
                       edgecolors="#1a1a2e", linewidth=0.5)
        elif ft == "false_confidence_no_score":
            ax.scatter(i, p, color=CLR_FALSE_NO_SCORE,
                       s=55, zorder=6, marker="^",
                       edgecolors="#1a1a2e", linewidth=0.5)

    # inning labels along top — spaced so they don't collide
    prev_x = -999
    for (ing, half), x in sorted(inning_firsts.items(),
                                   key=lambda kv: kv[1]):
        if x - prev_x > 5:
            label = f"{ing}{'T' if half=='Top' else 'B'}"
            ax.text(x, 1.04, label, fontsize=7,
                    color="#a0a0a0", ha="left", va="bottom")
            prev_x = x

    ax.legend(fontsize=8, loc="upper left",
              ncol=2, framealpha=0.3)
    ax.grid(True, alpha=0.3)

    # wrong prediction legend
    wrong_patches = [
        mpatches.Patch(color=CLR_FALSE_SCORE,
                       label="▼ Said run — no run scored"),
        mpatches.Patch(color=CLR_FALSE_NO_SCORE,
                       label="▲ Said no run — run scored"),
    ]
    ax.legend(handles=ax.get_legend_handles_labels()[0] + wrong_patches,
              labels=ax.get_legend_handles_labels()[1] + [p.get_label() for p in wrong_patches],
              fontsize=7.5, loc="upper right",
              ncol=2, framealpha=0.3)


def _plot_runs_per_half_inning(ax, pitch_log: list, away: str, home: str):
    """
    Team performance vs model expectation, per half-inning.

    BARS  — actual runs scored (a true count, not a rate). Away
    (Top) and home (Bottom) are separate bars per inning so the
    two teams' production is never blended, and one run doesn't
    get inflated by how many pitches were thrown before it scored.
    Relies on `runs_after_pitch`, set in live.py's resolve_inning():
    for the FIRST pitch logged in a half-inning, bat_score_at_pitch
    is the score at the start of that half-inning, so that entry's
    runs_after_pitch already equals the half-inning's total runs.

    MARKERS — the model's expected runs for that same half-inning,
    in the same units as the bars so they're directly comparable.
    Computed by taking the FIRST pitch of each at-bat (the model's
    read before that batter's outcome is known) and summing those
    per-at-bat probabilities across the half-inning. This avoids
    the old "average every pitch" approach, which over-weighted
    long at-bats and wasn't in run-count units at all.
    """
    ax.set_title("Team Performance — Actual vs Model-Expected Runs",
                 fontsize=10, pad=8)
    ax.set_xlabel("Inning", fontsize=9)
    ax.set_ylabel("Runs", fontsize=9)

    # group pitch_log by (inning, half)
    by_half = {}
    for e in pitch_log:
        key = (e.get("inning", 0), e.get("inning_half", "Top"))
        by_half.setdefault(key, []).append(e)

    if not by_half:
        ax.text(0.5, 0.5, "No inning data",
                ha="center", va="center",
                color="#a0a0a0", fontsize=11,
                transform=ax.transAxes)
        return

    def actual_runs(entries: list) -> float:
        # first entry logged in the half-inning already carries
        # the half-inning's total run count in runs_after_pitch
        return entries[0].get("runs_after_pitch", 0) if entries else 0

    def expected_runs(entries: list) -> float:
        # first pitch of each at-bat = model's pre-outcome read;
        # sum across at-bats = expected runs for the half-inning
        seen_ab   = set()
        total     = 0.0
        for e in entries:
            ab = e.get("at_bat_index")
            if ab in seen_ab:
                continue
            seen_ab.add(ab)
            total += e.get("probability", 0)
        return total

    innings = sorted(set(k[0] for k in by_half.keys()))
    x = np.arange(len(innings))
    w = 0.38

    away_actual, home_actual = [], []
    away_expected, home_expected = [], []
    for ing in innings:
        top_e = by_half.get((ing, "Top"), [])
        bot_e = by_half.get((ing, "Bottom"), [])
        away_actual.append(actual_runs(top_e))
        home_actual.append(actual_runs(bot_e))
        away_expected.append(expected_runs(top_e))
        home_expected.append(expected_runs(bot_e))

    b1 = ax.bar(x - w/2, away_actual, w, color=CLR_ACCENT,
                alpha=0.85, label=f"{away} (Top) — actual")
    b2 = ax.bar(x + w/2, home_actual, w, color=CLR_PINK,
                alpha=0.85, label=f"{home} (Bottom) — actual")

    # label bars with run counts, skip zeros for readability
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.05,
                        str(int(h)), ha="center", va="bottom",
                        fontsize=8, color="#e0e0e0", fontweight="bold")

    # model-expected overlay — diamonds on the same bar centers
    ax.scatter(x - w/2, away_expected, marker="D", s=45,
               color="#ffffff", edgecolors=CLR_ACCENT, linewidth=1.4,
               zorder=5, label=f"{away} — model expected")
    ax.scatter(x + w/2, home_expected, marker="D", s=45,
               color="#ffffff", edgecolors=CLR_PINK, linewidth=1.4,
               zorder=5, label=f"{home} — model expected")

    # flag innings where actual badly missed model expectation
    for xi, act, exp in zip(x - w/2, away_actual, away_expected):
        if abs(act - exp) > 1.0:
            ax.text(xi, max(act, exp) + 0.35, "⚠", ha="center",
                    fontsize=10, color=CLR_FALSE_NO_SCORE)
    for xi, act, exp in zip(x + w/2, home_actual, home_expected):
        if abs(act - exp) > 1.0:
            ax.text(xi, max(act, exp) + 0.35, "⚠", ha="center",
                    fontsize=10, color=CLR_FALSE_NO_SCORE)

    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in innings], fontsize=8)
    max_val = max(away_actual + home_actual + away_expected + home_expected,
                  default=1)
    ax.set_ylim(0, max_val + 1.2)
    ax.legend(fontsize=7.5, ncol=2, loc="upper left", framealpha=0.35)
    ax.grid(True, axis="y", alpha=0.35)

    total_away_act = sum(away_actual)
    total_home_act = sum(home_actual)
    total_away_exp = sum(away_expected)
    total_home_exp = sum(home_expected)
    ax.text(0.99, 0.97,
            f"Actual — {away} {total_away_act:.0f}, {home} {total_home_act:.0f}\n"
            f"Expected — {away} {total_away_exp:.1f}, {home} {total_home_exp:.1f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color="#a0a0a0")


def _plot_inning_calibration(ax, pitch_log: list):
    """Grouped bar chart: predicted vs actual run rate per inning."""
    ax.set_title("Predicted vs Actual Run Rate per Inning",
                 fontsize=10, pad=6)
    ax.set_xlabel("Inning", fontsize=9)
    ax.set_ylabel("Rate",   fontsize=9)

    by_inning = {}
    for e in pitch_log:
        ing = e.get("inning", 0)
        if ing not in by_inning:
            by_inning[ing] = {"pred": [], "actual": []}
        by_inning[ing]["pred"].append(e.get("probability", 0))
        act = e.get("actual_outcome")
        if act is not None:
            by_inning[ing]["actual"].append(act)

    innings     = sorted(by_inning.keys())
    x           = np.arange(len(innings))
    w           = 0.38
    pred_vals   = [np.mean(by_inning[i]["pred"]) for i in innings]
    actual_vals = [np.mean(by_inning[i]["actual"])
                   if by_inning[i]["actual"] else 0
                   for i in innings]

    b1 = ax.bar(x - w/2, pred_vals,   w,
                color=CLR_ACCENT, alpha=0.80, label="Predicted avg")
    b2 = ax.bar(x + w/2, actual_vals, w,
                color=CLR_PINK,   alpha=0.80, label="Actual run rate")

    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in innings], fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.35)

    # flag large gaps
    for xi, p, a in zip(x, pred_vals, actual_vals):
        if abs(p - a) > 0.10:
            ax.text(xi, max(p, a) + 0.04, "⚠",
                    ha="center", fontsize=11,
                    color=CLR_FALSE_NO_SCORE)


def _plot_wrong_distribution(ax, pitch_log: list):
    """Histogram of confidence gap on wrong predictions."""
    ax.set_title("Wrong Prediction Gap\n(how confidently wrong)",
                 fontsize=10, pad=8)
    ax.set_xlabel("|Predicted − Actual|", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)

    fs_gaps, fn_gaps = [], []
    for e in pitch_log:
        ft     = e.get("failure_type", "")
        prob   = e.get("probability", 0.5)
        actual = e.get("actual_outcome")
        if actual is None:
            continue
        gap = abs(prob - actual)
        if ft == "false_confidence_score":
            fs_gaps.append(gap)
        elif ft == "false_confidence_no_score":
            fn_gaps.append(gap)

    if not fs_gaps and not fn_gaps:
        ax.text(0.5, 0.5, "No wrong\npredictions!",
                ha="center", va="center",
                color=CLR_CORRECT_SCORE, fontsize=14,
                transform=ax.transAxes)
        return

    bins = np.linspace(0.25, 1.0, 9)
    if fs_gaps:
        ax.hist(fs_gaps, bins=bins, color=CLR_FALSE_SCORE,
                alpha=0.78, label=f"Said run, no run  (n={len(fs_gaps)})",
                edgecolor="#1a1a2e", linewidth=0.5)
    if fn_gaps:
        ax.hist(fn_gaps, bins=bins, color=CLR_FALSE_NO_SCORE,
                alpha=0.78, label=f"Said no run, run  (n={len(fn_gaps)})",
                edgecolor="#1a1a2e", linewidth=0.5)

    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.35)


def _plot_inning_table(ax, pitch_log: list):
    """Text table of per-inning stats rendered in matplotlib."""
    ax.set_title("Per-Inning Summary", fontsize=10, pad=8)
    ax.axis("off")

    by_inning = {}
    for e in pitch_log:
        ing  = e.get("inning", 0)
        half = e.get("inning_half", "Top")
        key  = f"{ing}{'T' if half=='Top' else 'B'}"
        if key not in by_inning:
            by_inning[key] = {
                "pred": [], "actual": [],
                "wrong": 0, "confident": 0
            }
        by_inning[key]["pred"].append(e.get("probability", 0))
        act = e.get("actual_outcome")
        if act is not None:
            by_inning[key]["actual"].append(act)
        ft = e.get("failure_type", "")
        if ft in ("false_confidence_score", "false_confidence_no_score"):
            by_inning[key]["wrong"] += 1
        if ft and ft != "uncertain":
            by_inning[key]["confident"] += 1

    headers = ["Inn", "Avg Prob", "Actual%", "Conf", "Wrong"]
    rows    = []
    for key in sorted(by_inning.keys()):
        d    = by_inning[key]
        pred = np.mean(d["pred"]) if d["pred"] else 0
        act  = np.mean(d["actual"]) if d["actual"] else 0
        rows.append([
            key,
            f"{pred:.0%}",
            f"{act:.0%}",
            str(d["confident"]),
            str(d["wrong"]) if d["wrong"] else "—",
        ])

    col_widths = [0.12, 0.22, 0.22, 0.14, 0.14]
    x_starts   = [0.02, 0.14, 0.38, 0.62, 0.78]

    # header row
    for hdr, x in zip(headers, x_starts):
        ax.text(x, 0.95, hdr, transform=ax.transAxes,
                fontsize=9, fontweight="bold",
                color=CLR_ACCENT, va="top")

    # draw separator line in axes coordinates using plot
    ax.plot([0, 1], [0.93, 0.93], color="#0f3460",
            linewidth=0.8, transform=ax.transAxes,
            clip_on=False)

    # data rows
    row_h = min(0.07, 0.85 / max(len(rows), 1))
    for ri, row in enumerate(rows):
        y = 0.90 - ri * row_h
        if y < 0.02:
            break
        bg = "#1a1a3a" if ri % 2 == 0 else "#16213e"
        ax.add_patch(plt.Rectangle(
            (0, y - row_h * 0.1), 1, row_h * 0.9,
            transform=ax.transAxes,
            color=bg, zorder=0
        ))
        wrong_flag = row[4] not in ("—", "0")
        for val, x in zip(row, x_starts):
            color = CLR_FALSE_NO_SCORE if (
                wrong_flag and val == row[4]
            ) else "#e0e0e0"
            ax.text(x, y, val, transform=ax.transAxes,
                    fontsize=8, color=color, va="top")


def plot_player_bars(log: dict, title_base: str = "",
                     save_path: str = None):
    """
    Two figures: one for batters, one for pitchers.
    Each player gets a stacked horizontal bar with enough
    vertical space that names never overlap.
    """
    _clean_nan_markers(log)

    game_info = log.get("game_info", {})
    summary   = log.get("summary", {})
    away      = game_info.get("away_team", "Away")
    home      = game_info.get("home_team", "Home")
    if not title_base:
        title_base = f"{away} @ {home}"

    all_players = []

    def collect(player_dict, role):
        """
        Flattens team-split or flat player dicts into all_players.
        Uses a per-call seen set so batters and pitchers don't
        interfere with each other's deduplication.
        team-split format: {team_name: {player_id: player_dict}}
        flat format:       {player_id: player_dict}
        """
        if not player_dict:
            return

        seen = set()   # scoped per collect() call — separate for batters/pitchers

        # detect format: in team-split the first value is a dict
        # of players (keyed by player id), so it won't have "name"
        first = next(iter(player_dict.values()), {})
        is_split = (
            isinstance(first, dict)
            and "name" not in first
            and len(first) > 0
        )

        if is_split:
            for team, members in player_dict.items():
                if not isinstance(members, dict):
                    continue
                for pid, p in members.items():
                    if not (isinstance(p, dict) and "name" in p):
                        continue
                    # dedup by player name within this role
                    key = p["name"]
                    if key not in seen:
                        seen.add(key)
                        all_players.append({
                            **p,
                            "role": role,
                            "team": team,
                        })
        else:
            for pid, p in player_dict.items():
                if not (isinstance(p, dict) and "name" in p):
                    continue
                key = p["name"]
                if key not in seen:
                    seen.add(key)
                    all_players.append({
                        **p,
                        "role": role,
                        "team": "All",
                    })

    collect(summary.get("batters",  {}), "Batter")
    collect(summary.get("pitchers", {}), "Pitcher")

    # debug — print what was collected so issues are visible
    batter_names  = [p["name"] for p in all_players if p["role"] == "Batter"]
    pitcher_names = [p["name"] for p in all_players if p["role"] == "Pitcher"]
    print(f"  Collected {len(batter_names)} unique batters, "
          f"{len(pitcher_names)} unique pitchers")

    batter_players  = sorted(
        [p for p in all_players if p["role"] == "Batter"],
        key=lambda x: x.get("xwoba", 0), reverse=True
    )
    pitcher_players = sorted(
        [p for p in all_players if p["role"] == "Pitcher"],
        key=lambda x: x.get("xwoba", 0), reverse=True
    )

    for players, label, suffix in [
        (batter_players,  "Batters",  "batter_bars"),
        (pitcher_players, "Pitchers", "pitcher_bars"),
    ]:
        if not players:
            continue

        n = len(players)
        # give each row at least 0.5in — more for small lists
        row_h  = max(0.50, min(0.90, 8.0 / max(n, 1)))
        height = max(5, n * row_h + 2)   # +2 for title/legend
        # font size scales down as player count grows
        font_sz = max(7, min(10, int(90 / max(n, 9))))

        fig, ax = plt.subplots(figsize=(14, height),
                               facecolor="#1a1a2e")
        fig.suptitle(
            f"GhostRunner — {label}\n{title_base}",
            fontsize=12, fontweight="bold",
            color=CLR_ACCENT, y=1.0
        )

        _draw_player_bars(ax, players, label, font_sz=font_sz)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        _save_or_show(fig, save_path, suffix)


def _draw_player_bars(ax, players: list, title: str,
                      font_sz: int = 9):
    """Stacked horizontal bars — one player per row."""
    ax.set_title(title, fontsize=11, pad=8)

    if not players:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, color="#a0a0a0")
        return

    n = len(players)
    # bar height shrinks slightly for large lists so bars fit
    bar_h = max(0.45, min(0.72, 10.0 / max(n, 10)))

    names  = []
    cs_v, fs_v, cn_v, fn_v = [], [], [], []

    for p in players:
        marker = p.get("marker", "")
        xwoba  = p.get("xwoba",  0)
        name   = p.get("name", "?")
        # truncate: first initial + last name for long names
        if len(name) > 18:
            parts = name.split()
            name  = (f"{parts[0][0]}. {' '.join(parts[1:])}"
                     if len(parts) > 1 else name[:18])
        # keep total label under 28 chars
        label_str = f"{name}{marker}  .{int(xwoba*1000):03d}"
        names.append(label_str[:30])
        cs_v.append(p.get("correct_confident_score",    0))
        fs_v.append(p.get("false_confidence_score",     0))
        cn_v.append(p.get("correct_confident_no_score", 0))
        fn_v.append(p.get("false_confidence_no_score",  0))

    y    = np.arange(n)
    left = np.zeros(n)

    for vals, color, lbl in [
        (cs_v, CLR_CORRECT_SCORE,    "✓ Said run — run scored"),
        (fs_v, CLR_FALSE_SCORE,      "✗ Said run — no run"),
        (cn_v, CLR_CORRECT_NO_SCORE, "✓ Said no run — no run"),
        (fn_v, CLR_FALSE_NO_SCORE,   "✗ Said no run — run scored"),
    ]:
        arr = np.array(vals)
        ax.barh(y, arr, left=left, color=color, alpha=0.87,
                edgecolor="#1a1a2e", linewidth=0.4,
                label=lbl, height=bar_h)
        for yi, (v, l) in enumerate(zip(arr, left)):
            if v > 0:
                ax.text(l + v / 2, yi, str(int(v)),
                        ha="center", va="center",
                        fontsize=max(6, font_sz - 1),
                        color="#111", fontweight="bold")
        left = left + arr

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=font_sz)
    # add extra left margin so long names aren't cut off
    ax.tick_params(axis="y", pad=4)
    ax.set_xlabel("Confident predictions", fontsize=9)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.4)
    ax.grid(True, axis="x", alpha=0.35)
    ax.set_xlim(0, max(left.max() + 2, 5))
    # ensure y axis has enough room — invert so top player is first
    ax.set_ylim(-0.7, n - 0.3)
    ax.invert_yaxis()


# ══════════════════════════════════════════════════════════════
# 3. HELPERS
# ══════════════════════════════════════════════════════════════

def _legend_patches(ax, items: list, loc: str = "upper left"):
    patches = [mpatches.Patch(color=c, label=l) for c, l in items]
    ax.legend(handles=patches, fontsize=8,
              loc=loc, framealpha=0.35)


def _clean_nan_markers(log: dict):
    for entry in log.get("pitch_log", []):
        for key in ("batter_marker", "pitcher_marker"):
            if entry.get(key) in ("nan", "NaN", None):
                entry[key] = ""
    summary = log.get("summary", {})
    for group in list(summary.get("batters", {}).values()) + \
                 list(summary.get("pitchers", {}).values()):
        if isinstance(group, dict):
            for p in group.values():
                if isinstance(p, dict) and \
                        p.get("marker") in ("nan","NaN",None):
                    p["marker"] = ""
            if group.get("marker") in ("nan","NaN",None):
                group["marker"] = ""


def _save_or_show(fig, save_path: Optional[str], suffix: str):
    import warnings
    if save_path:
        base, ext = os.path.splitext(save_path)
        path = f"{base}_{suffix}{ext or '.png'}"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig.savefig(path, bbox_inches="tight", dpi=120)
        print(f"  Saved: {path}")
        plt.close(fig)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                plt.tight_layout()
            except Exception:
                pass
        plt.show()
        plt.close(fig)


# ══════════════════════════════════════════════════════════════
# 4. ENTRY POINT
# ══════════════════════════════════════════════════════════════

def plot_from_menu(log_path: str = None,
                   pipeline_path: str = None,
                   calibration_data: dict = None):
    if log_path and os.path.exists(log_path):
        print(f"\n  Generating game charts for {log_path}...")
        plot_game(log_path)
    if pipeline_path and os.path.exists(pipeline_path):
        print(f"\n  Generating training model charts...")
        plot_training(pipeline_path, calibration_data)
    if not log_path and not pipeline_path:
        print("  No paths provided — nothing to plot.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg.endswith(".json"):
            plot_game(arg)
        elif arg.endswith(".pkl"):
            plot_training(arg)
        else:
            print(f"  Unknown file type: {arg}")
    else:
        # interactive game selection menu
        log_path = select_game_log()
        if log_path:
            plot_game(log_path)
        else:
            # offer training charts instead
            import glob
            pkl_files = sorted(glob.glob("models/*.pkl"), reverse=True)
            if pkl_files:
                show = input(
                    f"\n  Show training model charts? ({pkl_files[0]}) (y/n): "
                ).strip().lower()
                if show == "y":
                    plot_training(pkl_files[0])