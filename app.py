"""
NFL Picks dashboard. Run locally: streamlit run app.py
Deployed at share.streamlit.io for sharing with friends.
"""
from pathlib import Path
import datetime as dt
import pickle
import pandas as pd
import nflreadpy as nfl
import streamlit as st
from scipy.stats import norm

from weather import forecast_wind_for_game, under_recommendation

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "margin_model.pkl"
TEAM_WEEK_PATH = ROOT / "data" / "processed" / "team_week.parquet"

HFA = 55
MARGIN_STD = 13.5  # NFL game margin std dev — used for margin→win-prob conversion

st.set_page_config(page_title="NFL Picks", page_icon="🏈", layout="wide")

# ----- loaders (cached) -----
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_data(ttl=3600)
def load_team_week():
    return pd.read_parquet(TEAM_WEEK_PATH)

@st.cache_data(ttl=900)  # refresh every 15 min during gameday
def load_schedule():
    season = nfl.get_current_season()
    current_week = nfl.get_current_week()
    sched = nfl.load_schedules(seasons=[season]).to_pandas()
    return sched, season, current_week

@st.cache_data(ttl=900)
def load_qb_status(season, week):
    """For each team, find QB1 (from depth chart) and their injury status for the week."""
    try:
        dc = nfl.load_depth_charts(seasons=[season]).to_pandas()
        qb1 = (
            dc[(dc["position"] == "QB") & (dc["depth_team"] == "1") & (dc["week"] == week)]
            [["club_code", "full_name", "gsis_id"]]
            .rename(columns={"club_code": "team"})
            .drop_duplicates(subset=["team"])
        )
        inj = nfl.load_injuries(seasons=[season]).to_pandas()
        wk = inj[(inj["week"] == week) & (inj["position"] == "QB")][
            ["team", "gsis_id", "report_status", "report_primary_injury"]
        ]
        return qb1.merge(wk, on=["team", "gsis_id"], how="left")
    except Exception:
        return pd.DataFrame(columns=["team", "full_name", "report_status"])

def qb_alert_for(team, qb1_status):
    row = qb1_status[qb1_status["team"] == team]
    if row.empty:
        return ""
    r = row.iloc[0]
    status = r.get("report_status")
    name = r.get("full_name", "QB1")
    if status == "Out":
        return f"🚨 {name} OUT"
    if status == "Doubtful":
        return f"⚠️ {name} doubtful"
    if status == "Questionable":
        return f"❓ {name} questionable"
    return ""

# ----- header -----
st.title("🏈 NFL Picks")

try:
    bundle = load_model()
    model = bundle["model"]
    feature_cols = bundle["features"]
    elo_ratings = bundle["elo_ratings"]
    tw = load_team_week()
    sched, season, current_week = load_schedule()
except Exception as e:
    st.error(f"Couldn't load model or data: {e}")
    st.stop()

reg_sched = sched[sched["game_type"] == "REG"].copy()
weeks_available = sorted(reg_sched["week"].dropna().unique().astype(int).tolist())

if not weeks_available:
    st.info(f"No regular-season games found for {season} yet. Check back when the season starts.")
    st.stop()

default_week = current_week if current_week in weeks_available else weeks_available[-1]
col1, col2 = st.columns([1, 3])
with col1:
    week = st.selectbox("Week", weeks_available, index=weeks_available.index(default_week))
with col2:
    st.caption(f"Season {season} · Showing Week {week} · Data refreshes every 15 min")

# ----- build features for selected week -----
upcoming = reg_sched[reg_sched["week"] == week].copy()
if upcoming.empty:
    st.info("No games this week.")
    st.stop()

# Most recent team-week stats per team (already lagged in notebook 02)
latest = tw.sort_values(["team", "season", "week"]).groupby("team").tail(1)
stat_cols = [c for c in tw.columns if c not in ["season", "week", "team"]]

home = latest.rename(columns={"team": "home_team", **{c: f"home_{c}" for c in stat_cols}})
away = latest.rename(columns={"team": "away_team", **{c: f"away_{c}" for c in stat_cols}})

g = upcoming.merge(home.drop(columns=["season", "week"]), on="home_team", how="left")
g = g.merge(away.drop(columns=["season", "week"]), on="away_team", how="left")

raw = ["off_epa_play", "off_success_rate", "off_explosive_rate",
       "def_epa_play", "def_success_rate", "def_explosive_rate"]
for c in raw:
    for w in ["l4", "l8", "ytd"]:
        g[f"diff_{c}_{w}"] = g[f"home_{c}_{w}"] - g[f"away_{c}_{w}"]

g["home_elo"] = g["home_team"].map(elo_ratings).fillna(1500)
g["away_elo"] = g["away_team"].map(elo_ratings).fillna(1500)
g["elo_diff"] = g["home_elo"] - g["away_elo"] + HFA
g["rest_diff"] = g["home_rest"].fillna(7) - g["away_rest"].fillna(7)

# Predict
g["pred_margin"] = model.predict(g[feature_cols])
g["home_win_prob"] = norm.cdf(g["pred_margin"] / MARGIN_STD)

# ATS edge: home covers when home_margin > spread_line (nflverse: spread_line > 0 = home favored)
g["ats_edge"] = g["pred_margin"] - g["spread_line"]
g["ats_edge_pts"] = g["ats_edge"].abs()
g["ats_pick_team"] = g.apply(
    lambda r: r["home_team"] if r["ats_edge"] > 0 else r["away_team"], axis=1
)

def tier(e):
    if e >= 5: return "🔥 STRONG"
    if e >= 3: return "📈 LEAN"
    if e >= 1.5: return "🤏 WEAK"
    return "— PASS"

g["confidence"] = g["ats_edge_pts"].apply(tier)

def fmt_line(home_team, spread):
    if pd.isna(spread):
        return "—"
    if spread > 0:
        return f"{home_team} -{spread:g}"
    if spread < 0:
        return f"{home_team} +{abs(spread):g}"
    return "PK"

g["vegas_line"] = g.apply(lambda r: fmt_line(r["home_team"], r["spread_line"]), axis=1)
g["model_line"] = g.apply(lambda r: fmt_line(r["home_team"], -r["pred_margin"]), axis=1)
g["matchup"] = g["away_team"] + " @ " + g["home_team"]

# QB status alerts (display only — not used in model, see diagnostics for why)
qb1_status = load_qb_status(season, week)
g["home_qb"] = g["home_team"].apply(lambda t: qb_alert_for(t, qb1_status))
g["away_qb"] = g["away_team"].apply(lambda t: qb_alert_for(t, qb1_status))
g["qb_notes"] = g.apply(
    lambda r: " · ".join(x for x in [r["away_qb"], r["home_qb"]] if x), axis=1
)

# ----- ATS picks table -----
st.subheader("Against the Spread")

confidence_filter = st.multiselect(
    "Confidence",
    ["🔥 STRONG", "📈 LEAN", "🤏 WEAK", "— PASS"],
    default=["🔥 STRONG", "📈 LEAN"],
)

ats_view = (
    g[g["confidence"].isin(confidence_filter)]
    .sort_values("ats_edge_pts", ascending=False)
    [["matchup", "vegas_line", "model_line", "ats_pick_team", "ats_edge_pts", "confidence", "qb_notes"]]
    .rename(columns={
        "matchup": "Matchup",
        "vegas_line": "Vegas",
        "model_line": "Model",
        "ats_pick_team": "Pick",
        "ats_edge_pts": "Edge (pts)",
        "confidence": "Confidence",
        "qb_notes": "QB status",
    })
)

st.dataframe(
    ats_view,
    hide_index=True,
    use_container_width=True,
    column_config={
        "Edge (pts)": st.column_config.NumberColumn(format="%.1f"),
    },
)

# ----- Straight-up picks -----
st.subheader("Straight-Up (for survivor pools)")

su = g.copy()
su["pick_team"] = su.apply(
    lambda r: r["home_team"] if r["home_win_prob"] >= 0.5 else r["away_team"], axis=1
)
su["pick_win_prob"] = su["home_win_prob"].where(
    su["home_win_prob"] >= 0.5, 1 - su["home_win_prob"]
)
su["opponent"] = su.apply(
    lambda r: r["away_team"] if r["pick_team"] == r["home_team"] else r["home_team"], axis=1
)
su["loc"] = su.apply(
    lambda r: "vs" if r["pick_team"] == r["home_team"] else "@", axis=1
)

su_view = (
    su.sort_values("pick_win_prob", ascending=False)
    [["pick_team", "loc", "opponent", "pick_win_prob"]]
    .rename(columns={
        "pick_team": "Pick",
        "loc": "",
        "opponent": "Opp",
        "pick_win_prob": "Win prob",
    })
)

st.dataframe(
    su_view,
    hide_index=True,
    use_container_width=True,
    column_config={
        "Win prob": st.column_config.ProgressColumn(format="%.0f%%", min_value=0.5, max_value=1.0),
    },
)

# ----- Totals: wind UNDER plays (Phase 2 weather edge) -----
st.subheader("Totals — Wind UNDER plays")
st.caption(
    "Backed by 1999-2025 nflverse data: outdoor games with wind ≥10 mph hit UNDER 54.7% "
    "(n=1,770); wind ≥15 mph hits 56.2% (n=642). Forecasts via NOAA, refresh every 15 min."
)

@st.cache_data(ttl=900, show_spinner=False)
def fetch_wind_for_game(home_team: str, kickoff_iso: str) -> float | None:
    """Cached wrapper — keyed on (team, kickoff). Returns mph or None."""
    if not kickoff_iso:
        return None
    return forecast_wind_for_game(home_team, dt.datetime.fromisoformat(kickoff_iso))

def kickoff_utc(row) -> str:
    """Best-effort kickoff timestamp from nflverse columns. Returns ISO or ''."""
    gd, gt = row.get("gameday"), row.get("gametime")
    if pd.isna(gd) or not gd:
        return ""
    try:
        if gt and not pd.isna(gt):
            local = dt.datetime.fromisoformat(f"{gd}T{gt}")
        else:
            local = dt.datetime.fromisoformat(f"{gd}T13:00:00")
        # nflverse times are US Eastern; convert to UTC for NOAA matching
        return (local - dt.timedelta(hours=5)).replace(tzinfo=dt.timezone.utc).isoformat()
    except (ValueError, TypeError):
        return ""

totals = g[["matchup", "home_team", "away_team", "total_line", "roof", "gameday", "gametime"]].copy()
totals["kickoff_iso"] = totals.apply(kickoff_utc, axis=1)
totals["wind_mph"] = totals.apply(
    lambda r: fetch_wind_for_game(r["home_team"], r["kickoff_iso"]), axis=1
)
totals[["rec", "reason"]] = totals.apply(
    lambda r: pd.Series(under_recommendation(r["wind_mph"], r["roof"])), axis=1
)

totals_view = (
    totals.sort_values(
        ["rec", "wind_mph"],
        key=lambda s: s.map({"🔥 STRONG UNDER": 0, "📈 LEAN UNDER": 1, "⏳ pending": 2, "— pass": 3, "— indoor": 4})
        if s.name == "rec" else -s.fillna(-1),
    )
    [["matchup", "total_line", "roof", "wind_mph", "rec", "reason"]]
    .rename(columns={
        "matchup": "Matchup",
        "total_line": "Total",
        "roof": "Roof",
        "wind_mph": "Wind (mph)",
        "rec": "Pick",
        "reason": "Why",
    })
)

st.dataframe(
    totals_view,
    hide_index=True,
    use_container_width=True,
    column_config={
        "Total": st.column_config.NumberColumn(format="%.1f"),
        "Wind (mph)": st.column_config.NumberColumn(format="%.0f"),
    },
)

# ----- diagnostics -----
with st.expander("Model diagnostics"):
    st.markdown(
        """
        **Backtest (walk-forward, training on prior seasons only):**

        | Season | MAE | Straight-Up % | ATS % |
        |---|---|---|---|
        | 2023 | 11.4 | 55.9% | 46.9% |
        | 2024 | 10.4 | 64.5% | 55.9% |
        | 2025 | 10.5 | 62.1% | 53.9% |
        | **Mean** | **10.8** | **60.8%** | **52.2%** |

        **Reality check:** 52.4% ATS is break-even at -110 vig. Real edge in this model is straight-up
        picks (~61%), not against the spread. Don't bet the farm on STRONG picks — sample size is small.
        """
    )
    st.caption(f"Model: XGBoost margin regressor + 538-style Elo. Features: {len(feature_cols)} EPA-based diffs + Elo diff + rest diff.")

st.caption("Not financial advice. Bet responsibly.")
