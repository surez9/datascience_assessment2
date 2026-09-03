# TASK 1: Goals conceded By European vs Non-European teams in FIFA World Cup 2026
# Note: Comparision is done for the team who qualified for Round of 32 i.e. not considering teams eliminated in Group Stage
# Suresh Bhandari (S400969)

import pandas as pd
import numpy as np
import scipy.stats as st

pd.set_option("display.width", 200)

# Information about the dataset used
details = pd.read_csv("dataset/post_match_details.csv")
stats = pd.read_csv("dataset/post_match_stats.csv")

details["home_score_final"] = details["home_score"] + details["extra_time_score_home"].fillna(0)
details["away_score_final"] = details["away_score"] + details["extra_time_score_away"].fillna(0)


stats_slim = stats[["event_id", "home_goals_prevented", "away_goals_prevented"]]
matches = details.merge(stats_slim, on="event_id", how="left")


def match_winner(row):
    if row["home_score_final"] > row["away_score_final"]:
        return row["home_team"]
    if row["away_score_final"] > row["home_score_final"]:
        return row["away_team"]
    if pd.notna(row["penalty_shootout_home"]):
        return row["home_team"] if row["penalty_shootout_home"] > row["penalty_shootout_away"] else row["away_team"]
    return None

matches["Winner"] = matches.apply(match_winner, axis=1)

stage_order_map = {
    None: 1, "Round of 32": 2, "Round of 16": 3, "Quarterfinals": 4,
    "Semifinals": 5, "Match for 3rd place": 6, "Final": 6,
}
matches["StageOrder"] = matches["round_name"].map(stage_order_map).fillna(1).astype(int)

home_rows = matches.rename(columns={
    "home_team": "Team", "away_team": "Opponent",
    "home_score_final": "GoalsScored", "away_score_final": "GoalsConceded",
    "home_goals_prevented": "GoalsPrevented",
})[["event_id", "Team", "Opponent", "GoalsScored", "GoalsConceded", "GoalsPrevented"]]

away_rows = matches.rename(columns={
    "away_team": "Team", "home_team": "Opponent",
    "away_score_final": "GoalsScored", "home_score_final": "GoalsConceded",
    "away_goals_prevented": "GoalsPrevented",
})[["event_id", "Team", "Opponent", "GoalsScored", "GoalsConceded", "GoalsPrevented"]]

long_df = pd.concat([home_rows, away_rows], ignore_index=True)

team_agg = long_df.groupby("Team").agg(
    MatchesPlayed=("event_id", "count"),
    TotalGoalsConceded=("GoalsConceded", "sum"),
    MeanGoalsPrevented=("GoalsPrevented", "mean"),
).reset_index()
team_agg["GoalsConcededPerMatch"] = (team_agg["TotalGoalsConceded"] / team_agg["MatchesPlayed"]).round(3)
team_agg["MeanGoalsPrevented"] = team_agg["MeanGoalsPrevented"].round(3)

assert len(team_agg) == 48, f"Expected 48 teams, got {len(team_agg)}"

# Furthest stage reached per team with winner, runner up, 3rd or 4th
# Resolved from the final and 3rd place match rows directly
team_stage = pd.concat([
    matches[["home_team", "StageOrder"]].rename(columns={"home_team": "Team"}),
    matches[["away_team", "StageOrder"]].rename(columns={"away_team": "Team"}),
]).groupby("Team")["StageOrder"].max().reset_index()

final_row = matches[matches["round_name"] == "Final"].iloc[0]
third_row = matches[matches["round_name"] == "Match for 3rd place"].iloc[0]
winner = final_row["Winner"]
runner_up = final_row["away_team"] if winner == final_row["home_team"] else final_row["home_team"]
third = third_row["Winner"]
fourth = third_row["away_team"] if third == third_row["home_team"] else third_row["home_team"]

stage_label_map = {1: "Group Stage", 2: "Round of 32", 3: "Round of 16",
                    4: "Quarterfinal", 5: "Semifinal"}

def label_stage(row):
    if row["Team"] == winner:
        return "Winner"
    if row["Team"] == runner_up:
        return "Runner-up"
    if row["Team"] == third:
        return "Third Place"
    if row["Team"] == fourth:
        return "Fourth Place"
    return stage_label_map[row["StageOrder"]]

team_stage["Stage"] = team_stage.apply(label_stage, axis=1)

# Stage Order: Used for pointing the team's eliminated stage
stage_final_order = {"Group Stage": 1, "Round of 32": 2, "Round of 16": 3,
                      "Quarterfinal": 4, "Fourth Place": 5, "Third Place": 6,
                      "Runner-up": 7, "Winner": 8}

team_stage["StageOrder"] = team_stage["Stage"].map(stage_final_order)

# Region: Europe = UEFA teams, others = Non-Europe
uefa_teams = {
    "Spain", "Portugal", "Switzerland", "Belgium", "Netherlands", "Germany",
    "France", "England", "Croatia", "Norway", "Bosnia & Herzegovina",
    "Austria", "Sweden", "Czechia", "Scotland", "Türkiye",
}

df_full = team_agg.merge(team_stage[["Team", "Stage", "StageOrder"]], on="Team")
df_full["Region"] = df_full["Team"].apply(lambda t: "Europe" if t in uefa_teams else "Non-Europe")
df_full = df_full[["Team", "Region", "Stage", "StageOrder", "MatchesPlayed","GoalsConcededPerMatch", "MeanGoalsPrevented"]]
df_full = df_full.sort_values("StageOrder", ascending=False).reset_index(drop=True)


print(f"Total team played in World Cup 2026: {len(df_full)} teams")
print(df_full.to_string(index=False))
print("\n",df_full["Region"].value_counts())
print("\n",df_full["Stage"].value_counts())


# Round of 32 (StageOrder >= 2), i.e. survived the group stage.
df = df_full[df_full["StageOrder"] >= 2].copy()

print(f"\nAfter Round of 32+: {len(df_full) - len(df)} eliminated in the group stage")
print("Remaining Count of Teams based on region")
print(df["Region"].value_counts())

europe = df[df["Region"] == "Europe"]["GoalsConcededPerMatch"].to_numpy()
non_europe = df[df["Region"] == "Non-Europe"]["GoalsConcededPerMatch"].to_numpy()
n1, n2 = len(europe), len(non_europe)

# Descriptive statistics
print("\nDescriptive statistics")
for label, sample in [("European teams", europe), ("Non-European teams", non_europe)]:
    mean, median = np.mean(sample), np.median(sample)
    sd = np.std(sample, ddof=1)
    se = sd / np.sqrt(len(sample))
    print(f"\n{label} (n={len(sample)}):")
    print(f"  Mean={mean:.3f}  Median={median:.3f}  SD={sd:.3f}  SE={se:.3f}")
    print(f"  Min={sample.min():.2f}  Max={sample.max():.2f}")


# Inferential stats - Confidence Interval

print("95% CONFIDENCE INTERVALS (t-distribution, n < 30)")
for label, sample in [("European teams", europe), ("Non-European teams", non_europe)]:
    n = len(sample)
    mean = np.mean(sample)
    se = np.std(sample, ddof=1) / np.sqrt(n)
    t_crit = st.t.ppf(0.975, df=n - 1)
    moe = t_crit * se
    print(f"{label}: mean={mean:.3f}, 95% CI = ({mean - moe:.3f}, {mean + moe:.3f})")


# Inferential stats - Two-sample t-test, Welch's
print("TWO-SAMPLE T-TEST: Europe vs Non-Europe?")
print("H0: mu_Europe = mu_NonEurope   Ha: mu_Europe != mu_NonEurope")
print("=" * 70)
t_stat, p_val = st.ttest_ind(europe, non_europe, equal_var=False)
print(f"t* = {t_stat:.3f}, p-value = {p_val:.4f}")
print("Reject H0" if p_val < 0.05 else "Fail to reject H0", "at alpha = 0.05")

mean_diff = np.mean(europe) - np.mean(non_europe)
se1, se2 = np.std(europe, ddof=1) / np.sqrt(n1), np.std(non_europe, ddof=1) / np.sqrt(n2)
se_diff = np.sqrt(se1**2 + se2**2)
df_welch = (se1**2 + se2**2)**2 / ((se1**4 / (n1 - 1)) + (se2**4 / (n2 - 1)))
t_crit_diff = st.t.ppf(0.975, df=df_welch)
moe_diff = t_crit_diff * se_diff
print(f"\nMean difference (Europe - Non-Europe): {mean_diff:.3f}")
print(f"95% CI for the difference: ({mean_diff - moe_diff:.3f}, {mean_diff + moe_diff:.3f})")
