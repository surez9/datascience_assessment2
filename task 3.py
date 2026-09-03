# 1. DATA LOADING

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

player_df_raw = pd.read_csv("player_stats.csv")

print("Dataset loaded. Shape:", player_df_raw.shape)
display(player_df_raw.head())

# 2. DATA INSPECTION
print("Unique players:", player_df_raw["player_id"].nunique())
print("Rows (player-level records):", len(player_df_raw))

print("\nPosition categories (raw codes):")
print(player_df_raw["position"].value_counts())

print("\nMissing values in key columns:")
print(player_df_raw[["player_name", "team_id", "position",
                      "goals", "assists"]].isnull().sum())

print("\nDuplicate player_id:", player_df_raw["player_id"].duplicated().sum())

# 3. DATA CLEANING & STANDARDISATION

position_map = {
    "FWD": "Forward",
    "MID": "Midfielder",
    "DEF": "Defender",
    "GK": "Goalkeeper",
}
player_df = player_df_raw.copy()
player_df["position"] = player_df["position"].map(position_map)

player_df["goals"] = pd.to_numeric(player_df["goals"], errors="coerce")
player_df["assists"] = pd.to_numeric(player_df["assists"], errors="coerce")

print("Player-level dataset shape:", player_df.shape)
print("\nPosition distribution (all players):")
print(player_df["position"].value_counts())

# 4. FILTERING — INCLUSION / EXCLUSION CRITERIA

eligible_players = player_df[
    (player_df["goals"] > 0) | (player_df["assists"] > 0)
].copy()

print("Eligible players (>=1 goal or assist):", len(eligible_players))

analysis_df = eligible_players[
    eligible_players["position"].isin(["Forward", "Midfielder"])
].copy()

print("\nFinal analysis dataset shape:", analysis_df.shape)
print(analysis_df["position"].value_counts())
display(analysis_df.head(10))

# 5. DATA PREPARATION AND SAMPLING
n_forward = (analysis_df["position"] == "Forward").sum()
n_midfielder = (analysis_df["position"] == "Midfielder").sum()
total_n = len(analysis_df)

print(f"Forwards:    n = {n_forward}  ({n_forward/total_n:.1%} of sample)")
print(f"Midfielders: n = {n_midfielder}  ({n_midfielder/total_n:.1%} of sample)")

# 6. DESCRIPTIVE STATISTICS

grouped_stats = (
    analysis_df.groupby("position")["assists"]
    .agg(n="count", mean="mean", median="median", std="std",
         min="min", max="max")
)

quartile_stats = (
    analysis_df.groupby("position")["assists"]
    .agg(Q1=lambda x: x.quantile(0.25),
         Q3=lambda x: x.quantile(0.75))
)
quartile_stats["IQR"] = quartile_stats["Q3"] - quartile_stats["Q1"]

descriptive_table = grouped_stats.join(quartile_stats)
print("Descriptive statistics — assists by position:")
display(descriptive_table)

forward_assists = analysis_df.loc[analysis_df["position"] == "Forward", "assists"]
midfielder_assists = analysis_df.loc[analysis_df["position"] == "Midfielder", "assists"]

# Boxplot comparison
plt.figure(figsize=(8, 6))
plt.boxplot([forward_assists, midfielder_assists],
            tick_labels=["Forward", "Midfielder"])
plt.title("Distribution of Total Assists by Player Position")
plt.xlabel("Player Position")
plt.ylabel("Total Assists")
plt.show()

# 7. 95% CONFIDENCE INTERVALS (per group)

forward_ci = stats.t.interval(
    confidence=0.95, df=len(forward_assists) - 1,
    loc=forward_assists.mean(), scale=stats.sem(forward_assists)
)
midfielder_ci = stats.t.interval(
    confidence=0.95, df=len(midfielder_assists) - 1,
    loc=midfielder_assists.mean(), scale=stats.sem(midfielder_assists)
)

print("95% Confidence Intervals for mean assists")
print(f"Forward:    mean = {forward_assists.mean():.3f}, "
      f"95% CI = ({forward_ci[0]:.3f}, {forward_ci[1]:.3f})")
print(f"Midfielder: mean = {midfielder_assists.mean():.3f}, "
      f"95% CI = ({midfielder_ci[0]:.3f}, {midfielder_ci[1]:.3f})")

# 8. ASSUMPTION CHECKS

shapiro_forward = stats.shapiro(forward_assists)
shapiro_midfielder = stats.shapiro(midfielder_assists)

print("Shapiro-Wilk normality test (raw assists distribution)")
print(f"Forward:    W = {shapiro_forward.statistic:.4f}, "
      f"p = {shapiro_forward.pvalue:.2e}")
print(f"Midfielder: W = {shapiro_midfielder.statistic:.4f}, "
      f"p = {shapiro_midfielder.pvalue:.2e}")
print("\n(Both groups depart from normality, as expected for a sparse "
      "count variable. Welch's t-test is used, and with n > 100 per "
      "group the CLT supports approximate normality of the sample means.)")

print(f"\nForward variance:    {forward_assists.var():.4f}")
print(f"Midfielder variance: {midfielder_assists.var():.4f}")

# 9. WELCH'S INDEPENDENT TWO-SAMPLE T-TEST
t_statistic, p_value = stats.ttest_ind(
    forward_assists, midfielder_assists, equal_var=False
)

# 95% CI for Mean Difference
forward_mean, midfielder_mean = forward_assists.mean(), midfielder_assists.mean()
forward_var, midfielder_var = forward_assists.var(ddof=1), midfielder_assists.var(ddof=1)
mean_difference = forward_mean - midfielder_mean

se_diff = np.sqrt(forward_var / n_forward + midfielder_var / n_midfielder)
df_welch = ((forward_var / n_forward + midfielder_var / n_midfielder) ** 2 /
            ((forward_var / n_forward) ** 2 / (n_forward - 1) +
             (midfielder_var / n_midfielder) ** 2 / (n_midfielder - 1)))
t_crit = stats.t.ppf(0.975, df_welch)
diff_ci = (mean_difference - t_crit * se_diff, mean_difference + t_crit * se_diff)

print("Welch's Independent Two-Sample t-Test")
print("--------------------------------------")
print(f"Forward mean assists:    {forward_mean:.4f}")
print(f"Midfielder mean assists: {midfielder_mean:.4f}")
print(f"Difference (F - M):      {mean_difference:.4f}")
print(f"t-statistic:             {t_statistic:.4f}")
print(f"Degrees of freedom:      {df_welch:.2f}")
print(f"p-value:                 {p_value:.4f}")
print(f"95% CI for difference:   ({diff_ci[0]:.4f}, {diff_ci[1]:.4f})")

alpha = 0.05
print("\nDecision at alpha = 0.05:")
if p_value < alpha:
    print("Reject H0 — statistically significant difference in mean "
          "assists between forwards and midfielders.")
else:
    print("Fail to reject H0 — insufficient statistical evidence of a "
          "difference in mean assists between forwards and midfielders.")


pooled_sd = np.sqrt(((n_forward - 1) * forward_var +
                      (n_midfielder - 1) * midfielder_var) /
                     (n_forward + n_midfielder - 2))
cohens_d = mean_difference / pooled_sd
print(f"\nCohen's d (effect size): {cohens_d:.4f}")


