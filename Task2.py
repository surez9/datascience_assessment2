# S399630
# Task 1: Average Yellow Cards Comparison

# Q: Did defenders receive significantly more or fewer 
# yellow cards on average at the FIFA World Cup 2026 compared
#  with midfielders?

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


pd.set_option("display.max_columns", 100)

PLAYER_FILE = "dataset/player_stats.csv"
players = pd.read_csv(PLAYER_FILE)

players.shape, players.head(1)


# Data wrangling and preparation
eligible = players.loc[
    (players["matches_played"] > 0) &
    (players["position"].isin(["DEF", "MID"]))
].copy()

eligible["yellow_cards"] = eligible["yellow_cards"].fillna(0)
eligible["red_cards"] = eligible["red_cards"].fillna(0)

# Stratified sample: 60% DEF and 40% MID, approximately matching the eligible population
def_n = round(len(eligible[eligible.position == "DEF"]) / len(eligible) * 200)
mid_n = 200 - def_n

sample_def = eligible[eligible.position == "DEF"].sample(n=def_n, random_state=4174315)
sample_mid = eligible[eligible.position == "MID"].sample(n=mid_n, random_state=4174315)
sample = pd.concat([sample_def, sample_mid]).sample(frac=1, random_state=4174315)

print("Eligible population:")
print(eligible["position"].value_counts())
print("\nSample:")
print(sample["position"].value_counts())
print("\nSample size:", len(sample))


# Descriptive statistics
task1_desc = (
    sample.groupby("position")["yellow_cards"]
    .agg(n="count", mean="mean", median="median", std="std", min="min", max="max")
    .round(4)
)
task1_desc


# 95% confidence intervals for each group mean
def mean_ci(x, confidence=0.95):
    x = np.asarray(x, dtype=float)
    n = len(x)
    mean = x.mean()
    se = stats.sem(x)
    margin = stats.t.ppf((1 + confidence) / 2, n - 1) * se
    return mean - margin, mean + margin

for pos in ["DEF", "MID"]:
    x = sample.loc[sample.position == pos, "yellow_cards"]
    lo, hi = mean_ci(x)
    print(f"{pos}: mean={x.mean():.4f}, 95% CI=({lo:.4f}, {hi:.4f})")


    # Welch two-sample t-test
def_yc = sample.loc[sample.position == "DEF", "yellow_cards"]
mid_yc = sample.loc[sample.position == "MID", "yellow_cards"]

t_stat, p_value = stats.ttest_ind(def_yc, mid_yc, equal_var=False)

print(f"t-statistic = {t_stat:.4f}")
print(f"p-value = {p_value:.4f}")

if p_value < 0.05:
    print("Decision: Reject H0. The average yellow-card counts differ significantly.")
else:
    print("Decision: Fail to reject H0. There is no statistically significant difference in average yellow-card counts.")


"""
Using the reproducible stratified sample, defenders had a lower sample mean yellow-card count than midfielders.
the Welch two-sample t-test produces p > 0.05, so the 
difference is not statistically significant.

Therefore, the evidence does not support the claim that 
defenders received significantly more or fewer yellow cards 
than midfielders. The conclusion is based on the tournament 
player data after excluding players who never appeared.
"""

# Objective 2: Linear Regression

# Predict the goal difference between two opposing teams using 104 match rows and exactly eight explanatory variables.

# Public World Cup 2026 prediction dataset

X_URL = (
    "https://raw.githubusercontent.com/mominullptr/"
    "FIFA-World-Cup-2026-Dataset/main/"
    "match_prediction_features_X.csv"
)

Y_URL = (
    "https://raw.githubusercontent.com/mominullptr/"
    "FIFA-World-Cup-2026-Dataset/main/"
    "match_prediction_targets_y.csv"
)

X = pd.read_csv(X_URL)
y = pd.read_csv(Y_URL)

print("Feature matrix shape:", X.shape)
print("Target matrix shape:", y.shape)

print("\nFeature columns:")
print(X.columns.tolist())

print("\nTarget columns:")
print(y.columns.tolist())


# Research question¶
# Can eight pre-match team characteristics predict the goal difference between two opposing 
# FIFA World Cup 2026 teams?

 # Build the required 104-row match dataset

lr21_features = [
    "home_fifa_rank",
    "away_fifa_rank",
    "home_elo",
    "away_elo",
    "home_rest_days",
    "away_rest_days",
    "home_prev_avg_goals_scored",
    "away_prev_avg_goals_scored",
]

required_X21 = ["match_id"] + lr21_features
required_y21 = ["match_id", "home_score", "away_score"]

missing_X21 = [c for c in required_X21 if c not in X.columns]
missing_y21 = [c for c in required_y21 if c not in y.columns]

assert not missing_X21, f"Missing X columns: {missing_X21}"
assert not missing_y21, f"Missing y columns: {missing_y21}"

matches_lr = X[required_X21].merge(
    y[required_y21],
    on="match_id",
    how="inner",
    validate="one_to_one"
)

matches_lr["goal_difference"] = (
    matches_lr["home_score"] - matches_lr["away_score"]
)

matches_lr = matches_lr.dropna(
    subset=lr21_features + ["goal_difference"]
).copy()

print("Rows:", len(matches_lr))
print("Explanatory variables:", len(lr21_features))

# Assignment requirements
assert len(matches_lr) == 104, (
    f"Regression 2.1 requires exactly 104 rows, "
    f"but the prepared dataset has {len(matches_lr)}."
)
assert len(lr21_features) == 8
assert matches_lr["match_id"].nunique() == 104

matches_lr.head()


# Descriptive statistics

matches_lr[lr21_features + ["goal_difference"]].describe().T


# Define constant before function call
RANDOM_STATE = 42

# Train/test split
X21 = matches_lr[lr21_features]
y21 = matches_lr["goal_difference"]

X21_train, X21_test, y21_train, y21_test = train_test_split(
    X21,
    y21,
    test_size=0.20,
    random_state=RANDOM_STATE
)

model21 = LinearRegression()
model21.fit(X21_train, y21_train)

pred21 = model21.predict(X21_test)

r2_21 = r2_score(y21_test, pred21)
mae_21 = mean_absolute_error(y21_test, pred21)
rmse_21 = np.sqrt(mean_squared_error(y21_test, pred21))

print(f"Test R²: {r2_21:.4f}")
print(f"Test MAE: {mae_21:.4f}")
print(f"Test RMSE: {rmse_21:.4f}")

coef21 = pd.DataFrame({
    "variable": lr21_features,
    "coefficient": model21.coef_
}).sort_values(
    "coefficient",
    key=lambda s: s.abs(),
    ascending=False
)

coef21

# Actual versus predicted values

plt.figure(figsize=(7, 5))
plt.scatter(y21_test, pred21)
plt.axline((0, 0), slope=1)
plt.xlabel("Actual goal difference")
plt.ylabel("Predicted goal difference")
plt.title("Linear Regression 2.1: Actual vs Predicted")
plt.tight_layout()
plt.show()


# Residual diagnostic

residuals21 = y21_test - pred21

plt.figure(figsize=(7, 5))
plt.scatter(pred21, residuals21)
plt.axhline(0)
plt.xlabel("Predicted goal difference")
plt.ylabel("Residual")
plt.title("Linear Regression 2.1: Residual Plot")
plt.tight_layout()
plt.show()

print("Residual mean:", round(residuals21.mean(), 4))



# Linear Regression 2.2
# Research question
# Can eight pre-match team characteristics predict the number of goals scored by a team in a 
# single FIFA World Cup 2026 match?

# The 104 matches are converted into 208 team-match observations, with one row for each team's 
# perspective in each match.

# Construct 208 team-match observations from the 104 matches

score_lookup = y.set_index("match_id")[["home_score", "away_score"]]

home = X[[
    "match_id",
    "home_fifa_rank", "away_fifa_rank",
    "home_elo", "away_elo",
    "home_rest_days", "away_rest_days",
    "home_prev_avg_goals_scored", "away_prev_avg_goals_scored"
]].copy()

home["team_fifa_rank"] = home["home_fifa_rank"]
home["opponent_fifa_rank"] = home["away_fifa_rank"]
home["team_elo"] = home["home_elo"]
home["opponent_elo"] = home["away_elo"]
home["team_rest_days"] = home["home_rest_days"]
home["opponent_rest_days"] = home["away_rest_days"]
home["team_prev_avg_goals_scored"] = home["home_prev_avg_goals_scored"]
home["opponent_prev_avg_goals_scored"] = home["away_prev_avg_goals_scored"]
home["team_goals"] = home["match_id"].map(score_lookup["home_score"])
home["side"] = "Home"

away = X[[
    "match_id",
    "home_fifa_rank", "away_fifa_rank",
    "home_elo", "away_elo",
    "home_rest_days", "away_rest_days",
    "home_prev_avg_goals_scored", "away_prev_avg_goals_scored"
]].copy()

away["team_fifa_rank"] = away["away_fifa_rank"]
away["opponent_fifa_rank"] = away["home_fifa_rank"]
away["team_elo"] = away["away_elo"]
away["opponent_elo"] = away["home_elo"]
away["team_rest_days"] = away["away_rest_days"]
away["opponent_rest_days"] = away["home_rest_days"]
away["team_prev_avg_goals_scored"] = away["away_prev_avg_goals_scored"]
away["opponent_prev_avg_goals_scored"] = away["home_prev_avg_goals_scored"]
away["team_goals"] = away["match_id"].map(score_lookup["away_score"])
away["side"] = "Away"

lr22_features = [
    "team_fifa_rank",
    "opponent_fifa_rank",
    "team_elo",
    "opponent_elo",
    "team_rest_days",
    "opponent_rest_days",
    "team_prev_avg_goals_scored",
    "opponent_prev_avg_goals_scored",
]

lr22 = pd.concat([home, away], ignore_index=True)

lr22 = lr22[
    ["match_id", "side"] + lr22_features + ["team_goals"]
].dropna().copy()

print("Rows:", len(lr22))
print("Explanatory variables:", len(lr22_features))
print("Unique matches:", lr22["match_id"].nunique())


assert len(lr22) == 208, (
    f"Regression 2.2 requires exactly 208 rows, "
    f"but the prepared dataset has {len(lr22)}."
)
assert len(lr22_features) == 8
assert lr22["match_id"].nunique() == 104

# Every match must contribute exactly two team observations
obs_per_match = lr22.groupby("match_id").size()
assert obs_per_match.eq(2).all()

lr22.head()


# Descriptive statistics

lr22[lr22_features + ["team_goals"]].describe().T

# Match-level train/test split

unique_match_ids = lr22["match_id"].unique()

train_match_ids, test_match_ids = train_test_split(
    unique_match_ids,
    test_size=0.20,
    random_state=RANDOM_STATE
)

train22 = lr22[lr22["match_id"].isin(train_match_ids)].copy()
test22 = lr22[lr22["match_id"].isin(test_match_ids)].copy()

# Confirm that no match appears in both partitions
assert set(train22["match_id"]).isdisjoint(set(test22["match_id"]))

print("Training matches:", train22["match_id"].nunique())
print("Testing matches:", test22["match_id"].nunique())
print("Training team observations:", len(train22))
print("Testing team observations:", len(test22))

X22_train = train22[lr22_features]
y22_train = train22["team_goals"]

X22_test = test22[lr22_features]
y22_test = test22["team_goals"]

model22 = LinearRegression()
model22.fit(X22_train, y22_train)

pred22 = model22.predict(X22_test)

r2_22 = r2_score(y22_test, pred22)
mae_22 = mean_absolute_error(y22_test, pred22)
rmse_22 = np.sqrt(mean_squared_error(y22_test, pred22))

print(f"\nTest R²: {r2_22:.4f}")
print(f"Test MAE: {mae_22:.4f}")
print(f"Test RMSE: {rmse_22:.4f}")

coef22 = pd.DataFrame({
    "variable": lr22_features,
    "coefficient": model22.coef_
}).sort_values(
    "coefficient",
    key=lambda s: s.abs(),
    ascending=False
)

coef22


# Actual versus predicted values

plt.figure(figsize=(7, 5))
plt.scatter(y22_test, pred22)
plt.axline((0, 0), slope=1)
plt.xlabel("Actual team goals")
plt.ylabel("Predicted team goals")
plt.title("Linear Regression 2.2: Actual vs Predicted")
plt.tight_layout()
plt.show()

# Residual diagnostic

residuals22 = y22_test - pred22

plt.figure(figsize=(7, 5))
plt.scatter(pred22, residuals22)
plt.axhline(0)
plt.xlabel("Predicted team goals")
plt.ylabel("Residual")
plt.title("Linear Regression 2.2: Residual Plot")
plt.tight_layout()
plt.show()

print("Residual mean:", round(residuals22.mean(), 4))

"The target is a count of goals, so predictions can be fractional. A linear regression model"
"estimates the expected number of goals rather than restricting predictions to integer values."



import matplotlib.pyplot as plt

totals = eligible.groupby("position")["yellow_cards"].sum().reindex(["DEF", "MID"])

plt.bar(["Defenders", "Midfielders"], totals.values, color=["#4C72B0", "#DD8452"])
plt.ylabel("Total yellow cards")
plt.title("Total Yellow Cards: Defenders vs Midfielders (Full Population)")
plt.ylim(0, max(totals.values) + 5)
for i, v in enumerate(totals.values):
    plt.text(i, v + 0.3, str(v), ha="center", fontsize=14, fontweight="bold")
plt.savefig("total_yellow_cards_population.png", dpi=150)
plt.show()


