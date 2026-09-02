# 1. DATA LOADING

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

url = "https://raw.githubusercontent.com/Bustami/efi-fifa-data-wc-2026/refs/heads/master/data/wc2026_efi.csv"
player_match_df = pd.read_csv(url)

print("Dataset loaded. Shape:", player_match_df.shape)
display(player_match_df.head())

# 2. DATA INSPECTION
print("Unique players:", player_match_df["player_id"].nunique())
print("Unique matches:", player_match_df["match_id"].nunique())
print("Rows (player-match records):", len(player_match_df))

print("\nPosition categories:")
print(player_match_df["position"].value_counts())

print("\nMissing values in key columns:")
print(player_match_df[["player_name", "team_name", "position",
                        "goals", "assists"]].isnull().sum())

# 3. DATA CLEANING & AGGREGATION TO PLAYER LEVEL

player_df = (
    player_match_df
    .groupby(["player_id", "player_name", "team_name", "position"],
              as_index=False)
    .agg(
        goals=("goals", "sum"),
        assists=("assists", "sum"),
        matches_played=("match_id", "nunique")
    )
)

print("Player-level dataset shape:", player_df.shape)
print("\nPosition distribution (all players):")
print(player_df["position"].value_counts())

print("\nDuplicate player_id after aggregation:",
      player_df["player_id"].duplicated().sum())

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



