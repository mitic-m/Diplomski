import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np

name_map = {
    "Panathinaikos AKTOR": "Panathinaikos",
    "Partizan Mozzart Bet": "Partizan",
    "Maccabi Playtika Tel Aviv": "Maccabi Tel Aviv",
    "EA7 Emporio Armani Milano": "Milano",
    "Crvena zvezda Meridianbet": "Crvena zvezda",
    "Virtus Segafredo Bologna": "Virtus Bologna",
    "Fenerbahçe Beko": "Fenerbahce",
    "Žalgiris": "Zalgiris",
    "LDLC ASVEL": "LDLC Asvel",
    "Panathinaikos OPAP": "Panathinaikos",
    "Cazoo Baskonia": "Baskonia",
    "Bitci Baskonia": "Baskonia",
    "Maccabi Tel Aviv": "Maccabi Tel Aviv",
    "EA7 Emporio Armani Milan": "Milano",
    "AIX Armani Exchange Milan": "Milano",
    "Crvena zvezda mts": "Crvena zvezda",
    "Hapoel IBI Tel Aviv": "Hapoel Tel Aviv",
    "Alba Berlin": "ALBA Berlin",
    "CSKA Moscow": None,
    "UNICS": None,
    "Zenit Saint Petersburg": None
}

# ucitaj sve sezone BEZ 2022
games_all = pd.read_csv(r"D:\Diplomski\csv files\euroleague_games.csv")
games_all["HomeTeam"] = games_all["HomeTeam"].replace(name_map)
games_all["AwayTeam"] = games_all["AwayTeam"].replace(name_map)
games_all = games_all[games_all["HomeTeam"].notna() & games_all["AwayTeam"].notna()]
games_all = games_all[games_all["Season"] != 2022]

# distances
distances = pd.read_csv(r"D:\Diplomski\csv files\euroleague_distances.csv")
distances["Team1"] = distances["Team1"].replace(name_map)
distances["Team2"] = distances["Team2"].replace(name_map)

dist_dict = {}
for _, row in distances.iterrows():
    dist_dict[(row["Team1"], row["Team2"])] = row["Distance_km"]

max_distance = distances["Distance_km"].max()

def get_strength_dict(games_df, season):
    s = games_df[games_df["Season"] == season]
    home = s.groupby("HomeTeam").agg(
        HomePoints=("HomeScore", "sum"),
        HomeConceded=("AwayScore", "sum"),
        HomeGames=("HomeScore", "count")
    )
    away = s.groupby("AwayTeam").agg(
        AwayPoints=("AwayScore", "sum"),
        AwayConceded=("HomeScore", "sum"),
        AwayGames=("AwayScore", "count")
    )
    pd_df = home.join(away)
    pd_df["PointDiff"] = (pd_df["HomePoints"] + pd_df["AwayPoints"] -
                          pd_df["HomeConceded"] - pd_df["AwayConceded"]) / (
                          pd_df["HomeGames"] + pd_df["AwayGames"])
    return pd_df["PointDiff"].to_dict()

# koristimo samo 2023 i 2024 kao prethodne sezone
strength_by_season = {
    2024: get_strength_dict(games_all, 2023),
    2025: get_strength_dict(games_all, 2024)
}

rows = []

for _, game in games_all.iterrows():
    home_team = game["HomeTeam"]
    away_team = game["AwayTeam"]
    season = game["Season"]

    if season not in strength_by_season:
        continue

    strength_dict = strength_by_season[season]

    if home_team not in strength_dict or away_team not in strength_dict:
        continue

    if (away_team, home_team) not in dist_dict:
        continue

    strength_diff = strength_dict[home_team] - strength_dict[away_team]
    distance_norm = dist_dict[(away_team, home_team)] / max_distance

    rows.append({
        "strength_diff": strength_diff,
        "distance_norm": distance_norm,
        "home_win": game["HomeWin"]
    })

df_reg = pd.DataFrame(rows)

print(f"Utakmica za regresiju: {len(df_reg)}")

X = df_reg[["strength_diff", "distance_norm"]]
y = df_reg["home_win"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

print(f"\nKoeficijent strength_diff: {model.coef_[0][0]:.4f}")
print(f"Koeficijent distance_norm: {model.coef_[0][1]:.4f}")
print(f"Intercept: {model.intercept_[0]:.4f}")
print(f"Train accuracy: {model.score(X_train, y_train):.4f}")
print(f"Test accuracy: {model.score(X_test, y_test):.4f}")

p_home = 1 / (1 + np.exp(-model.intercept_[0]))
print(f"\nGlobalni home bonus: {p_home:.3f}")

print(f"\n# Koeficijenti za model_D.py:")
print(f"INTERCEPT = {model.intercept_[0]:.4f}")
print(f"COEF_STRENGTH = {model.coef_[0][0]:.4f}")
print(f"COEF_DISTANCE = {model.coef_[0][1]:.4f}")