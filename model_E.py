import random
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy.stats import spearmanr

winrates = pd.read_csv(r"D:\Diplomski\csv files\winrates.csv")
games = pd.read_csv(r"D:\Diplomski\csv files\euroleague_games.csv")

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
    "CSKA Moscow": None,
    "UNICS": None,
    "Zenit Saint Petersburg": None
}

games["HomeTeam"] = games["HomeTeam"].replace(name_map)
games["AwayTeam"] = games["AwayTeam"].replace(name_map)
games = games[games["HomeTeam"].notna() & games["AwayTeam"].notna()]

INTERCEPT = 0.6790
COEF_STRENGTH = 0.1055
SIMULATIONS = 10000

def get_point_diffs(games_df, seasons):
    results = []
    for season in seasons:
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
        df = home.join(away)
        df["PointDiff"] = (df["HomePoints"] + df["AwayPoints"] -
                           df["HomeConceded"] - df["AwayConceded"]) / (
                           df["HomeGames"] + df["AwayGames"])
        df["Season"] = season
        df["Team"] = df.index
        results.append(df[["Team", "Season", "PointDiff"]])
    point_diffs = pd.concat(results)
    point_diffs_wide = point_diffs.pivot(index="Team", columns="Season", values="PointDiff")
    point_diffs_wide.columns = [f"PD_{col}" for col in point_diffs_wide.columns]
    return point_diffs_wide.reset_index()

def simulate_season(teams, strengths):
    wins = {team: 0 for team in teams}
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            team_i = teams[i]
            team_j = teams[j]
            s_i = strengths[team_i]
            s_j = strengths[team_j]

            log_odds = INTERCEPT + COEF_STRENGTH * (s_i - s_j)
            p = 1 / (1 + np.exp(-log_odds))
            if random.random() < p:
                wins[team_i] += 1
            else:
                wins[team_j] += 1

            log_odds = INTERCEPT + COEF_STRENGTH * (s_j - s_i)
            p = 1 / (1 + np.exp(-log_odds))
            if random.random() < p:
                wins[team_j] += 1
            else:
                wins[team_i] += 1
    return wins

point_diffs_wide = get_point_diffs(games, [2022, 2023, 2024, 2025])
winrates_indexed = winrates.set_index("Team")
point_diffs_indexed = point_diffs_wide.set_index("Team")
combined = winrates_indexed.join(point_diffs_indexed, how="left").reset_index()
combined = combined.drop_duplicates(subset="Team").reset_index(drop=True)

# linearna regresija
seasons = ["2022", "2023", "2024", "2025"]
rows = []
for i in range(len(seasons) - 1):
    current = seasons[i]
    next_season = seasons[i + 1]
    pd_current = f"PD_{current}"
    for _, row in combined.iterrows():
        if pd.notna(row[next_season]) and pd.notna(row[pd_current]):
            rows.append({
                "Team": row["Team"],
                "PD_current": row[pd_current],
                "WinRate_next": row[next_season]
            })

df_reg = pd.DataFrame(rows)
X = df_reg[["PD_current"]]
y = df_reg["WinRate_next"]
model = LinearRegression()
model.fit(X, y)

print(f"Parova za regresiju: {len(df_reg)}")
print(f"Koeficijent PD: {model.coef_[0]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")
print(f"R²: {model.score(X, y):.4f}")

# budzeti za 2025 sezonu
budgets_2025 = {
    "Real Madrid": 45, "Milano": 39, "Panathinaikos": 37,
    "Olympiacos": 37, "Anadolu Efes": 34.5, "Fenerbahce": 33.5,
    "Bayern München": 33, "Barcelona": 32, "AS Monaco": 29.4,
    "Virtus Bologna": 25, "Maccabi Tel Aviv": 22.5, "Partizan": 20,
    "Paris Basketball": 18, "Valencia Basket": 18, "Baskonia": 16,
    "Crvena zvezda": 15.8, "Zalgiris": 13, "LDLC Asvel": 12,
    "Dubai Basketball": 18.5, "Hapoel Tel Aviv": 20
}

# regresija budzet -> PD
budget_pd_rows = []
for _, row in combined.iterrows():
    team = row["Team"]
    pd_2025 = row["PD_2025"]
    if team in budgets_2025 and pd.notna(pd_2025):
        budget_pd_rows.append({
            "Team": team,
            "Budget": budgets_2025[team],
            "PD_2025": pd_2025
        })

df_budget = pd.DataFrame(budget_pd_rows)
print(f"\nTimova za budget->PD regresiju: {len(df_budget)}")

X_b = df_budget[["Budget"]]
y_b = df_budget["PD_2025"]
model_budget = LinearRegression()
model_budget.fit(X_b, y_b)

print(f"Koeficijent Budget: {model_budget.coef_[0]:.4f}")
print(f"Intercept: {model_budget.intercept_:.4f}")
print(f"R²: {model_budget.score(X_b, y_b):.4f}")

# predvidi PD za sve timove
predicted = []
for _, row in combined.iterrows():
    team = row["Team"]
    pd_2025 = row["PD_2025"]
    pd_2024 = row.get("PD_2024")
    pd_2023 = row.get("PD_2023")
    pd_2022 = row.get("PD_2022")

    if pd.notna(pd_2025):
        final_pd = pd_2025
        source = "PD_2025"
    elif pd.notna(pd_2024):
        final_pd = pd_2024
        source = "PD_2024"
    elif pd.notna(pd_2023):
        final_pd = pd_2023
        source = "PD_2023"
    elif pd.notna(pd_2022):
        final_pd = pd_2022
        source = "PD_2022"
    elif team in budgets_2025:
        final_pd = model_budget.predict(
            pd.DataFrame([[budgets_2025[team]]], columns=["Budget"]))[0]
        source = "Budget"
    else:
        final_pd = 0.0
        source = "Prosek"

    wr_2026 = model.predict(
        pd.DataFrame([[final_pd]], columns=["PD_current"]))[0]

    predicted.append({
        "Team": team,
        "PD_2025": pd_2025,
        "PD_final": round(final_pd, 4),
        "Source": source,
        "WinRate_2026_predicted": round(wr_2026, 4)
    })

predicted_df = pd.DataFrame(predicted)
predicted_df = predicted_df.sort_values("WinRate_2026_predicted", ascending=False).reset_index(drop=True)
predicted_df.index = predicted_df.index + 1

print("\nPredvidjeni winrate za sezonu 2026:")
print(predicted_df.to_string())

# simulacija
teams_2026 = predicted_df["Team"].tolist()
strengths_2026 = dict(zip(predicted_df["Team"], predicted_df["PD_final"]))

first_place = {team: 0 for team in teams_2026}
top6 = {team: 0 for team in teams_2026}
top10 = {team: 0 for team in teams_2026}
total_wins = {team: 0 for team in teams_2026}

for sim in range(SIMULATIONS):
    wins = simulate_season(teams_2026, strengths_2026)
    table = sorted(wins.items(), key=lambda x: x[1], reverse=True)
    for pos, (team, w) in enumerate(table):
        total_wins[team] += w
        if pos == 0:
            first_place[team] += 1
        if pos < 6:
            top6[team] += 1
        if pos < 10:
            top10[team] += 1

results = []
for team in teams_2026:
    results.append({
        "Team": team,
        "Avg_Wins": round(total_wins[team] / SIMULATIONS, 1),
        "1st_%": round(first_place[team] / SIMULATIONS * 100, 2),
        "Top6_%": round(top6[team] / SIMULATIONS * 100, 2),
        "Top10_%": round(top10[team] / SIMULATIONS * 100, 2)
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("1st_%", ascending=False).reset_index(drop=True)
results_df.index = results_df.index + 1

print("\nSimulacija sezone 2026:")
print(results_df.to_string())

# spearman za 2026
actual_2026 = {
    "Olympiacos": 1, "Valencia Basket": 2, "Real Madrid": 3,
    "Fenerbahce": 4, "Zalgiris": 5, "Hapoel Tel Aviv": 6,
    "Panathinaikos": 7, "AS Monaco": 8, "Barcelona": 9,
    "Crvena zvezda": 10, "Dubai Basketball": 11, "Maccabi Tel Aviv": 12,
    "Bayern München": 13, "Milano": 14, "Partizan": 15,
    "Paris Basketball": 16, "Virtus Bologna": 17, "Baskonia": 18,
    "Anadolu Efes": 19, "LDLC Asvel": 20
}

predicted_ranks = {team: pos for pos, team in enumerate(results_df["Team"], 1)}

common = [t for t in predicted_ranks if t in actual_2026]
pred_r = [predicted_ranks[t] for t in common]
actual_r = [actual_2026[t] for t in common]

corr, pvalue = spearmanr(pred_r, actual_r)

print(f"\nSpearman za 2026: {corr:.3f}")
print(f"P-value: {pvalue:.3f}")

print(combined[combined["Team"] == "Valencia Basket"][["Team", "PD_2022", "PD_2023", "PD_2024", "PD_2025"]])