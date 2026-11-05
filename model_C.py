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

# predvidi winrate za 2026
predicted = []
for _, row in combined.iterrows():
    team = row["Team"]
    pd_2025 = row["PD_2025"]
    pd_val = pd_2025 if pd.notna(pd_2025) else 0.0

    wr_2026 = model.predict(
        pd.DataFrame([[pd_val]], columns=["PD_current"]))[0]

    predicted.append({
        "Team": team,
        "PD_2025": pd_2025,
        "WinRate_2026_predicted": round(wr_2026, 4)
    })

predicted_df = pd.DataFrame(predicted)
predicted_df = predicted_df.sort_values("WinRate_2026_predicted", ascending=False).reset_index(drop=True)
predicted_df.index = predicted_df.index + 1

print("\nPredvidjeni winrate za sezonu 2026:")
print(predicted_df.to_string())

# simulacija - strength je PD_2025 direktno
strengths_2026 = dict(zip(predicted_df["Team"], predicted_df["PD_2025"].fillna(0)))
teams_2026 = predicted_df["Team"].tolist()

first_place = {team: 0 for team in teams_2026}
top4 = {team: 0 for team in teams_2026}
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
        if pos < 4:
            top4[team] += 1
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
        "Top4_%": round(top4[team] / SIMULATIONS * 100, 2),
        "Top6_%": round(top6[team] / SIMULATIONS * 100, 2),
        "Top10_%": round(top10[team] / SIMULATIONS * 100, 2)
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("1st_%", ascending=False).reset_index(drop=True)
results_df.index = results_df.index + 1

print("\nSimulacija sezone 2026:")
print(results_df.to_string())


def validate(target_season, training_seasons):
    print(f"\n{'=' * 60}")
    print(f"VALIDACIJA: predvidjamo {target_season} na osnovu {training_seasons}")
    print(f"{'=' * 60}")

    rows_val = []
    season_pairs = [(training_seasons[i], training_seasons[i + 1])
                    for i in range(len(training_seasons) - 1)]
    season_pairs.append((training_seasons[-1], target_season))

    for current, next_s in season_pairs:
        pd_col = f"PD_{current}"
        if pd_col not in combined.columns:
            continue
        for _, row in combined.iterrows():
            wr_next = row.get(str(next_s))
            if pd.isna(wr_next) or pd.isna(row[pd_col]):
                continue
            rows_val.append({
                "Team": row["Team"],
                "PD_current": row[pd_col],
                "WinRate_next": wr_next
            })

    df_val = pd.DataFrame(rows_val)
    model_val = LinearRegression()
    model_val.fit(df_val[["PD_current"]], df_val["WinRate_next"])

    # strength je PD direktno
    pd_last = f"PD_{training_seasons[-1]}"
    teams_val = []
    strengths_val = {}

    for _, row in combined.iterrows():
        team = row["Team"]
        pd_val = row.get(pd_last)
        strengths_val[team] = pd_val if pd.notna(pd_val) else 0.0
        teams_val.append(team)

    total_wins_val = {team: 0 for team in teams_val}

    for sim in range(SIMULATIONS):
        wins = simulate_season(teams_val, strengths_val)
        for team, w in wins.items():
            total_wins_val[team] += w

    avg_wins_val = {team: total_wins_val[team] / SIMULATIONS for team in teams_val}
    pred_ranking = sorted(avg_wins_val.items(), key=lambda x: x[1], reverse=True)
    pred_ranks = {team: pos + 1 for pos, (team, _) in enumerate(pred_ranking)}

    season_games = games[games["Season"] == target_season]
    home_wins = season_games.groupby("HomeTeam")["HomeWin"].sum()
    away_wins = season_games.groupby("AwayTeam").apply(
        lambda x: (1 - x["HomeWin"]).sum())
    actual_table = home_wins.add(away_wins, fill_value=0).sort_values(ascending=False)
    actual_ranks = {team: pos + 1 for pos, team in enumerate(actual_table.index)}

    common_teams = [t for t in pred_ranks if t in actual_ranks]
    pred_r = [pred_ranks[t] for t in common_teams]
    actual_r = [actual_ranks[t] for t in common_teams]

    corr, pvalue = spearmanr(pred_r, actual_r)

    print(f"\n{'Mesto':<6} {'Tim':<25} {'Stvarno mesto':<15} {'Stvarne pobede'}")
    print("-" * 60)
    for pos, (team, _) in enumerate(pred_ranking):
        actual_pos = actual_ranks.get(team, "N/A")
        actual_w = round(actual_table.get(team, 0))
        print(f"{pos + 1:<6} {team:<25} {actual_pos:<15} {actual_w}")

    print(f"\nSpearman korelacija: {corr:.3f}")
    print(f"P-value: {pvalue:.3f}")


validate(target_season=2023, training_seasons=[2022])
validate(target_season=2024, training_seasons=[2022, 2023])
validate(target_season=2025, training_seasons=[2022, 2023, 2024])
validate(target_season=2026, training_seasons=[2022, 2023, 2024, 2025])