import random
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy.stats import spearmanr

# ucitavanje podataka
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


def get_avg_pd(games_df, team, seasons):
    """Prosecna kos razlika tima kroz vise sezona"""
    pd_wide = get_point_diffs(games_df, seasons)
    team_row = pd_wide[pd_wide["Team"] == team]
    if len(team_row) == 0:
        return 0.0

    total = 0
    count = 0
    for season in seasons:
        col = f"PD_{season}"
        if col in team_row.columns and pd.notna(team_row[col].values[0]):
            total += team_row[col].values[0]
            count += 1

    if count == 0:
        return 0.0
    return total / count


# izracunaj kos razliku za sve sezone
point_diffs_wide = get_point_diffs(games, [2022, 2023, 2024, 2025])

winrates_indexed = winrates.set_index("Team")
point_diffs_indexed = point_diffs_wide.set_index("Team")
combined = winrates_indexed.join(point_diffs_indexed, how="left").reset_index()

# napravi parove za regresiju koristeci prosek kos razlike
seasons = ["2022", "2023", "2024", "2025"]

rows = []
for i in range(len(seasons) - 1):
    current = seasons[i]
    next_season = seasons[i + 1]

    # prosek kos razlike do trenutne sezone
    available_seasons = [int(s) for s in seasons[:i + 1]]

    for _, row in combined.iterrows():
        if pd.notna(row[next_season]):
            team = row["Team"]
            avg_pd = get_avg_pd(games, team, available_seasons)

            rows.append({
                "Team": team,
                "PD_avg": avg_pd,
                "WinRate_next": row[next_season]
            })

df_reg = pd.DataFrame(rows)
df_reg = df_reg.dropna()

print(df_reg["Team"].value_counts())
print(f"\nUkupno parova: {len(df_reg)}")

print(f"Parova za regresiju: {len(df_reg)}")

# regresija
X = df_reg[["PD_avg"]]
y = df_reg["WinRate_next"]

model = LinearRegression()
model.fit(X, y)

print(f"Koeficijent PD: {model.coef_[0]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")
print(f"R²: {model.score(X, y):.4f}")

# predvidi winrate za 2026 koristeci prosek svih sezona
predicted = []
for _, row in combined.iterrows():
    team = row["Team"]
    avg_pd = get_avg_pd(games, team, [2022, 2023, 2024, 2025])

    wr_2026 = model.predict(
        pd.DataFrame([[avg_pd]], columns=["PD_avg"]))[0]

    predicted.append({
        "Team": team,
        "PD_avg": round(avg_pd, 4),
        "WinRate_2026_predicted": round(wr_2026, 4)
    })

predicted_df = pd.DataFrame(predicted)
predicted_df = predicted_df.sort_values("WinRate_2026_predicted", ascending=False).reset_index(drop=True)
predicted_df.index = predicted_df.index + 1

print("\nPredvidjeni winrate za sezonu 2026:")
print(predicted_df.to_string())

# simulacija
SIMULATIONS = 10000

teams = predicted_df["Team"].tolist()
strengths = dict(zip(predicted_df["Team"], predicted_df["WinRate_2026_predicted"]))

first_place = {team: 0 for team in teams}
top6 = {team: 0 for team in teams}
top10 = {team: 0 for team in teams}
total_wins = {team: 0 for team in teams}

for sim in range(SIMULATIONS):
    wins = {team: 0 for team in teams}

    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            team_i = teams[i]
            team_j = teams[j]

            s_i = strengths[team_i]
            s_j = strengths[team_j]

            p = s_i / (s_i + s_j)

            if random.random() < p:
                wins[team_i] += 1
            else:
                wins[team_j] += 1

            if random.random() < p:
                wins[team_i] += 1
            else:
                wins[team_j] += 1

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
for team in teams:
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


# validacija
def validate(target_season, training_seasons):
    print(f"\n{'=' * 60}")
    print(f"VALIDACIJA: predvidjamo {target_season} na osnovu {training_seasons}")
    print(f"{'=' * 60}")

    rows_val = []
    season_pairs = [(training_seasons[i], training_seasons[i + 1])
                    for i in range(len(training_seasons) - 1)]
    season_pairs.append((training_seasons[-1], target_season))

    for idx, (current, next_s) in enumerate(season_pairs):
        available = training_seasons[:idx + 1]
        for _, row in winrates.iterrows():
            team = row["Team"]
            wr_next = row.get(str(next_s))
            if pd.isna(wr_next):
                continue
            avg_pd = get_avg_pd(games, team, available)
            rows_val.append({
                "Team": team,
                "PD_avg": avg_pd,
                "WinRate_next": wr_next
            })

    df_val = pd.DataFrame(rows_val).dropna()

    X_val = df_val[["PD_avg"]]
    y_val = df_val["WinRate_next"]

    model_val = LinearRegression()
    model_val.fit(X_val, y_val)

    # predvidi koristeci prosek svih trening sezona
    predicted_val = []
    for _, row in winrates.iterrows():
        team = row["Team"]
        avg_pd = get_avg_pd(games, team, training_seasons)

        wr_pred = model_val.predict(
            pd.DataFrame([[avg_pd]], columns=["PD_avg"]))[0]
        predicted_val.append({"Team": team, "WinRate_predicted": wr_pred})

    pred_df = pd.DataFrame(predicted_val)
    pred_df = pred_df.sort_values("WinRate_predicted", ascending=False)
    pred_ranks = {team: pos + 1 for pos, team in enumerate(pred_df["Team"])}

    # stvarna tabela
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
    for pos, team in enumerate(pred_df["Team"]):
        actual_pos = actual_ranks.get(team, "N/A")
        actual_w = round(actual_table.get(team, 0))
        print(f"{pos + 1:<6} {team:<25} {actual_pos:<15} {actual_w}")

    print(f"\nSpearman korelacija: {corr:.3f}")
    print(f"P-value: {pvalue:.3f}")


validate(target_season=2023, training_seasons=[2022])
validate(target_season=2024, training_seasons=[2022, 2023])
validate(target_season=2025, training_seasons=[2022, 2023, 2024])