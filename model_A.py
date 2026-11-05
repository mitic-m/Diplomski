import random

import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy.stats import spearmanr

winrates = pd.read_csv(r"D:\Diplomski\csv files\winrates.csv")

# napravi parove: prethodna sezona -> sledeca sezona
seasons = ["2022", "2023", "2024", "2025"]

rows = []

for i in range(len(seasons) - 1):
    current = seasons[i]
    next_season = seasons[i + 1]

    for _, row in winrates.iterrows():
        if pd.notna(row[current]) and pd.notna(row[next_season]):
            rows.append({
                "Team": row["Team"],
                "WinRate_current": row[current],
                "WinRate_next": row[next_season]
            })

df = pd.DataFrame(rows)

# regresija
X = df[["WinRate_current"]]
y = df["WinRate_next"]

model = LinearRegression()
model.fit(X, y)

print(f"Koeficijent: {model.coef_[0]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")
print(f"R²: {model.score(X, y):.4f}")

# predvidi winrate za 2026
predicted = []

for _, row in winrates.iterrows():
    team = row["Team"]
    wr_2025 = row["2025"]

    if pd.notna(wr_2025):
        wr_2026 = model.predict([[wr_2025]])[0]
    else:
        # nema podatke -> koristimo prosecni winrate 0.500
        wr_2026 = model.predict([[0.500]])[0]

    predicted.append({
        "Team": team,
        "WinRate_2025": wr_2025,
        "WinRate_2026_predicted": round(wr_2026, 4)
    })

predicted_df = pd.DataFrame(predicted)
predicted_df = predicted_df.sort_values("WinRate_2026_predicted", ascending=False).reset_index(drop=True)
predicted_df.index = predicted_df.index + 1

print("\nPredvidjeni winrate za sezonu 2026:")
print(predicted_df.to_string())


SIMULATIONS = 10000

teams = predicted_df["Team"].tolist()
strengths = dict(zip(predicted_df["Team"], predicted_df["WinRate_2026_predicted"]))

first_place = {team: 0 for team in teams}
top6 = {team: 0 for team in teams}
top4 = {team: 0 for team in teams}
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
        if pos < 4:
            top4[team] += 1
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
        "Top4_%": round(top4[team] / SIMULATIONS * 100, 2),
        "Top6_%": round(top6[team] / SIMULATIONS * 100, 2),
        "Top10_%": round(top10[team] / SIMULATIONS * 100, 2),
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("1st_%", ascending=False).reset_index(drop=True)
results_df.index = results_df.index + 1

print("\nSimulacija sezone 2026:")
print(results_df.to_string())

from scipy.stats import spearmanr


def validate(target_season, training_seasons):
    print(f"\n{'=' * 60}")
    print(f"VALIDACIJA: predvidjamo {target_season} na osnovu {training_seasons}")
    print(f"{'=' * 60}")

    # napravi parove samo iz trening sezona
    rows = []
    season_pairs = [(training_seasons[i], training_seasons[i + 1])
                    for i in range(len(training_seasons) - 1)]

    # dodaj i poslednja trening sezona -> target
    season_pairs.append((str(training_seasons[-1]), str(target_season)))

    for current, next_s in season_pairs:
        for _, row in winrates.iterrows():
            if pd.notna(row.get(str(current))) and pd.notna(row.get(str(next_s))):
                rows.append({
                    "Team": row["Team"],
                    "WinRate_current": row[str(current)],
                    "WinRate_next": row[str(next_s)]
                })

    df_val = pd.DataFrame(rows)

    # regresija
    X_val = df_val[["WinRate_current"]]
    y_val = df_val["WinRate_next"]

    model_val = LinearRegression()
    model_val.fit(X_val, y_val)

    # predvidi winrate za target sezonu
    predicted_val = []
    for _, row in winrates.iterrows():
        team = row["Team"]
        wr_prev = row.get(str(training_seasons[-1]))

        if pd.notna(wr_prev):
            wr_pred = model_val.predict(
                pd.DataFrame([[wr_prev]], columns=["WinRate_current"]))[0]
        else:
            wr_pred = model_val.predict(
                pd.DataFrame([[0.500]], columns=["WinRate_current"]))[0]

        predicted_val.append({"Team": team, "WinRate_predicted": wr_pred})

    pred_df = pd.DataFrame(predicted_val)
    pred_df = pred_df.sort_values("WinRate_predicted", ascending=False)
    pred_ranks = {team: pos + 1 for pos, team in enumerate(pred_df["Team"])}

    # stvarna tabela iz games.csv
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
        "Alba Berlin": "ALBA Berlin",
        "Hapoel IBI Tel Aviv": "Hapoel Tel Aviv"
    }

    games["HomeTeam"] = games["HomeTeam"].replace(name_map)
    games["AwayTeam"] = games["AwayTeam"].replace(name_map)

    season_games = games[games["Season"] == target_season]
    home_wins = season_games.groupby("HomeTeam")["HomeWin"].sum()
    away_wins = season_games.groupby("AwayTeam").apply(
        lambda x: (1 - x["HomeWin"]).sum())
    actual_table = home_wins.add(away_wins, fill_value=0).sort_values(ascending=False)
    actual_ranks = {team: pos + 1 for pos, team in enumerate(actual_table.index)}

    # spearman
    common_teams = [t for t in pred_ranks if t in actual_ranks]
    pred_r = [pred_ranks[t] for t in common_teams]
    actual_r = [actual_ranks[t] for t in common_teams]

    corr, pvalue = spearmanr(pred_r, actual_r)

    # ispis
    print(f"\n{'Mesto':<6} {'Tim':<20} {'Stvarno mesto':<15} {'Stvarne pobede'}")
    print("-" * 55)
    for pos, team in enumerate(pred_df["Team"]):
        actual_pos = actual_ranks.get(team, "N/A")
        actual_w = round(actual_table.get(team, 0))
        print(f"{pos + 1:<6} {team:<20} {actual_pos:<15} {actual_w}")

    print(f"\nSpearman korelacija: {corr:.3f}")
    print(f"P-value: {pvalue:.3f}")


# pokretanje validacija
validate(target_season=2023, training_seasons=[2022])
validate(target_season=2024, training_seasons=[2022, 2023])
validate(target_season=2025, training_seasons=[2022, 2023, 2024])
validate(target_season=2026, training_seasons=[2022, 2023, 2024, 2025])

print(f"Broj timova: {len(teams)}")
print(f"Utakmica po timu: {(len(teams)-1)*2}")
print(f"Ukupno utakmica: {len(teams)*(len(teams)-1)}")

print(f"\nProsek pobeda po timu: {sum(total_wins.values()) / SIMULATIONS / len(teams):.1f}")
print(f"Ocekivani prosek: {(len(teams)-1):.1f}")
