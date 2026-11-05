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

# NET RTG podaci
net_rtg_data = {
    2023: {
        "ALBA Berlin": -5.9, "Anadolu Efes": 2.9, "AS Monaco": 2.0,
        "LDLC Asvel": -11.6, "Baskonia": 3.8, "Zalgiris": -2.2,
        "Barcelona": 6.6, "Bayern München": -5.9, "Fenerbahce": 3.3,
        "Crvena zvezda": -0.8, "Partizan": 4.4, "Maccabi Tel Aviv": 3.4,
        "Milano": -3.0, "Olympiacos": 12.2, "Panathinaikos": -5.6,
        "Real Madrid": 8.7, "Valencia Basket": -5.4, "Virtus Bologna": -6.6
    },
    2024: {
        "ALBA Berlin": -13.8, "Anadolu Efes": 1.2, "AS Monaco": 3.5,
        "LDLC Asvel": -8.8, "Baskonia": -1.2, "Zalgiris": 0.3,
        "Barcelona": 4.8, "Bayern München": -3.7, "Fenerbahce": 5.8,
        "Crvena zvezda": -2.7, "Partizan": -0.1, "Maccabi Tel Aviv": 1.5,
        "Milano": 0.1, "Olympiacos": 5.6, "Panathinaikos": 5.9,
        "Real Madrid": 9.8, "Valencia Basket": -4.2, "Virtus Bologna": -3.7
    },
    2025: {
        "Anadolu Efes": 7.1, "AS Monaco": 4.7,
        "LDLC Asvel": -5.3, "Baskonia": -2.0, "Zalgiris": -2.1,
        "Barcelona": 5.0, "Bayern München": -1.1, "Fenerbahce": 3.1,
        "Crvena zvezda": 2.0, "Partizan": 2.4, "Maccabi Tel Aviv": -5.5,
        "Milano": -1.0, "Olympiacos": 7.0, "Panathinaikos": 6.3,
        "Paris Basketball": 0.7, "Real Madrid": 3.6, "Virtus Bologna": -6.7,
        "Dubai Basketball": 0.0, "Hapoel Tel Aviv": 0.0, "Valencia Basket": 0.0
    }
}

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

# napravi parove za regresiju sa NET_RTG
rows = []
seasons = ["2023", "2024", "2025"]
for i in range(len(seasons) - 1):
    current = seasons[i]
    next_season = seasons[i + 1]
    current_int = int(current)

    for team, net_rtg in net_rtg_data[current_int].items():
        wr_next = winrates[winrates["Team"] == team][next_season]
        if len(wr_next) == 0 or pd.isna(wr_next.values[0]):
            continue
        rows.append({
            "Team": team,
            "NET_RTG": net_rtg,
            "WinRate_next": wr_next.values[0]
        })

df_reg = pd.DataFrame(rows)
print(f"Parova za regresiju: {len(df_reg)}")

X = df_reg[["NET_RTG"]]
y = df_reg["WinRate_next"]
model = LinearRegression()
model.fit(X, y)

print(f"Koeficijent NET_RTG: {model.coef_[0]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")
print(f"R²: {model.score(X, y):.4f}")

# predvidi winrate za 2026
predicted = []
for team, net_rtg in net_rtg_data[2025].items():
    wr_2026 = model.predict(pd.DataFrame([[net_rtg]], columns=["NET_RTG"]))[0]
    predicted.append({
        "Team": team,
        "NET_RTG_2025": net_rtg,
        "WinRate_2026": round(wr_2026, 4)
    })

predicted_df = pd.DataFrame(predicted)
predicted_df = predicted_df.sort_values("WinRate_2026", ascending=False).reset_index(drop=True)
predicted_df.index = predicted_df.index + 1

print("\nPredvidjeni winrate za sezonu 2026:")
print(predicted_df.to_string())

teams_2026 = predicted_df["Team"].tolist()
strengths_2026 = dict(zip(predicted_df["Team"], predicted_df["WinRate_2026"]))

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

# validacija
def validate(target_season, training_seasons):
    print(f"\n{'=' * 60}")
    print(f"VALIDACIJA: predvidjamo {target_season} na osnovu {training_seasons}")
    print(f"{'=' * 60}")

    rows_val = []
    for ts in training_seasons:
        if ts not in net_rtg_data:
            continue
        next_s = ts + 1
        for team, net_rtg in net_rtg_data[ts].items():
            wr_next = winrates[winrates["Team"] == team][str(next_s)]
            if len(wr_next) == 0 or pd.isna(wr_next.values[0]):
                continue
            rows_val.append({
                "Team": team,
                "NET_RTG": net_rtg,
                "WinRate_next": wr_next.values[0]
            })

    df_val = pd.DataFrame(rows_val)
    if len(df_val) == 0:
        print("Nema dovoljno podataka!")
        return

    model_val = LinearRegression()
    model_val.fit(df_val[["NET_RTG"]], df_val["WinRate_next"])

    # predvidi
    last_ts = training_seasons[-1]
    strengths_val = {}
    for team, net_rtg in net_rtg_data[last_ts].items():
        wr_pred = model_val.predict(
            pd.DataFrame([[net_rtg]], columns=["NET_RTG"]))[0]
        strengths_val[team] = wr_pred

    teams_val = list(strengths_val.keys())
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


validate(target_season=2024, training_seasons=[2023])
validate(target_season=2025, training_seasons=[2023, 2024])
validate(target_season=2026, training_seasons=[2023, 2024, 2025])