import random
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy.stats import spearmanr

winrates = pd.read_csv(r"D:\Diplomski\csv files\winrates.csv")
games = pd.read_csv(r"D:\Diplomski\csv files\euroleague_games.csv")
stats_2023 = pd.read_csv(r"D:\Diplomski\csv files\euroleague_team_stats_2023.csv")
stats_2024 = pd.read_csv(r"D:\Diplomski\csv files\euroleague_team_stats_2024.csv")
stats_2025 = pd.read_csv(r"D:\Diplomski\csv files\euroleague_team_stats_2025.csv")

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
    "Zenit Saint Petersburg": None,
    "Real Madrid *": "Real Madrid",
    "Olympiacos *": "Olympiacos",
    "AS Monaco *": "AS Monaco",
    "Panathinaikos AKTOR *": "Panathinaikos",
    "Fenerbahçe Beko *": "Fenerbahce",
    "Barcelona *": "Barcelona",
    "Anadolu Efes *": "Anadolu Efes",
    "Paris Basketball *": "Paris Basketball",
    "Partizan Mozzart Bet *": "Partizan",
    "Maccabi Tel Aviv *": "Maccabi Tel Aviv",
    "EA7 Emporio Armani Milano *": "Milano",
    "Crvena zvezda Meridianbet *": "Crvena zvezda",
    "Virtus Segafredo Bologna *": "Virtus Bologna",
    "Žalgiris *": "Zalgiris",
    "LDLC ASVEL *": "LDLC Asvel",
    "ALBA Berlin": "ALBA Berlin",
    "Bayern München": "Bayern München",
    "Real Madrid": "Real Madrid",
    "Olympiacos": "Olympiacos",
    "Baskonia *": "Baskonia",
    "Maccabi Playtika Tel Aviv *": "Maccabi Tel Aviv",
    "Panathinaikos *": "Panathinaikos",
    "Barça *": "Barcelona",
    "Alba Berlin": "ALBA Berlin",
    "Maccabi Playtika Tel Aviv": "Maccabi Tel Aviv",
    "Valencia Basket *": "Valencia Basket"
}

games["HomeTeam"] = games["HomeTeam"].replace(name_map)
games["AwayTeam"] = games["AwayTeam"].replace(name_map)
games = games[games["HomeTeam"].notna() & games["AwayTeam"].notna()]

INTERCEPT = 0.6790
COEF_STRENGTH = 0.1055
SIMULATIONS = 10000

# standardizuj imena u stats fajlovima
def standardize_stats(df, season):
    df = df.copy()
    df["Season"] = season
    df["Team"] = df["Club"].replace(name_map)
    df["TOV_pg"] = df["TOV"] / df["G"]
    df["AST_pg"] = df["AST"] / df["G"]
    df["TRB_pg"] = df["TRB"] / df["G"]
    return df[["Team", "Season", "FG%", "3P%", "TOV_pg", "AST_pg", "TRB_pg"]]

stats_all = pd.concat([
    standardize_stats(stats_2023, 2023),
    standardize_stats(stats_2024, 2024),
    standardize_stats(stats_2025, 2025)
], ignore_index=True)

print("Standardizovane statistike:")
print(stats_all.head(5).to_string())

# izracunaj PD po timu po sezoni
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

# spoji sa statistikama
combined = combined.merge(
    stats_all.rename(columns={
        "FG%": "FG_pct", "3P%": "3P_pct",
        "TOV_pg": "TOV_pg", "AST_pg": "AST_pg"
    }),
    left_on=["Team"],
    right_on=["Team"],
    how="left"
)

print(f"\nKombinована tabela — prvih 5:")
print(combined[["Team", "Season", "PD_2025", "FG_pct", "TOV_pg"]].head(10).to_string())

# napravi parove za regresiju sa FG% i TOV
seasons = ["2022", "2023", "2024", "2025"]
rows = []
for i in range(len(seasons) - 1):
    current = seasons[i]
    next_season = seasons[i + 1]
    pd_current = f"PD_{current}"
    current_int = int(current)

    for _, row in combined[combined["Season"] == current_int].iterrows():
        wr_next = winrates[winrates["Team"] == row["Team"]][next_season]
        if len(wr_next) == 0 or pd.isna(wr_next.values[0]):
            continue
        if pd.isna(row[pd_current]):
            continue
        if pd.isna(row["FG_pct"]) or pd.isna(row["TOV_pg"]):
            continue

        rows.append({
            "Team": row["Team"],
            "PD_current": row[pd_current],
            "FG_pct": row["FG_pct"],
            "TOV_pg": row["TOV_pg"],
            "WinRate_next": wr_next.values[0]
        })

df_reg = pd.DataFrame(rows)
print(f"\nParova za regresiju: {len(df_reg)}")

X = df_reg[["PD_current", "FG_pct", "TOV_pg"]]
y = df_reg["WinRate_next"]
model = LinearRegression()
model.fit(X, y)

print(f"Koeficijent PD: {model.coef_[0]:.4f}")
print(f"Koeficijent FG%: {model.coef_[1]:.4f}")
print(f"Koeficijent TOV: {model.coef_[2]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")
print(f"R²: {model.score(X, y):.4f}")

# validacija
def validate(target_season, training_seasons):
    print(f"\n{'=' * 60}")
    print(f"VALIDACIJA: predvidjamo {target_season} na osnovu {training_seasons}")
    print(f"{'=' * 60}")

    rows_val = []
    for ts in training_seasons:
        ts_str = str(ts)
        next_s = str(ts + 1)
        pd_col = f"PD_{ts_str}"
        if pd_col not in combined.columns:
            continue

        season_data = combined[combined["Season"] == ts]
        for _, row in season_data.iterrows():
            wr_next = winrates[winrates["Team"] == row["Team"]][next_s]
            if len(wr_next) == 0 or pd.isna(wr_next.values[0]):
                continue
            if pd.isna(row[pd_col]):
                continue
            if pd.isna(row["FG_pct"]) or pd.isna(row["TOV_pg"]):
                continue
            rows_val.append({
                "Team": row["Team"],
                "PD_current": row[pd_col],
                "FG_pct": row["FG_pct"],
                "TOV_pg": row["TOV_pg"],
                "WinRate_next": wr_next.values[0]
            })

    # dodaj poslednji trening -> target
    last_ts = training_seasons[-1]
    pd_col = f"PD_{last_ts}"
    season_data = combined[combined["Season"] == last_ts]
    for _, row in season_data.iterrows():
        wr_next = winrates[winrates["Team"] == row["Team"]][str(target_season)]
        if len(wr_next) == 0 or pd.isna(wr_next.values[0]):
            continue
        if pd.isna(row[pd_col]):
            continue
        if pd.isna(row["FG_pct"]) or pd.isna(row["TOV_pg"]):
            continue
        rows_val.append({
            "Team": row["Team"],
            "PD_current": row[pd_col],
            "FG_pct": row["FG_pct"],
            "TOV_pg": row["TOV_pg"],
            "WinRate_next": wr_next.values[0]
        })

    df_val = pd.DataFrame(rows_val)
    if len(df_val) == 0:
        print("Nema dovoljno podataka!")
        return

    model_val = LinearRegression()
    model_val.fit(df_val[["PD_current", "FG_pct", "TOV_pg"]], df_val["WinRate_next"])

    # predvidi
    last_season_data = combined[combined["Season"] == last_ts]
    strengths_val = {}
    for _, row in last_season_data.iterrows():
        team = row["Team"]
        pd_val = row.get(f"PD_{last_ts}") if pd.notna(row.get(f"PD_{last_ts}")) else 0.0
        fg_val = row["FG_pct"] if pd.notna(row["FG_pct"]) else combined["FG_pct"].mean()
        tov_val = row["TOV_pg"] if pd.notna(row["TOV_pg"]) else combined["TOV_pg"].mean()

        wr_pred = model_val.predict(
            pd.DataFrame([[pd_val, fg_val, tov_val]],
                         columns=["PD_current", "FG_pct", "TOV_pg"]))[0]
        strengths_val[team] = wr_pred

    print("\nTimovi u validaciji 2025:")
    print(sorted(strengths_val.keys()))
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

# proveri koji timovi nemaju statistike za 2025
teams_2026 = winrates["Team"].tolist()
stats_2025_teams = standardize_stats(stats_2025, 2025)["Team"].tolist()

print("\nTimovi bez statistika za 2025:")
for team in teams_2026:
    if team not in stats_2025_teams:
        print(f"  {team}")

# predvidi 2026
avg_fg = stats_all[stats_all["Season"] == 2025]["FG%"].mean()
avg_tov = stats_all[stats_all["Season"] == 2025]["TOV_pg"].mean()

predicted = []
for _, row in winrates.iterrows():
    team = row["Team"]

    # uzmi PD iz 2025
    pd_row = point_diffs_wide[point_diffs_wide["Team"] == team]
    pd_val = pd_row["PD_2025"].values[0] if len(pd_row) > 0 and pd.notna(pd_row["PD_2025"].values[0]) else 0.0

    # uzmi statistike iz 2025
    stats_row = stats_all[(stats_all["Team"] == team) & (stats_all["Season"] == 2025)]
    if len(stats_row) > 0:
        fg_val = stats_row["FG%"].values[0]
        tov_val = stats_row["TOV_pg"].values[0]
        source = "Stats"
    else:
        fg_val = avg_fg
        tov_val = avg_tov
        source = "Prosek"

    wr_2026 = model.predict(
        pd.DataFrame([[pd_val, fg_val, tov_val]],
                     columns=["PD_current", "FG_pct", "TOV_pg"]))[0]

    predicted.append({
        "Team": team,
        "PD_2025": round(pd_val, 4),
        "FG%_2025": round(fg_val, 3),
        "TOV_2025": round(tov_val, 2),
        "Source": source,
        "WinRate_2026": round(wr_2026, 4)
    })

predicted_df = pd.DataFrame(predicted)
predicted_df = predicted_df.sort_values("WinRate_2026", ascending=False).reset_index(drop=True)
predicted_df.index = predicted_df.index + 1

print("\nPredvidjeni winrate za sezonu 2026:")
print(predicted_df.to_string())

print("\nTimovi u stats_all za 2025:")
print(sorted(stats_all[stats_all["Season"] == 2025]["Team"].tolist()))

print("\nTimovi u winrates:")
print(sorted(winrates["Team"].tolist()))

print("\nTimovi u stats_all za 2024:")
print(sorted(stats_all[stats_all["Season"] == 2024]["Team"].tolist()))

print("\nTimovi u stats_all za 2023:")
print(sorted(stats_all[stats_all["Season"] == 2023]["Team"].tolist()))