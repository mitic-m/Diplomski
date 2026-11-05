import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
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

point_diffs_wide = get_point_diffs(games, [2022, 2023, 2024, 2025])
winrates_indexed = winrates.set_index("Team")
point_diffs_indexed = point_diffs_wide.set_index("Team")
combined = winrates_indexed.join(point_diffs_indexed, how="left").reset_index()

# napravi parove sa OBA parametra
seasons = ["2022", "2023", "2024", "2025"]
rows = []
for i in range(len(seasons) - 1):
    current = seasons[i]
    next_season = seasons[i + 1]
    pd_current = f"PD_{current}"

    for _, row in combined.iterrows():
        if (pd.notna(row[current]) and
            pd.notna(row[next_season]) and
            pd.notna(row[pd_current])):
            rows.append({
                "Team": row["Team"],
                "WinRate_current": row[current],
                "PD_current": row[pd_current],
                "WinRate_next": row[next_season]
            })

df_reg = pd.DataFrame(rows)
print(f"Parova za regresiju: {len(df_reg)}")

X_both = df_reg[["WinRate_current", "PD_current"]]
y = df_reg["WinRate_next"]

# probaj razlicite alpha vrednosti
print("\nAlpha    Koef_WR    Koef_PD    Intercept    R²")
print("-" * 60)

alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_both, y)
    print(f"{alpha:<8} {ridge.coef_[0]:.4f}     {ridge.coef_[1]:.4f}     "
          f"{ridge.intercept_:.4f}       {ridge.score(X_both, y):.4f}")

# izaberi najbolji alpha i validuj
print("\n--- VALIDACIJA SA RIDGE REGRESIJOM ---")

best_alpha = 10.0  # promeniti nakon sto vidimo rezultate

ridge_model = Ridge(alpha=best_alpha)
ridge_model.fit(X_both, y)

print(f"\nKoeficijent WinRate: {ridge_model.coef_[0]:.4f}")
print(f"Koeficijent PD: {ridge_model.coef_[1]:.4f}")
print(f"Intercept: {ridge_model.intercept_:.4f}")
print(f"R²: {ridge_model.score(X_both, y):.4f}")

# predvidi za 2026
predicted = []
for _, row in combined.iterrows():
    team = row["Team"]
    wr_2025 = row.get("2025")
    pd_2025 = row["PD_2025"]

    wr_val = wr_2025 if pd.notna(wr_2025) else 0.5
    pd_val = pd_2025 if pd.notna(pd_2025) else 0.0

    wr_2026 = ridge_model.predict(
        pd.DataFrame([[wr_val, pd_val]], columns=["WinRate_current", "PD_current"]))[0]

    predicted.append({
        "Team": team,
        "WinRate_2025": wr_2025,
        "PD_2025": pd_2025,
        "WinRate_2026": round(wr_2026, 4)
    })

predicted_df = pd.DataFrame(predicted)
predicted_df = predicted_df.sort_values("WinRate_2026", ascending=False).reset_index(drop=True)
predicted_df.index = predicted_df.index + 1

print("\nPredvidjeni winrate za sezonu 2026:")
print(predicted_df.to_string())

# validacija Spearman
def validate_ridge(target_season, training_seasons, alpha):
    rows_val = []
    pd_wide = get_point_diffs(games, training_seasons)

    season_pairs = [(training_seasons[i], training_seasons[i+1])
                    for i in range(len(training_seasons)-1)]
    season_pairs.append((training_seasons[-1], target_season))

    for current, next_s in season_pairs:
        pd_col = f"PD_{current}"
        if pd_col not in pd_wide.columns:
            continue
        for _, row in pd_wide.iterrows():
            wr_curr = winrates[winrates["Team"] == row["Team"]][str(current)]
            wr_next = winrates[winrates["Team"] == row["Team"]][str(next_s)]
            if len(wr_next) == 0 or pd.isna(wr_next.values[0]):
                continue
            if len(wr_curr) == 0 or pd.isna(wr_curr.values[0]):
                continue
            if pd.isna(row[pd_col]):
                continue
            rows_val.append({
                "Team": row["Team"],
                "WinRate_current": wr_curr.values[0],
                "PD_current": row[pd_col],
                "WinRate_next": wr_next.values[0]
            })

    df_val = pd.DataFrame(rows_val)
    X_val = df_val[["WinRate_current", "PD_current"]]
    y_val = df_val["WinRate_next"]

    model_val = Ridge(alpha=alpha)
    model_val.fit(X_val, y_val)

    # predvidi
    predicted_val = []
    pd_last = f"PD_{training_seasons[-1]}"
    pd_wide_last = get_point_diffs(games, [training_seasons[-1]])

    for _, row in winrates.iterrows():
        team = row["Team"]
        wr_prev = row.get(str(training_seasons[-1]))
        team_pd = pd_wide_last[pd_wide_last["Team"] == team]

        wr_val = wr_prev if pd.notna(wr_prev) else 0.5
        pd_val = team_pd[pd_last].values[0] if len(team_pd) > 0 and pd.notna(team_pd[pd_last].values[0]) else 0.0

        wr_pred = model_val.predict(
            pd.DataFrame([[wr_val, pd_val]], columns=["WinRate_current", "PD_current"]))[0]
        predicted_val.append({"Team": team, "WinRate_predicted": wr_pred})

    pred_df = pd.DataFrame(predicted_val)
    pred_df = pred_df.sort_values("WinRate_predicted", ascending=False)
    pred_ranks = {team: pos+1 for pos, team in enumerate(pred_df["Team"])}

    season_games = games[games["Season"] == target_season]
    home_wins = season_games.groupby("HomeTeam")["HomeWin"].sum()
    away_wins = season_games.groupby("AwayTeam").apply(
        lambda x: (1 - x["HomeWin"]).sum())
    actual_table = home_wins.add(away_wins, fill_value=0).sort_values(ascending=False)
    actual_ranks = {team: pos+1 for pos, team in enumerate(actual_table.index)}

    common_teams = [t for t in pred_ranks if t in actual_ranks]
    pred_r = [pred_ranks[t] for t in common_teams]
    actual_r = [actual_ranks[t] for t in common_teams]

    corr, pvalue = spearmanr(pred_r, actual_r)
    return corr, pvalue

print(f"\n{'Sezona':<10} {'Spearman':<12} {'P-value'}")
print("-" * 35)
for target, training in [(2023, [2022]), (2024, [2022,2023]), (2025, [2022,2023,2024])]:
    corr, pvalue = validate_ridge(target, training, best_alpha)
    print(f"{target:<10} {corr:.3f}        {pvalue:.3f}")