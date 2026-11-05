import pandas as pd

stats_2023 = pd.read_csv(r"D:\Diplomski\csv files\euroleague_team_stats_2023.csv")
stats_2024 = pd.read_csv(r"D:\Diplomski\csv files\euroleague_team_stats_2024.csv")
stats_2025 = pd.read_csv(r"D:\Diplomski\csv files\euroleague_team_stats_2025.csv")

print("2023 kolone:")
print(stats_2023.columns.tolist())
print(f"Timova: {len(stats_2023)}")
print()
print(stats_2023[["Club", "3P%", "TRB", "TOV", "AST"]].head(5))

import numpy as np

# dodaj sezonu i broj pobeda
stats_2023["Season"] = 2023
stats_2024["Season"] = 2024
stats_2025["Season"] = 2025

# spoji sve sezone
all_stats = pd.concat([stats_2023, stats_2024, stats_2025], ignore_index=True)

# izvuci broj pobeda iz kolone W (ako postoji) ili G
print(all_stats.columns.tolist())
print(all_stats.head(3))

winrates = pd.read_csv(r"D:\Diplomski\csv files\winrates.csv")

# name_map za standardizaciju imena
name_map = {
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
    "Maccabi Playtika Tel Aviv": "Maccabi Tel Aviv",
    "EA7 Emporio Armani Milano": "Milano",
    "EA7 Emporio Armani Milan": "Milano",
    "Crvena zvezda Meridianbet": "Crvena zvezda",
    "Virtus Segafredo Bologna": "Virtus Bologna",
    "Žalgiris": "Zalgiris",
    "LDLC ASVEL": "LDLC Asvel",
    "ALBA Berlin": "ALBA Berlin",
    "Bayern München": "Bayern München",
    "Crvena zvezda Meridianbet *": "Crvena zvezda",
    "Real Madrid": "Real Madrid",
}

all_stats["Team"] = all_stats["Club"].replace(name_map)

# dodaj winrate po sezoni
def get_winrate(row):
    team = row["Team"]
    season = str(row["Season"])
    wr = winrates[winrates["Team"] == team][season]
    if len(wr) > 0 and pd.notna(wr.values[0]):
        return wr.values[0]
    return None

all_stats["WinRate"] = all_stats.apply(get_winrate, axis=1)

# per game statistike
all_stats["3P%_pg"] = all_stats["3P%"]
all_stats["TRB_pg"] = all_stats["TRB"] / all_stats["G"]
all_stats["TOV_pg"] = all_stats["TOV"] / all_stats["G"]
all_stats["AST_pg"] = all_stats["AST"] / all_stats["G"]
all_stats["STL_pg"] = all_stats["STL"] / all_stats["G"]

# korelacija sa winrate
stats_cols = ["3P%_pg", "TRB_pg", "TOV_pg", "AST_pg", "STL_pg", "FG%"]
df_clean = all_stats.dropna(subset=["WinRate"])

print("Korelacija sa WinRate:")
for col in stats_cols:
    corr = df_clean[col].corr(df_clean["WinRate"])
    print(f"  {col:<12} {corr:.3f}")


# dodaj PD iz games.csv
games = pd.read_csv(r"D:\Diplomski\csv files\euroleague_games.csv")

name_map_games = {
    "Panathinaikos AKTOR": "Panathinaikos",
    "Partizan Mozzart Bet": "Partizan",
    "Maccabi Playtika Tel Aviv": "Maccabi Tel Aviv",
    "EA7 Emporio Armani Milano": "Milano",
    "Crvena zvezda Meridianbet": "Crvena zvezda",
    "Virtus Segafredo Bologna": "Virtus Bologna",
    "Fenerbahçe Beko": "Fenerbahce",
    "Žalgiris": "Zalgiris",
    "LDLC ASVEL": "LDLC Asvel",
    "Hapoel IBI Tel Aviv": "Hapoel Tel Aviv",
    "CSKA Moscow": None,
    "UNICS": None,
    "Zenit Saint Petersburg": None
}

games["HomeTeam"] = games["HomeTeam"].replace(name_map_games)
games["AwayTeam"] = games["AwayTeam"].replace(name_map_games)
games = games[games["HomeTeam"].notna() & games["AwayTeam"].notna()]

# izracunaj PD po timu po sezoni
pd_rows = []
for season in [2023, 2024, 2025]:
    s = games[games["Season"] == season]
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
    df["PD"] = (df["HomePoints"] + df["AwayPoints"] -
                df["HomeConceded"] - df["AwayConceded"]) / (
                df["HomeGames"] + df["AwayGames"])
    df["Season"] = season
    df["Team"] = df.index
    pd_rows.append(df[["Team", "Season", "PD"]])

pd_df = pd.concat(pd_rows)

# spoji sa all_stats
all_stats = all_stats.merge(pd_df, on=["Team", "Season"], how="left")

# korelacija FG% i TOV sa PD
print(f"\nKorelacija FG% sa PD: {all_stats['FG%'].corr(all_stats['PD']):.3f}")
print(f"Korelacija TOV_pg sa PD: {all_stats['TOV_pg'].corr(all_stats['PD']):.3f}")
print(f"Korelacija AST_pg sa PD: {all_stats['AST_pg'].corr(all_stats['PD']):.3f}")