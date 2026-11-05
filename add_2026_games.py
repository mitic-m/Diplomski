import pandas as pd

# ucitaj postojeci games.csv
games = pd.read_csv(r"D:\Diplomski\csv files\euroleague_games.csv")

# izbaci 2026 ako vec postoji
games = games[games["Season"] != 2026]
print(f"Games bez 2026: {len(games)} utakmica")

# ucitaj 2026 utakmice
games_2026 = pd.read_csv(r"D:\Diplomski\csv files\euroleague_2026_games.csv")

name_map = {
    "Panathinaikos AKTOR": "Panathinaikos",
    "Partizan Mozzart Bet": "Partizan",
    "EA7 Emporio Armani Milano": "Milano",
    "Crvena zvezda Meridianbet": "Crvena zvezda",
    "Virtus Olidata Bologna": "Virtus Bologna",
    "FenerbahÃ§e Beko": "Fenerbahce",
    "Å½algiris": "Zalgiris",
    "LDLC ASVEL": "LDLC Asvel",
    "Bayern MÃ¼nchen": "Bayern München",
    "Monaco": "AS Monaco",
    "Dubai": "Dubai Basketball",
    "Maccabi": "Maccabi Tel Aviv",
    "Paris": "Paris Basketball",
    "Valencia": "Valencia Basket",
    "Fenerbahce Beko": "Fenerbahce",
    "Zalgiris": "Zalgiris",
    "Bayern München": "Bayern München",
}

games_2026["home_team"] = games_2026["home_team"].replace(name_map)
games_2026["away_team"] = games_2026["away_team"].replace(name_map)

# konvertuj u isti format
games_2026_formatted = pd.DataFrame({
    "Date": games_2026["date"],
    "AwayTeam": games_2026["away_team"],
    "AwayScore": games_2026["away_pts"],
    "HomeTeam": games_2026["home_team"],
    "HomeScore": games_2026["home_pts"],
    "HomeWin": games_2026["home_win"],
    "Season": 2026
})

print(f"Formatiranih 2026 utakmica: {len(games_2026_formatted)}")

# spoji i sacuvaj
games_combined = pd.concat([games, games_2026_formatted], ignore_index=True)
print(f"Ukupno utakmica posle spajanja: {len(games_combined)}")
print(f"Sezone: {sorted(games_combined['Season'].unique())}")

# proveri timove u 2026
season_2026 = games_combined[games_combined["Season"] == 2026]
all_teams = set(season_2026["HomeTeam"].tolist() + season_2026["AwayTeam"].tolist())
print(f"\nTimovi u 2026 sezoni:")
print(sorted(all_teams))

games_combined.to_csv(r"D:\Diplomski\csv files\euroleague_games.csv", index=False)
print("\nSacuvano!")