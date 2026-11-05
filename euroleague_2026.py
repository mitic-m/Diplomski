import pandas as pd
import numpy as np
import random
from scipy.stats import spearmanr

# ucitaj utakmice 2025/26
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
    "Hapoel Tel Aviv": "Hapoel Tel Aviv",
    "Bayern MÃ¼nchen": "Bayern München",
    "Monaco": "AS Monaco",
    "Dubai": "Dubai Basketball",
    "Maccabi": "Maccabi Tel Aviv",
    "Paris": "Paris Basketball",
    "Valencia": "Valencia Basket",
    "Barcelona": "Barcelona",
    "Real Madrid": "Real Madrid",
    "Olympiacos": "Olympiacos",
    "Baskonia": "Baskonia",
    "Anadolu Efes": "Anadolu Efes"
}

games_2026["home_team"] = games_2026["home_team"].replace(name_map)
games_2026["away_team"] = games_2026["away_team"].replace(name_map)

# konvertuj datum
games_2026["date"] = pd.to_datetime(games_2026["date"], format="%a, %b %d, %Y")

# podeli na prvu i drugu polovinu po datumu
midpoint = pd.Timestamp("2026-01-15")

first_half = games_2026[games_2026["date"] <= midpoint]
second_half = games_2026[games_2026["date"] > midpoint]

print(f"Utakmica u prvoj polovini: {len(first_half)}")
print(f"Utakmica u drugoj polovini: {len(second_half)}")

# izracunaj snagu timova iz prve polovine
def get_strengths(games_df):
    teams = set(games_df["home_team"].tolist() + games_df["away_team"].tolist())
    stats = {team: {"pts_for": 0, "pts_against": 0, "games": 0, "wins": 0} for team in teams}

    for _, row in games_df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        stats[home]["pts_for"] += row["home_pts"]
        stats[home]["pts_against"] += row["away_pts"]
        stats[home]["games"] += 1
        stats[home]["wins"] += row["home_win"]

        stats[away]["pts_for"] += row["away_pts"]
        stats[away]["pts_against"] += row["home_pts"]
        stats[away]["games"] += 1
        stats[away]["wins"] += (1 - row["home_win"])

    strengths = {}
    for team, s in stats.items():
        if s["games"] > 0:
            strengths[team] = (s["pts_for"] - s["pts_against"]) / s["games"]

    return strengths

strengths = get_strengths(first_half)

print("\nSnage timova iz prve polovine (PD):")
for team, pd_val in sorted(strengths.items(), key=lambda x: x[1], reverse=True):
    print(f"  {team:<25} {pd_val:.2f}")

# pobede iz prve polovine
def get_wins(games_df):
    wins = {}
    for _, row in games_df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        wins[home] = wins.get(home, 0) + row["home_win"]
        wins[away] = wins.get(away, 0) + (1 - row["home_win"])
    return wins

actual_first_half_wins = get_wins(first_half)

# simuliraj drugu polovinu
INTERCEPT = 0.6790
COEF_STRENGTH = 0.1055
SIMULATIONS = 10000

teams = list(strengths.keys())

# utakmice koje treba da se odigraju u drugoj polovini
remaining_games = []
for _, row in second_half.iterrows():
    remaining_games.append((row["home_team"], row["away_team"]))

print(f"\nPreostalih utakmica za simulaciju: {len(remaining_games)}")

total_wins = {team: 0 for team in teams}

for sim in range(SIMULATIONS):
    sim_wins = {team: actual_first_half_wins.get(team, 0) for team in teams}

    for home_team, away_team in remaining_games:
        if home_team not in strengths or away_team not in strengths:
            continue

        s_i = strengths[home_team]
        s_j = strengths[away_team]

        log_odds = INTERCEPT + COEF_STRENGTH * (s_i - s_j)
        p = 1 / (1 + np.exp(-log_odds))

        if random.random() < p:
            sim_wins[home_team] = sim_wins.get(home_team, 0) + 1
        else:
            sim_wins[away_team] = sim_wins.get(away_team, 0) + 1

    for team in teams:
        total_wins[team] += sim_wins.get(team, 0)

avg_wins = {team: total_wins[team] / SIMULATIONS for team in teams}
pred_ranking = sorted(avg_wins.items(), key=lambda x: x[1], reverse=True)

print("\nPredvidjeni konacni plasman:")
print(f"{'Mesto':<6} {'Tim':<25} {'Predv. pobede':<15} {'Stvarne pobede':<15} {'Stvarno mesto'}")
print("-" * 75)

actual_2026 = {
    "Olympiacos": (26, 1),
    "Valencia Basket": (25, 2),
    "Real Madrid": (24, 3),
    "Fenerbahce": (24, 4),
    "Zalgiris": (23, 5),
    "Hapoel Tel Aviv": (23, 6),
    "Panathinaikos": (22, 7),
    "AS Monaco": (22, 8),
    "Barcelona": (21, 9),
    "Crvena zvezda": (21, 10),
    "Dubai Basketball": (19, 11),
    "Maccabi Tel Aviv": (18, 12),
    "Bayern München": (17, 13),
    "Milano": (17, 14),
    "Partizan": (16, 15),
    "Paris Basketball": (15, 16),
    "Virtus Bologna": (14, 17),
    "Baskonia": (13, 18),
    "Anadolu Efes": (12, 19),
    "LDLC Asvel": (8, 20)
}

pred_ranks = {}
for pos, (team, wins) in enumerate(pred_ranking):
    actual_wins, actual_pos = actual_2026.get(team, (0, "N/A"))
    print(f"{pos+1:<6} {team:<25} {wins:.1f}          {actual_wins:<15} {actual_pos}")
    pred_ranks[team] = pos + 1

# spearman
common = [t for t in pred_ranks if t in actual_2026]
pred_r = [pred_ranks[t] for t in common]
actual_r = [actual_2026[t][1] for t in common]

corr, pvalue = spearmanr(pred_r, actual_r)

print(f"\nSpearman korelacija: {corr:.3f}")
print(f"P-value: {pvalue:.3f}")

print("\nPoređenje predviđenih i stvarnih mesta:")
print(f"{'Mesto pred.':<12} {'Tim':<25} {'Mesto stvarno':<15} {'Razlika'}")
print("-" * 65)

for pos, (team, wins) in enumerate(pred_ranking):
    actual_wins, actual_pos = actual_2026.get(team, (0, "N/A"))
    if actual_pos != "N/A":
        razlika = pos + 1 - actual_pos
        znak = "+" if razlika > 0 else ""
        print(f"{pos+1:<12} {team:<25} {actual_pos:<15} {znak}{razlika}")
    else:
        print(f"{pos+1:<12} {team:<25} {'N/A':<15} {'N/A'}")