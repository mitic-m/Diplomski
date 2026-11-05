import pandas as pd
from geopy.distance import geodesic

teams = {
    "Real Madrid": (40.4531, -3.6883),
    "Barcelona": (41.3851, 2.1734),
    "Olympiacos": (37.9838, 23.7275),
    "Panathinaikos": (37.9838, 23.7275),
    "Fenerbahce": (41.0082, 28.9784),
    "Anadolu Efes": (41.0082, 28.9784),
    "Maccabi Tel Aviv": (32.0853, 34.7818),
    "Hapoel Tel Aviv": (32.0853, 34.7818),
    "AS Monaco": (43.7384, 7.4246),
    "Baskonia": (42.8467, -2.6716),
    "Virtus Bologna": (44.4949, 11.3426),
    "Milano": (45.4642, 9.1900),
    "Bayern München": (48.1351, 11.5820),
    "Zalgiris": (54.8985, 23.9036),
    "Crvena zvezda": (44.7866, 20.4489),
    "Partizan": (44.7866, 20.4489),
    "Valencia Basket": (39.4699, -0.3763),
    "LDLC Asvel": (45.7640, 4.8357),
    "Paris Basketball": (48.8566, 2.3522),
    "Dubai Basketball": (25.2048, 55.2708),
    "ALBA Berlin": (52.5200, 13.4050)
}

rows = []

for team1 in teams:
    for team2 in teams:
        coord1 = teams[team1]
        coord2 = teams[team2]
        distance = geodesic(coord1, coord2).km
        rows.append({
            "Team1": team1,
            "Team2": team2,
            "Distance_km": round(distance, 1)
        })

df = pd.DataFrame(rows)
df.to_csv(r"D:\Diplomski\csv files\euroleague_distances.csv", index=False)

print(f"Ukupno parova: {len(df)}")
print(df.head(10))