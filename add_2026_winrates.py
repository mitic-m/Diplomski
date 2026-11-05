import pandas as pd

winrates = pd.read_csv(r"D:\Diplomski\csv files\winrates.csv")

winrates_2026 = {
    "Olympiacos": round(26/38, 3),
    "Valencia Basket": round(25/38, 3),
    "Real Madrid": round(24/38, 3),
    "Fenerbahce": round(24/38, 3),
    "Zalgiris": round(23/38, 3),
    "Hapoel Tel Aviv": round(23/38, 3),
    "Panathinaikos": round(22/38, 3),
    "AS Monaco": round(22/38, 3),
    "Barcelona": round(21/38, 3),
    "Crvena zvezda": round(21/38, 3),
    "Dubai Basketball": round(19/38, 3),
    "Maccabi Tel Aviv": round(18/38, 3),
    "Bayern München": round(17/38, 3),
    "Milano": round(17/38, 3),
    "Partizan": round(16/38, 3),
    "Paris Basketball": round(15/38, 3),
    "Virtus Bologna": round(14/38, 3),
    "Baskonia": round(13/38, 3),
    "Anadolu Efes": round(12/38, 3),
    "LDLC Asvel": round(8/38, 3)
}

winrates["2026"] = winrates["Team"].map(winrates_2026)

print(winrates[["Team", "2025", "2026"]].to_string())

winrates.to_csv(r"D:\Diplomski\csv files\winrates.csv", index=False)
print("\nSacuvano!")