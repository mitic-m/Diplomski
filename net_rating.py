import pandas as pd

# parsiraj podatke
lines = """1
ALBA Berlin
ALBA Berlin
34	5	29	6,875	2,673	768	1,406	54.6%	246	818	30.1%	1,014	2,224	45.6%	399	549	72.7%	398	772	1,170	669	526	219	70	742	645	2,697	-431	2,585	75.2	103.4	120.7	-17.3	62.2%	54.4%	17.1%	1.51	21.1%	0.89	8.9%	0.78	2.7%	1.06	24.3%	0.90	7.8%	17.6%	9.9%	7.7%	54.1%	52.7%	76.4%	34.1%	22.4%	32.0%	34.1%	31.5%	18.5%	68.0%	64.7%	70.7%	73.6%	34.0%	8.5%	3.2%	5.4%	0.4%	131"""

# ekstraktuj rucno
teams_data = {
    "ALBA Berlin": {"OFF_RTG": 103.4, "DEF_RTG": 120.7, "NET_RTG": -17.3},
    "Anadolu Efes": {"OFF_RTG": 121.5, "DEF_RTG": 114.4, "NET_RTG": 7.1},
    "AS Monaco": {"OFF_RTG": 118.3, "DEF_RTG": 113.6, "NET_RTG": 4.7},
    "LDLC Asvel": {"OFF_RTG": 112.0, "DEF_RTG": 117.3, "NET_RTG": -5.3},
    "Baskonia": {"OFF_RTG": 115.0, "DEF_RTG": 117.0, "NET_RTG": -2.0},
    "Zalgiris": {"OFF_RTG": 110.0, "DEF_RTG": 112.1, "NET_RTG": -2.1},
    "Barcelona": {"OFF_RTG": 119.2, "DEF_RTG": 114.2, "NET_RTG": 5.0},
    "Bayern München": {"OFF_RTG": 119.2, "DEF_RTG": 120.3, "NET_RTG": -1.1},
    "Fenerbahce": {"OFF_RTG": 116.6, "DEF_RTG": 113.5, "NET_RTG": 3.1},
    "Crvena zvezda": {"OFF_RTG": 114.8, "DEF_RTG": 112.8, "NET_RTG": 2.0},
    "Partizan": {"OFF_RTG": 117.2, "DEF_RTG": 114.8, "NET_RTG": 2.4},
    "Maccabi Tel Aviv": {"OFF_RTG": 115.9, "DEF_RTG": 121.4, "NET_RTG": -5.5},
    "Milano": {"OFF_RTG": 117.3, "DEF_RTG": 118.3, "NET_RTG": -1.0},
    "Olympiacos": {"OFF_RTG": 121.4, "DEF_RTG": 114.4, "NET_RTG": 7.0},
    "Panathinaikos": {"OFF_RTG": 122.1, "DEF_RTG": 115.8, "NET_RTG": 6.3},
    "Paris Basketball": {"OFF_RTG": 117.2, "DEF_RTG": 116.5, "NET_RTG": 0.7},
    "Real Madrid": {"OFF_RTG": 115.1, "DEF_RTG": 111.5, "NET_RTG": 3.6},
    "Virtus Bologna": {"OFF_RTG": 110.1, "DEF_RTG": 116.8, "NET_RTG": -6.7},
}

teams_2024 = {
    "ALBA Berlin": {"OFF_RTG": 104.2, "DEF_RTG": 118.0, "NET_RTG": -13.8},
    "Anadolu Efes": {"OFF_RTG": 119.4, "DEF_RTG": 118.2, "NET_RTG": 1.2},
    "AS Monaco": {"OFF_RTG": 115.5, "DEF_RTG": 112.0, "NET_RTG": 3.5},
    "LDLC Asvel": {"OFF_RTG": 109.0, "DEF_RTG": 117.8, "NET_RTG": -8.8},
    "Baskonia": {"OFF_RTG": 114.9, "DEF_RTG": 116.1, "NET_RTG": -1.2},
    "Zalgiris": {"OFF_RTG": 113.5, "DEF_RTG": 113.2, "NET_RTG": 0.3},
    "Barcelona": {"OFF_RTG": 114.9, "DEF_RTG": 110.1, "NET_RTG": 4.8},
    "Bayern München": {"OFF_RTG": 110.1, "DEF_RTG": 113.8, "NET_RTG": -3.7},
    "Fenerbahce": {"OFF_RTG": 118.9, "DEF_RTG": 113.1, "NET_RTG": 5.8},
    "Crvena zvezda": {"OFF_RTG": 113.6, "DEF_RTG": 116.3, "NET_RTG": -2.7},
    "Partizan": {"OFF_RTG": 117.1, "DEF_RTG": 117.2, "NET_RTG": -0.1},
    "Maccabi Tel Aviv": {"OFF_RTG": 117.1, "DEF_RTG": 115.6, "NET_RTG": 1.5},
    "Milano": {"OFF_RTG": 111.3, "DEF_RTG": 111.2, "NET_RTG": 0.1},
    "Olympiacos": {"OFF_RTG": 115.3, "DEF_RTG": 109.7, "NET_RTG": 5.6},
    "Panathinaikos": {"OFF_RTG": 114.3, "DEF_RTG": 108.4, "NET_RTG": 5.9},
    "Real Madrid": {"OFF_RTG": 118.2, "DEF_RTG": 108.4, "NET_RTG": 9.8},
    "Valencia Basket": {"OFF_RTG": 105.6, "DEF_RTG": 109.8, "NET_RTG": -4.2},
    "Virtus Bologna": {"OFF_RTG": 111.5, "DEF_RTG": 115.2, "NET_RTG": -3.7},
}

df = pd.DataFrame(teams_data).T.reset_index()
df.columns = ["Team", "OFF_RTG", "DEF_RTG", "NET_RTG"]
df["Season"] = 2025

print(df.to_string())