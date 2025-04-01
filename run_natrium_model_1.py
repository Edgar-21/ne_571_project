from natrium_model import Natrium
import pandas as pd

market_data = pd.read_csv("market_data_2022.csv")
market_data = market_data.sort_values(by="hour_of_year", ascending=True)
lmp_data_IL = market_data["Illinois Hub"].values

# Case 1
bids = [
    [7, 100],
    [14, 200],
    [21, 300],
    [28, 400],
    [35, 500],
]
max_output = 500
IL_reactor = Natrium(bids, max_output)
for lmp in lmp_data_IL:
    IL_reactor.update(lmp)

results_df = IL_reactor.get_dataframe()

results_df.to_csv("IL_reactor.csv")
print("case 1 revenue:", results_df["Revenue"].sum() / 1e6)

# Case 2
bids = [
    [14, 100],
    [35, 200],
    [60, 300],
    [90, 400],
    [100, 500],
]
IL_reactor = Natrium(bids, max_output)
for lmp in lmp_data_IL:
    IL_reactor.update(lmp)

results_df = IL_reactor.get_dataframe()

results_df.to_csv("IL_reactor.csv")
print("case 2 revenue:", results_df["Revenue"].sum() / 1e6)

# Case 3
bids = [[7, 500]]
IL_reactor = Natrium(bids, max_output)
for lmp in lmp_data_IL:
    IL_reactor.update(lmp)

results_df = IL_reactor.get_dataframe()

results_df.to_csv("IL_reactor.csv")
print("case 3 revenue:", results_df["Revenue"].sum() / 1e6)
