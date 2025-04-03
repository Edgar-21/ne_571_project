from natrium_model import Natrium, DayAheadNatrium
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

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
    [20, 100],
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


print("------------------ DAY AHEAD MODEL -------------------------")

storage_quantiles = np.linspace(0.10, 0.40, 31)
discharge_quantiles = np.linspace(0.60, 0.90, 31)

cases = []
for sq in storage_quantiles:
    for dq in discharge_quantiles:
        cases.append([sq, dq])

day_cycles = lmp_data_IL.reshape((365, 24))
initial_day = day_cycles[37]
revenues = []

for i, case in enumerate(cases):
    sq, dq = case
    print(f"Case {i+1}")
    print(f"sq: {sq}, dq: {dq}")

    reactor = DayAheadNatrium(
        initial_day, discharge_quantile=dq, storage_quantile=sq
    )
    for day in day_cycles:
        reactor.update(day)

    results_df = reactor.get_dataframe()
    revenue = results_df["Revenue"].sum() / 1e6
    revenues.append(revenue)
    print("revenue:", revenue)

revenues = np.array(revenues)

max_revenue = np.max(revenues)
max_revenue_case = np.argmax(revenues)

print(f"Best case {max_revenue_case+1}")
print(f"sq: {cases[max_revenue_case][0]}, dq:{cases[max_revenue_case][1]}")
print(f"Max Revenue: {max_revenue}")
