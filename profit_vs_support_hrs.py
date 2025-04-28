# %%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from natrium_model import *

market_data = pd.read_csv("market_data_2022.csv")
market_data = market_data.sort_values(by="hour_of_year", ascending=True)
lmp_data_IL = market_data["Illinois Hub"].values
day_cycles = lmp_data_IL.reshape((365, 24))
initial_day = day_cycles[1]
storage_values = np.linspace(0, 24, 25)
profits = []
for sv in storage_values:
    print(sv)
    reactor = DayAheadNatrium(initial_day, max_output_hours=sv)
    sq, dq = optimize_reactor_bids(reactor, day_cycles)
    reactor = DayAheadNatrium(
        initial_day,
        storage_quantile=sq,
        discharge_quantile=dq,
        max_output_hours=sv,
    )
    for day in day_cycles:
        reactor.update(day)
    profits.append(
        reactor.get_average_price_for_electricity()
        - reactor.get_cost_of_electricity_per_mwh()
    )
plt.plot(storage_values, profits)

# %%
market_data = pd.read_csv("market_data_2022_caiso.csv")
market_data = market_data.sort_values(by="hour_of_year", ascending=True)
lmp_data = market_data["PALOVRDE_ASR-APND LMP"].values
day_cycles = np.nan_to_num(lmp_data.reshape((365, 24)))
initial_day = day_cycles[1]
storage_values = np.linspace(0, 24, 25)
profits_ca = []
for sv in storage_values:
    print(sv)
    reactor = DayAheadNatrium(initial_day, max_output_hours=sv)
    sq, dq = optimize_reactor_bids(reactor, day_cycles)
    reactor = DayAheadNatrium(
        initial_day,
        storage_quantile=sq,
        discharge_quantile=dq,
        max_output_hours=sv,
    )
    for day in day_cycles:
        reactor.update(day)
    profits_ca.append(
        reactor.get_average_price_for_electricity()
        - reactor.get_cost_of_electricity_per_mwh()
    )
plt.plot(storage_values, profits_ca)

# %%
plt.plot(storage_values, profits, label="MISO Data - Illinois")
plt.plot(storage_values, profits_ca, label="CAISO Data - Palo Verde")
plt.title("Profit Per MWhe vs. Storage Capacity")
plt.xlabel("Maximum Output Support [hrs]")
plt.ylabel("Profit [$/MWhe]")
plt.legend()
plt.show()

# %%
