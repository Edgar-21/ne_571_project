from natrium_model import DayAheadNatrium
import pandas as pd
import numpy as np


def determine_quantiles(
    lmp_data, storage_quantiles, discharge_quantiles, reactor_name
):
    cases = []
    for sq in storage_quantiles:
        for dq in discharge_quantiles:
            cases.append([sq, dq])

    day_cycles = lmp_data.reshape((365, 24))
    initial_day = day_cycles[1]
    revenues = []

    for case in cases:
        sq, dq = case

        reactor = DayAheadNatrium(
            initial_day, discharge_quantile=dq, storage_quantile=sq
        )
        for day in day_cycles:
            reactor.update(day)

        results_df = reactor.get_dataframe()
        revenue = results_df["Revenue"].sum() / 1e6
        revenues.append(revenue)

    revenues = np.array(revenues)

    max_revenue = np.max(revenues)
    max_revenue_case_id = np.argmax(revenues)
    max_rev_storage_quantile = cases[max_revenue_case_id][0]
    max_rev_discharge_quantile = cases[max_revenue_case_id][1]

    max_revenue_reactor = DayAheadNatrium(
        initial_day,
        storage_quantile=round(cases[max_revenue_case_id][0],2)
        discharge_quantile=round(cases[max_revenue_case_id][1],2)
    )
    for day in day_cycles:
        max_revenue_reactor.update(day)

    df = max_revenue_reactor.get_dataframe()
    df.to_csv(
        f"{reactor_name}_{max_rev_storage_quantile}_{max_rev_discharge_quantile}.csv"
    )

    return max_revenue, cases[max_revenue_case_id]


market_data = pd.read_csv("market_data_2022.csv")
market_data = market_data.sort_values(by="hour_of_year", ascending=True)
lmp_data_IL = market_data["Illinois Hub"].values
lmp_data_AR = market_data["Arkansas Hub"].values

storage_quantiles = np.linspace(0.10, 0.40, 31)
discharge_quantiles = np.linspace(0.60, 0.90, 31)

max_revenue_il, max_revenue_case_il = determine_quantiles(
    lmp_data_IL, storage_quantiles, discharge_quantiles, "il_reactor"
)

print("IL Reactor")
print(f"sq: {max_revenue_case_il[0]}, dq:{max_revenue_case_il[1]}")
print(f"Max Revenue: {max_revenue_il}")

max_revenue_ar, max_revenue_case_ar = determine_quantiles(
    lmp_data_AR, storage_quantiles, discharge_quantiles, "ar_reactor"
)

print("AR Reactor")
print(f"sq: {max_revenue_case_ar[0]}, dq:{max_revenue_case_ar[1]}")
print(f"Max Revenue: {max_revenue_ar}")
