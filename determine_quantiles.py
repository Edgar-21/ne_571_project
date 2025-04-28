from natrium_model import DayAheadNatrium
import pandas as pd
import numpy as np
import concurrent.futures


def determine_quantiles(
    lmp_data,
    storage_quantiles,
    discharge_quantiles,
    reactor_name,
    storage_time,
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
            initial_day,
            discharge_quantile=dq,
            storage_quantile=sq,
            max_output_hours=storage_time,
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
        storage_quantile=round(cases[max_revenue_case_id][0], 2),
        discharge_quantile=round(cases[max_revenue_case_id][1], 2),
        max_output_hours=storage_time,
    )
    for day in day_cycles:
        max_revenue_reactor.update(day)
    df = max_revenue_reactor.get_dataframe()
    df.to_csv(
        f"{reactor_name}_{round(max_rev_storage_quantile,2)}_{round(max_rev_discharge_quantile,2)}_storage_{round(storage_time,2)}.csv"
    )
    return max_revenue, cases[max_revenue_case_id]


def process_storage_time(
    st, lmp_data, storage_quantiles, discharge_quantiles, reactor_name
):
    print(f"Processing storage time: {st}")
    max_revenue, quantiles = determine_quantiles(
        lmp_data, storage_quantiles, discharge_quantiles, reactor_name, st
    )
    return [st, max_revenue, *quantiles]


# Main code
market_data = pd.read_csv("market_data_2022.csv")
market_data = market_data.sort_values(by="hour_of_year", ascending=True)
lmp_data_IL = market_data["Illinois Hub"].values
lmp_data_AR = market_data["Arkansas Hub"].values
storage_quantiles = np.linspace(0.10, 0.40, 31)
discharge_quantiles = np.linspace(0.60, 0.90, 31)
storage_times = np.linspace(1, 24, 24)

# Parallelize the outer loop
results = []
with concurrent.futures.ProcessPoolExecutor(max_workers=18) as executor:
    # Create a list of futures
    futures = [
        executor.submit(
            process_storage_time,
            st,
            lmp_data_IL,
            storage_quantiles,
            discharge_quantiles,
            "il_reactor",
        )
        for st in storage_times
    ]

    # Collect results as they complete
    for future in concurrent.futures.as_completed(futures):
        try:
            result = future.result()
            results.append(result)
        except Exception as e:
            print(f"An error occurred: {e}")

# Sort results by storage time to maintain original order
results.sort(key=lambda x: x[0])

np.save("results.npy", results)

print(np.array(results))
