import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from natrium_model import *


def plotContour(
    solutions,
    x,
    y,
    barLabel,
    xlabel,
    ylabel,
    title,
    colormap="plasma",
    levels=10,
    filename=None,
    decimals=1,
):  # plot 2d solutions

    # plot it
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.title(title)
    levels = np.linspace(np.min(solutions), np.max(solutions), levels)
    print(levels)
    c = ax.contourf(x, y, solutions, levels, cmap=colormap)
    cbar = fig.colorbar(c, ax=ax, label=barLabel)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter(f"%.{decimals}f"))
    if filename is not None:
        fig.savefig(filename)
    else:
        fig.savefig(title + ".png")
    plt.close()


def plotHeatmap(
    solutions,
    x,
    y,
    barLabel,
    xlabel,
    ylabel,
    title,
    colormap="plasma",
    filename=None,
    decimals=1,
):
    """
    Plot 2D solutions as a heatmap
    """
    # Create figure and axis
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # Set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.title(title)

    # Create heatmap
    # Note: imshow by default puts origin at top-left, so we use origin='lower' to match contourf
    im = ax.imshow(
        solutions,
        cmap=colormap,
        origin="lower",
        extent=[np.min(x), np.max(x), np.min(y), np.max(y)],
        aspect="auto",
    )

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, label=barLabel)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter(f"%.{decimals}f"))

    # Save figure
    if filename is not None:
        fig.savefig(filename)
    else:
        fig.savefig(title + ".png")

    plt.close()


market_data = pd.read_csv("market_data_2022.csv")
market_data = market_data.sort_values(by="hour_of_year", ascending=True)
lmp_data_IL = market_data["Illinois Hub"].values
day_cycles = np.nan_to_num(lmp_data_IL.reshape((365, 24)))
initial_day = day_cycles[1]

storage_time_data = np.array(
    [
        [0, 0, 0.1, 0.9],
        [1.0, 189.97738795, 0.36, 0.9],
        [2.0, 192.0155615, 0.36, 0.9],
        [3.0, 193.516199, 0.36, 0.76],
        [4.0, 195.06230165, 0.36, 0.74],
        [5.0, 196.4097745, 0.38, 0.68],
        [6.0, 197.88314915, 0.38, 0.65],
        [7.0, 199.43246145, 0.38, 0.6],
        [8.0, 200.85001175, 0.38, 0.61],
        [9.0, 201.93187575, 0.38, 0.6],
        [10.0, 202.80010855, 0.39, 0.6],
        [11.0, 203.52193135, 0.39, 0.6],
        [12.0, 204.00577165, 0.4, 0.6],
        [13.0, 204.5912712, 0.4, 0.6],
        [14.0, 204.92166215, 0.4, 0.6],
        [15.0, 205.5354957, 0.4, 0.6],
        [16.0, 206.1971767, 0.4, 0.6],
        [17.0, 206.7644915, 0.4, 0.6],
        [18.0, 207.1700066, 0.39, 0.6],
        [19.0, 207.51469465, 0.39, 0.6],
        [20.0, 207.818064, 0.38, 0.6],
        [21.0, 208.04803885, 0.37, 0.6],
        [22.0, 208.26194885, 0.37, 0.6],
        [23.0, 208.4094155, 0.36, 0.6],
        [24.0, 208.56600285, 0.36, 0.6],
    ]
)
tes_fixed_costs_per_year_per_mwhth = np.linspace(1, 30, 10)

data = np.zeros(
    (len(storage_time_data), len(tes_fixed_costs_per_year_per_mwhth), 3)
)

for i, st in enumerate(storage_time_data):
    sq = st[2]
    dq = st[3]
    hours = st[0]
    for j, cc in enumerate(tes_fixed_costs_per_year_per_mwhth):
        reactor = DayAheadNatrium(
            initial_day,
            discharge_quantile=dq,
            storage_quantile=sq,
            max_output_hours=hours,
        )
        reactor.tes_fixed_costs_per_year_per_mwhth = cc * 1000
        reactor.tes_capital_per_mwhth = 0
        for day in day_cycles:
            reactor.update(day)
        lcoe = reactor.get_cost_of_electricity_per_mwh()
        average_price = (
            sum(reactor.data_dict["Revenue"])
            / reactor.continuous_output
            / 365
            / 24
        )
        data[i, j] = [lcoe, average_price, average_price - lcoe]

np.save("tes_capex_sensitivity.npy", data)

for subset, name in enumerate(
    ["LCOE", "Average Price Per MWhe", "Profit Per MWhe"]
):
    plotHeatmap(
        data[:, :, subset].T,
        storage_time_data[:, 0],
        tes_fixed_costs_per_year_per_mwhth,
        name,
        "Storage Time at Max Output [hrs]",
        "Fixed Costs [$/KWh_th/year]",
        f"MISO Price Data\n {name}",
        colormap="viridis",
        filename=f"miso_{name}.png",
        decimals=1,
    )
