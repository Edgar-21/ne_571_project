# %%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from natrium_model import *
from matplotlib.ticker import FormatStrFormatter


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
    annotate=True,
    annotation_color="black",
    annotation_size=8,
):
    """
    Plot 2D solutions as a heatmap with cell centers aligned to x and y values
    """
    # Create figure and axis
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # Set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.title(title)

    # Calculate cell extents - ensure each cell is centered on its x,y value
    # For n values, we need n+1 edges
    if len(x) > 1:
        dx = np.min(np.diff(x))
    else:
        dx = 1.0  # Default if only one x value

    if len(y) > 1:
        dy = np.min(np.diff(y))
    else:
        dy = 1.0  # Default if only one y value

    # Create extended x and y edges
    x_edges = np.zeros(len(x) + 1)
    x_edges[0] = x[0] - dx / 2
    x_edges[1:] = x + dx / 2

    y_edges = np.zeros(len(y) + 1)
    y_edges[0] = y[0] - dy / 2
    y_edges[1:] = y + dy / 2

    # Create heatmap using pcolormesh which allows explicit cell edge positions
    mesh = ax.pcolormesh(x_edges, y_edges, solutions, cmap=colormap)

    # Add colorbar
    cbar = fig.colorbar(mesh, ax=ax, label=barLabel)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter(f"%.{decimals}f"))

    # Set axis ticks to match your actual data values
    ax.set_xticks(x)
    ax.set_yticks(y)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center")

    # Add padding at the bottom for the rotated labels
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Adjust this value as needed
    # Add annotations
    if annotate:
        # Add text annotations at center of each cell
        for i in range(len(y)):
            for j in range(len(x)):
                value = solutions[i, j]
                if not np.isnan(value):  # Skip annotation for NaN values
                    ax.text(
                        x[j],
                        y[i],
                        f"{round(value)}",
                        ha="center",
                        va="center",
                        color=annotation_color,
                        fontsize=annotation_size,
                    )

    # Save figure
    if filename is not None:
        fig.savefig(filename)
    else:
        fig.savefig(title + ".png")

    plt.close()

    return fig, ax


def prepare_heatmap_data(df, x_col, y_col, z_col):
    """
    Convert a dataframe with x, y, z columns to a format suitable for the heatmap function.

    Parameters:
    -----------
    df : pandas DataFrame
        Input dataframe with columns for x, y, and z values
    x_col : str
        Name of column containing x values
    y_col : str
        Name of column containing y values
    z_col : str
        Name of column containing z values (color/heatmap values)

    Returns:
    --------
    solutions : ndarray
        2D array of z values arranged in a grid
    x : ndarray
        Unique x values
    y : ndarray
        Unique y values
    """
    # Extract unique x and y values, sorted
    x_values = sorted(df[x_col].unique())
    y_values = sorted(df[y_col].unique())

    # Create empty solutions grid
    solutions = np.zeros((len(y_values), len(x_values)))
    solutions.fill(
        np.nan
    )  # Fill with NaN so missing values are handled properly

    # Fill the solutions grid with z values
    for _, row in df.iterrows():
        x_idx = x_values.index(row[x_col])
        y_idx = y_values.index(row[y_col])
        solutions[y_idx, x_idx] = row[z_col]

    # Convert to numpy arrays
    x = np.array(x_values)
    y = np.array(y_values)

    return solutions, x, y


storage_values = np.linspace(0, 24, 12)  # hrs
overhead_costs = np.linspace(0, 30, 10)  # $/kwhth/yr

# %%

market_data = pd.read_csv("market_data_2022.csv")
market_data = market_data.sort_values(by="hour_of_year", ascending=True)
lmp_data_IL = market_data["Illinois Hub"].values
day_cycles = lmp_data_IL.reshape((365, 24))
initial_day = day_cycles[1]

# for each configuration we may wish to know
# - fixed costs due to TES per MWhe
# - storage and discharge percentiles
# - revenue per mwhe
# - total cost per mwhe
# - profit per mwhe

data_dict = {
    "tes_capacity_hours": [],
    "tes_fixed_costs_per_mwhe": [],
    "tes_fixed_costs_per_mwhth_per_year": [],
    "storage_percentile": [],
    "discharge_percentile": [],
    "total_cost_per_mwhe": [],
    "revenue_per_mwhe": [],
    "profit_per_mwhe": [],
}

for sv in storage_values:
    for oc in overhead_costs:
        reactor = DayAheadNatrium(initial_day, max_output_hours=sv)
        reactor.tes_capital_per_mwhth = 0
        reactor.tes_fixed_costs_per_year_per_mwhth = oc * 1000
        sq, dq = optimize_reactor_bids(reactor, day_cycles)
        reactor = DayAheadNatrium(
            initial_day,
            max_output_hours=sv,
            storage_quantile=sq,
            discharge_quantile=dq,
        )
        reactor.tes_capital_per_mwhth = 0
        reactor.tes_fixed_costs_per_year_per_mwhth = oc * 1000
        for day in day_cycles:
            reactor.update(day)
        _, _, tes_fixed_costs_per_mwhe, total_fixed_costs = (
            reactor.get_fixed_costs()
        )
        _, _, _, total_capex_costs = reactor.get_capex_costs()
        lcoe = reactor.get_cost_of_electricity_per_mwh()
        revenue_mwhe = reactor.get_average_price_for_electricity()

        data_dict["tes_capacity_hours"].append(sv)
        data_dict["tes_fixed_costs_per_mwhe"].append(tes_fixed_costs_per_mwhe)
        data_dict["tes_fixed_costs_per_mwhth_per_year"].append(oc * 1000)
        data_dict["storage_percentile"].append(sq)
        data_dict["discharge_percentile"].append(dq)
        data_dict["total_cost_per_mwhe"].append(lcoe)
        data_dict["revenue_per_mwhe"].append(revenue_mwhe)
        data_dict["profit_per_mwhe"].append(revenue_mwhe - lcoe)

miso_df = pd.DataFrame(data_dict)
miso_df.to_csv("miso_heatmap.csv")


# %%
miso_df = pd.read_csv("miso_heatmap.csv")
sol, x, y = prepare_heatmap_data(
    miso_df,
    "tes_capacity_hours",
    "tes_fixed_costs_per_mwhth_per_year",
    "profit_per_mwhe",
)
plotHeatmap(
    sol,
    x,
    y / 1000,
    "Profit \$/MWh$_e$",
    "Hours of Max Output Support",
    "TES Overhead Costs \$/KWh$_{th}$/year",
    "MISO Data - TES Capacity and Cost Effects",
    filename="fixed_cost_heatmap_il.png",
    annotate=True,
    colormap="viridis",
)

# %%

market_data = pd.read_csv("market_data_2022_caiso.csv")
market_data = market_data.sort_values(by="hour_of_year", ascending=True)
lmp_data_ca = market_data["PALOVRDE_ASR-APND LMP"].values
day_cycles = np.nan_to_num(lmp_data_ca.reshape((365, 24)))
initial_day = day_cycles[1]

# for each configuration we may wish to know
# - fixed costs due to TES per MWhe
# - storage and discharge percentiles
# - revenue per mwhe
# - total cost per mwhe
# - profit per mwhe

data_dict = {
    "tes_capacity_hours": [],
    "tes_fixed_costs_per_mwhe": [],
    "tes_fixed_costs_per_mwhth_per_year": [],
    "storage_percentile": [],
    "discharge_percentile": [],
    "total_cost_per_mwhe": [],
    "revenue_per_mwhe": [],
    "profit_per_mwhe": [],
}
storage_values = np.linspace(0, 24, 12)  # hrs
overhead_costs = np.linspace(0, 30, 10)  # $/kwhth/yr
for sv in storage_values:
    for oc in overhead_costs:
        reactor = DayAheadNatrium(initial_day, max_output_hours=sv)
        reactor.tes_capital_per_mwhth = 0
        reactor.tes_fixed_costs_per_year_per_mwhth = oc * 1000
        sq, dq = optimize_reactor_bids(reactor, day_cycles)
        reactor = DayAheadNatrium(
            initial_day,
            max_output_hours=sv,
            storage_quantile=sq,
            discharge_quantile=dq,
        )
        reactor.tes_capital_per_mwhth = 0
        reactor.tes_fixed_costs_per_year_per_mwhth = oc * 1000
        for day in day_cycles:
            reactor.update(day)
        _, _, tes_fixed_costs_per_mwhe, total_fixed_costs = (
            reactor.get_fixed_costs()
        )
        _, _, _, total_capex_costs = reactor.get_capex_costs()
        lcoe = reactor.get_cost_of_electricity_per_mwh()
        revenue_mwhe = reactor.get_average_price_for_electricity()

        data_dict["tes_capacity_hours"].append(sv)
        data_dict["tes_fixed_costs_per_mwhe"].append(tes_fixed_costs_per_mwhe)
        data_dict["tes_fixed_costs_per_mwhth_per_year"].append(oc * 1000)
        data_dict["storage_percentile"].append(sq)
        data_dict["discharge_percentile"].append(dq)
        data_dict["total_cost_per_mwhe"].append(lcoe)
        data_dict["revenue_per_mwhe"].append(revenue_mwhe)
        data_dict["profit_per_mwhe"].append(revenue_mwhe - lcoe)

caiso_df = pd.DataFrame(data_dict)


# %%
sol, x, y = prepare_heatmap_data(
    caiso_df,
    "tes_capacity_hours",
    "tes_fixed_costs_per_mwhth_per_year",
    "profit_per_mwhe",
)
plotHeatmap(
    sol,
    x,
    y / 1000,
    "Profit \$/MWh$_e$",
    "Hours of Max Output Support",
    "TES Overhead Costs \$/KWh$_{th}$/year",
    "CAISO Data - TES Capacity and Cost Effects",
    filename="fixed_cost_heatmap_ca.png",
    annotate=True,
    colormap="viridis",
)

# %%
miso_df.to_csv("miso_heatmap.csv")
caiso_df.to_csv("caiso_heatmap.csv")

# %%
