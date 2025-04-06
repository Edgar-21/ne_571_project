import pandas as pd
import numpy as np

# Cost Data
# https://doi.org/10.1016/j.apenergy.2024.124105
npp_capital_per_mwth = 1.33e9 / 840  # $/MWth
npp_fixed_costs_per_year_per_mwth = 44.8e3 / 840  # $/MWth/yr
npp_variable_costs_per_mwhth = 7.43  # $/MWhth

bop_capital_per_mwe = 6e8 / 500  # $/MWe
bop_fixed_costs_per_year_per_mwe = 50e3 / 500  # $/MWe/yr
bop_variable_costs_per_mwhe = 1  # $/MWhe

tes_capital_per_mwhth = 6.75e6 / 2288  # $/MWhth
tes_fixed_costs_per_year_per_mwhth = 28.5e3 / 2288  # $/MWhth/yr
tes_variable_costs = 0  # The reference listed in the above paper has variable
# costs from a white paper on hot water sensible heat storage with dubious
# units. Unless I find a better number I'm leaving this at zero.
# Variable costs listed at 16$/MWhth (!)

# Assumed tax and interest rates
tau = 0.4
p_tax = 0.01
x = 0.08


def calc_a_over_p(i, n):
    return i * (1 + i) ** n / ((1 + i) ** n - 1)


def calc_levelized_capex(lifetime, p_tax, tau, x, cap_cost, power):
    """return the levelized capex in $/year, assuming straight line depreciation"""
    a_o_p = calc_a_over_p(x, lifetime)
    return (
        cap_cost
        * (p_tax + a_o_p / (1 - tau) - 1 / lifetime * tau / (1 - tau))
        / power
        / 365
        / 24
    )


class DayAheadNatrium(object):
    """Use the previous day's data to determine current day's bids"""

    def __init__(
        self,
        start_day,
        min_output=100,
        max_output=500,
        max_output_hours=5.5,
        continuous_output=345,
        storage_quantile=0.20,
        discharge_quantile=0.75,
        efficiency=0.4,
    ):
        self.previous_day = start_day
        self.variable_costs = 7
        # Some starting point for the thermal storage
        self.stored_energy = 0  # MWHe
        self.minimum_stored_energy = 0  # MWHe
        self.storage_quantile = storage_quantile
        self.discharge_quantile = discharge_quantile

        # Following values from https://www.terrapower.com/downloads/Natrium_Technology.pdf
        self.continuous_output = continuous_output  # MWe
        self.max_stored_energy = (
            max_output - continuous_output
        ) * max_output_hours  # MWhe
        self.thermal_efficiency = efficiency
        self.thermal_power = self.continuous_output / efficiency
        self.ramping_rate = 0.1  # fraction per minute
        self.max_output = max_output  # MWe
        self.min_output = min_output
        self.discharge_rate = max_output - continuous_output
        self.charge_rate = continuous_output
        self.continuous_output = continuous_output
        self.lifetime = 80  # years

        self.time_step = 0

        self.data_dict = {
            "Hour": [],
            "LMP": [],
            "Storage Bid": [],
            "Discharge Bid": [],
            "Actual Output [MWhe]": [],
            "Discharge Rate [MWhe]": [],
            "Thermal Storage Level (start of timestep) [MWhe]": [],
            "Revenue": [],
        }

    def determine_bid(self):
        self.storage_bid = np.quantile(
            self.previous_day, self.storage_quantile
        )
        self.discharge_bid = np.quantile(
            self.previous_day, self.discharge_quantile
        )

    def determine_output(self, lmp):
        # our battery wins vs the market
        if lmp < self.storage_bid:
            if self.stored_energy + self.charge_rate <= self.max_stored_energy:
                return self.min_output
            else:
                return self.continuous_output - (
                    self.max_stored_energy - self.stored_energy
                )

        if self.storage_bid <= lmp and lmp <= self.discharge_bid:
            return self.continuous_output

        # check if the market is bidding high enough to get the battery juice
        if lmp > self.discharge_bid:
            # check if we have juice to give for a full hour
            if (
                self.stored_energy - self.discharge_rate
                >= self.minimum_stored_energy
            ):
                return self.max_output
            # if not give continuous output plus whatever is in storage
            else:
                return self.continuous_output + self.stored_energy

    def get_capex_costs(self):
        """Calculates and returns the levelized cost of each major capital item
        in $/MWhe"""
        self.levelized_npp_capex = calc_levelized_capex(
            self.lifetime,
            p_tax,
            tau,
            x,
            self.thermal_power * npp_capital_per_mwth,
            self.continuous_output,
        )
        self.levelized_bop_capex = calc_levelized_capex(
            self.lifetime,
            p_tax,
            tau,
            x,
            self.max_output * bop_capital_per_mwe,
            self.continuous_output,
        )
        self.levelilzed_tes_capex = calc_levelized_capex(
            self.lifetime,
            p_tax,
            tau,
            x,
            self.max_stored_energy
            * tes_capital_per_mwhth
            / self.thermal_efficiency,
            self.continuous_output,
        )
        self.total_levelized_capex = (
            self.levelilzed_tes_capex
            + self.levelized_bop_capex
            + self.levelized_npp_capex
        )
        return (
            self.levelized_npp_capex,
            self.levelized_bop_capex,
            self.levelilzed_tes_capex,
            self.total_levelized_capex,
        )

    def get_fixed_costs(self):
        self.npp_fixed_om = (
            npp_fixed_costs_per_year_per_mwth
            * self.thermal_power
            / self.continuous_output
            / 365
            / 24
        )  # $/MWhe
        self.bop_fixed_om = (
            bop_fixed_costs_per_year_per_mwe
            * self.continuous_output
            / self.continuous_output
            / 365
            / 24
        )  # $/MWhe
        self.tes_fixed_om = (
            tes_fixed_costs_per_year_per_mwhth
            * self.max_stored_energy
            / self.thermal_efficiency
            / self.continuous_output
            / 365
            / 24
        )
        self.total_fixed_om = (
            self.npp_fixed_om + self.bop_fixed_om + self.tes_fixed_om
        )
        return (
            self.npp_fixed_om,
            self.bop_fixed_om,
            self.tes_fixed_om,
            self.total_fixed_om,
        )

    def update(self, day_lmp):
        for lmp in day_lmp:
            self.determine_bid()
            output = self.determine_output(lmp)
            discharge_rate = self.continuous_output - output
            revenue = output * lmp

            self.time_step += 1
            if self.stored_energy < 0:
                print("negative thermal storage")
            if self.stored_energy > self.max_stored_energy:
                print("salt too hot")

            self.data_dict["Hour"].append(self.time_step)
            self.data_dict["LMP"].append(lmp)
            self.data_dict["Storage Bid"].append(self.storage_bid)
            self.data_dict["Discharge Bid"].append(self.discharge_bid)
            self.data_dict["Actual Output [MWhe]"].append(output)
            self.data_dict["Discharge Rate [MWhe]"].append(discharge_rate)
            self.data_dict[
                "Thermal Storage Level (start of timestep) [MWhe]"
            ].append(self.stored_energy)
            self.data_dict["Revenue"].append(revenue)
            self.stored_energy += discharge_rate
        self.previous_day = day_lmp

    def get_dataframe(self):
        return pd.DataFrame(self.data_dict)

    def get_cost_of_electricity_per_mwh(self):
        self.get_capex_costs()
        self.get_fixed_costs()
        return (
            self.total_fixed_om
            + self.total_levelized_capex
            + bop_variable_costs_per_mwhe
            + tes_variable_costs
            + npp_variable_costs_per_mwhth / self.thermal_efficiency
        )
