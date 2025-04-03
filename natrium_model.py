import pandas as pd
import numpy as np


class Natrium(object):

    def __init__(self, bids, max_output):

        self.bids = bids

        # Some starting point for the thermal storage
        self.stored_energy = 100  # MWHe
        self.minimum_stored_energy = 0  # MWHe

        # Following values from https://www.terrapower.com/downloads/Natrium_Technology.pdf
        self.max_stored_energy = (500 - 345) * 5.5  # MWhe
        self.thermal_efficiency = 345 / 840
        self.ramping_rate = 0.1  # fraction per minute
        self.max_output = max_output  # MWhe
        self.continuous_output = 345  # MWhe

        self.time_step = 0

        self.data_dict = {
            "Timestep": [],
            "LMP": [],
            "Bid": [],
            "Bid Output": [],
            "Actual Output [MWhe]": [],
            "Discharge Rate [MWhe]": [],
            "Thermal Storage Level (start of timestep) [MWhe]": [],
            "Revenue": [],
        }

    def determine_bid(self, lmp):
        bid_price = self.bids[0][0]
        output = self.bids[0][1]
        for bid in self.bids:
            if lmp >= bid[0]:
                bid_price = bid[0]
                output = bid[1]
        return output, bid_price

    def determine_output(self, lmp):
        # Determine how much we want to sell for a given price

        output, _ = self.determine_bid(lmp)

        storage_charge_rate = self.continuous_output - output

        # Charging
        if storage_charge_rate >= 0:
            # If there is enough space in the battery, supply only the output
            if (
                self.stored_energy + storage_charge_rate
                <= self.max_stored_energy
            ):
                return output
            # Fill the battery, supply the excess
            else:
                amount_to_store = self.max_stored_energy - self.stored_energy
                return self.continuous_output - amount_to_store

        # Discharging
        if storage_charge_rate < 0:
            # If there is enough in the battery, supply the output
            if (
                self.stored_energy + storage_charge_rate
                >= self.minimum_stored_energy
            ):
                return output
            else:
                amount_in_storage = (
                    self.stored_energy - self.minimum_stored_energy
                )
                return self.continuous_output + amount_in_storage

    def update(self, lmp):
        bid_output, bid = self.determine_bid(lmp)
        output = self.determine_output(lmp)
        discharge_rate = self.continuous_output - output
        revenue = output * lmp

        self.time_step += 1
        if self.stored_energy < 0:
            print("negative thermal storage")
        if self.stored_energy > self.max_stored_energy:
            print("salt too hot")

        self.data_dict["Timestep"].append(self.time_step)
        self.data_dict["LMP"].append(lmp)
        self.data_dict["Bid"].append(bid)
        self.data_dict["Bid Output"].append(bid_output)
        self.data_dict["Actual Output [MWhe]"].append(output)
        self.data_dict["Discharge Rate [MWhe]"].append(discharge_rate)
        self.data_dict[
            "Thermal Storage Level (start of timestep) [MWhe]"
        ].append(self.stored_energy)
        self.data_dict["Revenue"].append(revenue)
        self.stored_energy += discharge_rate

    def get_dataframe(self):
        return pd.DataFrame(self.data_dict)


class DayAheadNatrium(object):
    def __init__(
        self,
        start_day,
        min_output=100,
        max_output=500,
        max_output_hours=5.5,
        continuous_output=345,
        storage_quantile=0.20,
        discharge_quantile=0.75,
    ):
        self.previous_day = start_day
        self.variable_costs = 7
        # Some starting point for the thermal storage
        self.stored_energy = 0  # MWHe
        self.minimum_stored_energy = 0  # MWHe
        self.storage_quantile = storage_quantile
        self.discharge_quantile = discharge_quantile

        # Following values from https://www.terrapower.com/downloads/Natrium_Technology.pdf
        self.max_stored_energy = (
            max_output - continuous_output
        ) * max_output_hours
        self.thermal_efficiency = 345 / 840
        self.ramping_rate = 0.1  # fraction per minute
        self.max_output = max_output  # MWhe
        self.min_output = min_output
        self.discharge_rate = max_output - continuous_output
        self.charge_rate = continuous_output
        self.continuous_output = continuous_output

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
