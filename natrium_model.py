import pandas as pd


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

        self.time_step_hrs = 1  # hrs
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
                self.stored_energy + storage_charge_rate * self.time_step_hrs
                <= self.max_stored_energy
            ):
                return output * self.time_step_hrs
            # Fill the battery, supply the excess
            else:
                amount_to_store = self.max_stored_energy - self.stored_energy
                return (
                    self.continuous_output * self.time_step_hrs
                    - amount_to_store
                )

        # Discharging
        if storage_charge_rate < 0:
            # If there is enough in the battery, supply the output
            if (
                self.stored_energy + storage_charge_rate * self.time_step_hrs
                >= self.minimum_stored_energy
            ):
                return output * self.time_step_hrs
            else:
                amount_in_storage = (
                    self.stored_energy - self.minimum_stored_energy
                )
                return (
                    self.continuous_output * self.time_step_hrs
                    + amount_in_storage
                )

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
