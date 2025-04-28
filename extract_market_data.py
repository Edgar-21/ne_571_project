import pandas as pd
import os
import glob
from datetime import datetime


def compute_hour_of_year(row):
    day_of_year = (
        datetime(int(row["year"]), int(row["month"]), int(row["day"]))
        .timetuple()
        .tm_yday
    )
    return (day_of_year - 1) * 24 + row["hour"]


# Define the base directory
base_dir = "./market_data_miso"

# Initialize an empty list to store dataframes
df_list = []

# Loop through each directory in the base directory
for subdir in os.listdir(base_dir):
    subdir_path = os.path.join(base_dir, subdir)
    # Ensure it's a directory
    if os.path.isdir(subdir_path):
        # Find all .xls files in the subdirectory
        for file in glob.glob(os.path.join(subdir_path, "*.xls")):

            # Read the Excel file
            df = pd.read_excel(
                file,
                sheet_name="Sheet1",
                usecols="A:J",
                skiprows=11,
                nrows=24,
            )

            # Rename and clean up the hour column
            df.rename(columns={df.columns[0]: "hour"}, inplace=True)
            df["hour"] = df["hour"].str.extract(r"(\d+)").astype(int)

            # Append the dataframe to the list
            df_list.append(df)

# Concatenate all dataframes into a single dataframe
final_df = pd.concat(df_list, ignore_index=True)
final_df["hour_of_year"] = final_df.apply(compute_hour_of_year, axis=1)
final_df.to_csv("market_data_2022_miso.csv")
