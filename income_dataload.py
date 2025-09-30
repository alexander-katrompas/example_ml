#########################################################################
# Run this once to generate the CSV files needed for the assignment.
# You may also uncomment the print statements to inspect the datasets.
# But remember to re-comment them before submission.
# This file is used to load datasets from UCI Machine Learning Repository
# and save them as CSV files for further use.
#########################################################################

import income_config as cfg
import pandas as pd
from ucimlrepo import fetch_ucirepo

"""
Load the income Census dataset from UCI Machine Learning Repository
and save it as a CSV file named 'income_census.csv'.
"""
# Adult income Census dataset ID in UCI ML Repository
# URL: https://archive.ics.uci.edu/ml/datasets/income
# ID: 2
# The dataset contains information about individuals from the 1994 Census database.
# The task is to predict whether a person makes over $50K a year.

# fetch dataset
print("Fetching income Census dataset...", end="")
income = fetch_ucirepo(id=2)
print("Done.")

# if you want to inspect the dataset structure, uncomment the following lines:
# print(income.keys())
# print(income['data'].keys())
# print(income['data']['features'].head())
# print(income['data']['targets'].head())
# re-comment them once you are done inspecting

# Combine features and target(s)
df = pd.concat([income['data']['features'], income['data']['targets']], axis=1)
# Save to CSV
df.to_csv(cfg.DATAFILE, index=False)
print("Income Census dataset saved as:", cfg.DATAFILE)
