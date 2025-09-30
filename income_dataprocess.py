#########################################################################
# Run this for data preprocessing.
# You may change parameters in income_config.py to experiment with
# different settings.
# This file is used to load, clean, preprocess, and split the dataset.
#########################################################################

import income_config as cfg
import income_datafunctions as dfs
import pandas as pd
from sklearn.model_selection import train_test_split


# load the dataset
print("\nLoading dataset", cfg.DATAFILE , "...", end="")
df = pd.read_csv(cfg.DATAFILE)
print("done", end="\n\n")

# replace "?" with NaN
df = df.replace("?", pd.NA)
dfs.nan_report(df)
print()
df = df.dropna()
print("After dropping rows with missing values:")
dfs.nan_report(df)
print()

# print summary statistics for numerical features
print("Summary statistics for numerical features:")
print(df.describe())
print()

# print summary statistics for target label
print("Summary statistics for target label:")
print(df['income'].value_counts(normalize=True))
print()

# convert target labels to binary values
# notice the labels are mislabelled in the dataset
# and we must account for that
print("Converting target labels to binary values...")
df['income'] = df['income'].apply(lambda x: 1 if x == '>50K' or x == '>50K.' else 0)
print("Summary statistics for target label:")
print(df['income'].value_counts(normalize=True), end="\n\n")

# strip whitespace from string columns
df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

# group countries with less than COUNTRYTHRESHOLD occurrences into "Other"
# this is done so we don'end up with too many dummy variables later
print("Grouping countries with less than", cfg.COUNTRYTHRESHOLD, "occurrences into 'Other'...")
print(df['native-country'].value_counts(), end="\n\n")
df = dfs.group_rare_countres(df)
print(df['native-country'].value_counts(), end="\n\n")

# one-hot encode categorical features, drop the first category to avoid multicollinearity
print("One-hot encoding categorical features...")
df = pd.get_dummies(df, drop_first=True)
print("After one-hot encoding categorical features:")
print(df.head(), end="\n\n")

# get a list of numeric columns
numeric_df = df.select_dtypes(include=['number'])
# remove income from that list, it's already binary
numeric_df = numeric_df.drop(columns=['income'])
print("Numerical features to be scaled:")
print(numeric_df.columns, end="\n\n")

# scale those columns
# we can choose StandardScaler or MinMaxScaler 0,1 or -1,1
print("Scaling numerical features using", cfg.SCALER, "scaler...")
if cfg.SCALER == "standard":
    print("Scaling numerical features to have mean 0 and variance 1...")
elif cfg.SCALER == "minmax01":
    print("Scaling numerical features to be in the range 0,1...")
elif cfg.SCALER == "minmax-11":
    print("Scaling numerical features to be in the range -1,1...")
df = dfs.scaler(numeric_df, df)
print("After scaling numerical features, showing mins and maxs of numerical features:")
print(df[numeric_df.columns].agg(['min', 'max']).T, end="\n\n")

# get a correlation matrix
correlation_matrix = df.corr()
# save correlation matrix and save it as a CSV file
correlation_matrix.to_csv(cfg.CORRELATIONFILE)

# make and save a heatmap of the correlation matrix
dfs.correlation_heatmap(correlation_matrix, show_plot=cfg.SHOWHEATMAP)
print("Correlation matrix heatmap saved as:", cfg.HEATMAPFILE, end="\n\n")

#drop highly correlated features (correlation > cfg.CORRELATIONTHRESHOLD)
shape_before = df.shape
dropped, df = dfs.remove_highly_correlated_features(df, correlation_matrix, threshold=cfg.CORRELATIONTHRESHOLD)
print("Identified highly correlated features (correlation >", cfg.CORRELATIONTHRESHOLD, "):", end="")
if not dropped:
    print("None", end="\n\n")
else:
    print("\nDropping highly correlated features:", dropped, end="\n\n")
    print("Shape before dropping highly correlated features:", shape_before)
    print("Shape after dropping highly correlated features:", df.shape, end="\n\n")

# move income column to the end (if it's not already there)
income = df.pop('income')
df['income'] = income

# save the cleaned dataset
df.to_csv("cleaned_" + cfg.DATAFILE, index=False)
print("Cleaned dataset saved as 'cleaned_" + cfg.DATAFILE + "'.", end="\n\n")

# train test split
train_df, test_df = train_test_split(df, test_size=cfg.TESTSIZE, random_state=cfg.RANDOMSTATE, stratify=df['income'])
train_df.to_csv("train_" + cfg.DATAFILE, index=False)
test_df.to_csv("test_" + cfg.DATAFILE, index=False)
print("Train and test datasets saved as 'train_" + cfg.DATAFILE + "' and 'test_" + cfg.DATAFILE + "'.")
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape, end="\n\n")
print("Train target distribution:")
print(train_df['income'].value_counts(normalize=True), end="\n\n")
print("Test target distribution:")
print(test_df['income'].value_counts(normalize=True), end="\n\n")
print()
