#########################################################################
# This file is used to define functions for data preprocessing
# You may change parameters in income_config.py to experiment with
# different settings.
#########################################################################

import income_config as cfg
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


def nan_report(df):
    """
    Report the number of missing and non-missing values for each column in the dataframe.
    @param df: the dataframe to report on
    @return: na
    @exception None
    """
    print("Missing values report:")
    summary = pd.DataFrame({
        "total": len(df),
        "missing": df.isna().sum(),
        "non_missing": df.notna().sum()
    })
    print(summary)


def group_rare_countres(df):
    """
    Group countries with less than COUNTRYTHRESHOLD occurrences into "Other".
    This is done so we don't end up with too many dummy variables later.
    @param df: the dataframe to process
    @return: df
    @exception None
    """
    value_counts = df['native-country'].value_counts()
    rare_countries = value_counts[value_counts < cfg.COUNTRYTHRESHOLD].index
    df['native-country'] = df['native-country'].replace(rare_countries, "Other")
    return df


def correlation_heatmap(correlation_matrix, show_plot=False):
    """
    Plot a heatmap of the correlation matrix for numerical features in the dataframe.
    @param df:
    @return: na
    @exception None
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt=".2f", cbar=True)
    plt.title("Correlation Matrix Heatmap")
    # save the heatmap
    plt.savefig(cfg.HEATMAPFILE)
    # show the heatmap
    if show_plot:
        plt.show()
    plt.close()
    return


def scaler(numeric_df, df):
    """
    Scale numerical features in the dataframe using the specified scaler from config.
    Options are:
    - 'standard': StandardScaler to have mean 0 and variance 1
    - 'minmax01': Min-Max scaling to be in the range 0,1
    - 'minmax-11': Min-Max scaling to be in the range -1,
    1. Raises ValueError if an invalid SCALER value is provided in config.
    2. You may need to adjust the scaling method based on your dataset and requirements.
    3. Different scaling methods can have different effects on model performance.
    4. It's often a good idea to experiment with different scaling methods to see which one
    works best for your specific use case.
    @param numeric_df: the dataframe containing only numerical features to be scaled
    @param df: the original dataframe containing all features
    @return: df with scaled numerical features
    @exception ValueError if an invalid SCALER value is provided in config
    5. Note: This function modifies the original dataframe by scaling the specified numerical features.
    """
    if cfg.SCALER == 'standard':
        # to have mean 0 and variance 1
        df[numeric_df.columns] = StandardScaler().fit_transform(df[numeric_df.columns])
    elif cfg.SCALER == 'minmax01':
        # to be in the range 0,1
        df[numeric_df.columns] = (df[numeric_df.columns] - df[numeric_df.columns].min()) / (
                df[numeric_df.columns].max() - df[numeric_df.columns].min())
    elif cfg.SCALER == 'minmax-11':
        # to be in the range -1,1
        df[numeric_df.columns] = 2 * (df[numeric_df.columns] - df[numeric_df.columns].min()) / (
                df[numeric_df.columns].max() - df[numeric_df.columns].min()) - 1
    else:
        raise ValueError("Invalid SCALER value in config. Options are: 'standard', 'minmax01', 'minmax-11'.")
    return df


def remove_highly_correlated_features(df, correlation_matrix, threshold=0.80):
    """
    Remove highly correlated features from the dataframe based on the correlation matrix and threshold.
    1. Create an upper triangle matrix of the correlation matrix
    2. Find features with correlation greater than the threshold
    3. Drop those features from the dataframe
    4. Return the list of dropped features and the modified dataframe
    5. Note: This function does not modify the original dataframe, it returns a new
    dataframe with the highly correlated features removed.
    6. You may need to adjust the threshold based on your dataset and requirements.
    7. A threshold of 0.80 means that if two features have a correlation greater than 0.80,
    one of them will be dropped.
    8. You can experiment with different thresholds to see how it affects your model.
    9. Be cautious when setting the threshold too low, as it may lead to dropping
    too many features and losing important information.
    10. Conversely, setting it too high may not effectively reduce multicollinearity.
    11. It's often a good idea to analyze the correlation matrix and understand the relationships
    between features before deciding on a threshold.
    12. You can also consider domain knowledge and the specific context of your dataset
    when making decisions about feature selection.
    @param df: the original dataframe
    @param correlation_matrix: the correlation matrix of the dataframe
    @param threshold: the correlation threshold for dropping features
    @return: to_drop: list of dropped features
             df: modified dataframe with highly correlated features removed
    @exception None
    13. Note: This function assumes that the input dataframe and correlation matrix are valid
    and that the correlation matrix is symmetric and has the same columns as the dataframe.
    """
    upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column].abs() > cfg.CORRELATIONTHRESHOLD)]
    return to_drop, df.drop(columns=to_drop)
