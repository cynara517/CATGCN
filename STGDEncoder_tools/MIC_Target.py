import pandas as pd
import numpy as np
from minepy import MINE

def calculate_mic(df, target_column):
    """Compute MIC between each feature and the target."""
    mine = MINE()
    mic_values = {}
    for column in df.columns:
        if column != target_column:
            mine.compute_score(df[column].values, df[target_column].values)
            mic_values[column] = mine.mic()
    return mic_values

def calculate_pairwise_mic(df):
    """Compute pairwise MIC for all variable pairs."""
    mine = MINE()
    pairwise_mic_values = {}
    columns = df.columns
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            mine.compute_score(df[columns[i]].values, df[columns[j]].values)
            pairwise_mic_values[(columns[i], columns[j])] = mine.mic()
    return pairwise_mic_values

def select_top_variables(mic_values, num_variables):
    """Select top variables with highest MIC scores."""
    sorted_variables = sorted(mic_values.items(), key=lambda x: x[1], reverse=True)
    top_variables = [var for var, mic in sorted_variables[:num_variables]]
    return top_variables

# Example: Load dataset
df = pd.read_csv('/home/home_new/chensf/WorkSpace/TII/PPP/EP.csv')

# Set target column (e.g., last column)
target_column = df.columns[-1]

# Compute MIC with respect to target
mic_values = calculate_mic(df, target_column)

print("MIC values with respect to target:")
for var, mic in mic_values.items():
    print(f"{var}: {mic}")

# Optional: Compute pairwise MIC across all features
# pairwise_mic_values = calculate_pairwise_mic(df)
# print("\nPairwise MIC values:")
# for (var1, var2), mic in pairwise_mic_values.items():
#     print(f"{var1} - {var2}: {mic}")

# Optional: Select top-N features by MIC
# num_variables = 5
# top_variables = select_top_variables(mic_values, num_variables)
# print(f"\nTop {num_variables} variables: {top_variables}")
