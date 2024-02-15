import pandas as pd
import openpyxl
import numpy as np

#Vantar gogn fra indices

fyrirtaeki = pd.read_excel("lokaverkefni/gogn.xlsx", sheet_name="sheet 4", index_col="Dates", na_values=["#N/A N/A", "#N/A"])
indices = pd.read_excel("lokaverkefni/gogn.xlsx", sheet_name="indices", index_col="Dates", na_values=["#N/A N/A", "#N/A"])
iceland_indices = pd.read_excel("lokaverkefni/gogn.xlsx", sheet_name="sheet 3", index_col="Dates", na_values=["#N/A N/A", "#N/A"])

fyrirtaeki.index = pd.to_datetime(fyrirtaeki.index)
indices.index = pd.to_datetime(indices.index)
iceland_indices.index = pd.to_datetime(iceland_indices.index)

fyrirtaeki_before_2014 = fyrirtaeki[fyrirtaeki.index < '2014-01-01']
fyrirtaeki_2014 = fyrirtaeki[fyrirtaeki.index >= '2014-01-01']

indices_before_2014 = indices[indices.index < '2014-01-01']
indices_2014 = indices[indices.index >= '2014-01-01']

iceland_indices_before_2014 = iceland_indices[iceland_indices.index < '2014-01-01']
iceland_indices_2014 = iceland_indices[iceland_indices.index >= '2014-01-01']

def calculate_correlation(fyrirtaeki_data, iceland_indices_data):
    # Ensure the index is in datetime format if it's not already
    fyrirtaeki_data.index = pd.to_datetime(fyrirtaeki_data.index)
    iceland_indices_data.index = pd.to_datetime(iceland_indices_data.index)
    
    # Focus on the OMXIGI Index PX_LAST column from the iceland_indices dataset
    omxigi_px_last = iceland_indices_data['OMXIGI Index PX_LAST']
    
    # Initialize an empty dictionary to store correlation results
    correlation_results = {}
    
    # Iterate through each column in the fyrirtaeki_data dataframe
    for column in fyrirtaeki_data.columns:
        # Calculate the correlation only for overlapping dates
        common_index = fyrirtaeki_data.index.intersection(omxigi_px_last.index)
        
        # Ensure there is enough data to calculate a correlation
        if len(common_index) > 1:
            # Subset the data to only those dates that exist in both dataframes
            fyrirtaeki_sub = fyrirtaeki_data.loc[common_index, column]
            omxigi_sub = omxigi_px_last.loc[common_index]
            
            # Calculate correlation and store it in the dictionary
            correlation = fyrirtaeki_sub.corr(omxigi_sub)
            correlation_results[column] = correlation
        else:
            # If not enough data, indicate this in the results
            correlation_results[column] = 'Not enough data for correlation'
    
    return correlation_results

# Calculate correlations for both time periods
correlation_before_2014 = calculate_correlation(fyrirtaeki_before_2014, iceland_indices_before_2014)
correlation_2014_onwards = calculate_correlation(fyrirtaeki_2014, iceland_indices_2014)

# You can then print or further analyze these correlations
print("Correlation Before 2014:")
print(correlation_before_2014)
print("\nCorrelation 2014 Onwards:")
print(correlation_2014_onwards)