import pandas as pd
import openpyxl
import numpy as np


#!#########################################################################################################
#!MUUNA AÐ NOTA RÉTT GÖGN Í ÞESSI
#!#########################################################################################################
fyrirtaeki = pd.read_excel("gogn (1).xlsx", sheet_name="Sheet4", na_values=["#N/A N/A", "#N/A"])
# fyrirtaeki = pd.read_excel("//center1.ad.local/dfs$/IS/RVK/Desktop02/sigurdurbl/Desktop/Lokaverkefni/lokaverkefni/gogn (1).xlsx", sheet_name="Sheet4", na_values=["#N/A N/A", "#N/A"])
fyrirtaeki.set_index("Dates", inplace=True)
indices = pd.read_excel("gogn (1).xlsx", sheet_name="Indices", na_values=["#N/A N/A", "#N/A"])
indices.set_index("Dates", inplace=True)
iceland_indices = pd.read_excel("gogn (1).xlsx", sheet_name="Sheet3", na_values=["#N/A N/A", "#N/A"])
iceland_indices.set_index("Dates", inplace=True)


fyrirtaeki.index = pd.to_datetime(fyrirtaeki.index)
indices.index = pd.to_datetime(indices.index)
iceland_indices.index = pd.to_datetime(iceland_indices.index)

fyrirtaeki_before_2014 = fyrirtaeki[fyrirtaeki.index < '2014-01-01']
fyrirtaeki_2014 = fyrirtaeki[fyrirtaeki.index >= '2014-01-01']

indices_before_2014 = indices[indices.index < '2014-01-01']
indices_2014 = indices[indices.index >= '2014-01-01']

iceland_indices_before_2014 = iceland_indices[iceland_indices.index < '2014-01-01']
iceland_indices_2014 = iceland_indices[iceland_indices.index >= '2014-01-01']

def calculate_monthly_returns(data):
    monthly_returns = pd.DataFrame()
    for column in data.columns:
        monthly_returns[column] = data[column].pct_change(fill_method=None)
    return monthly_returns

fyrirtaeki_before_2014_returns = calculate_monthly_returns(fyrirtaeki_before_2014)
fyrirtaeki_2014_returns = calculate_monthly_returns(fyrirtaeki_2014)
indices_before_2014_returns = calculate_monthly_returns(indices_before_2014)
indices_2014_returns = calculate_monthly_returns(indices_2014)
iceland_indices_before_2014_returns = calculate_monthly_returns(iceland_indices_before_2014)
iceland_indices_2014_returns = calculate_monthly_returns(iceland_indices_2014)

def calculate_correlation(fyrirtaeki_data, iceland_indices_data):
    # Ensure the index is in datetime format if it's not already
    fyrirtaeki_data.index = pd.to_datetime(fyrirtaeki_data.index)
    iceland_indices_data.index = pd.to_datetime(iceland_indices_data.index)
    
    # Focus on the OMXIGI Index PX_LAST column from the iceland_indices dataset
    omxigi_px_last = iceland_indices_data['OMXIGI Index PX_LAST']
    
    # Initialize an empty list to store correlation results along with observation counts
    correlation_results = []
    
    # Iterate through each column in the fyrirtaeki_data dataframe
    for column in fyrirtaeki_data.columns:
        # Calculate the correlation only for overlapping dates
        common_index = fyrirtaeki_data.index.intersection(omxigi_px_last.index)
        
        # Ensure there is enough data to calculate a correlation
        if len(common_index) > 1:
            # Subset the data to only those dates that exist in both dataframes and drop NaN values
            fyrirtaeki_sub = fyrirtaeki_data.loc[common_index, column].dropna()
            omxigi_sub = omxigi_px_last.loc[common_index].dropna()
            
            # Find common index after dropping NaN to ensure both series have the same dates
            common_index_no_nan = fyrirtaeki_sub.index.intersection(omxigi_sub.index)
            fyrirtaeki_sub = fyrirtaeki_sub.loc[common_index_no_nan]
            omxigi_sub = omxigi_sub.loc[common_index_no_nan]
            
            # Check again if there's enough data after removing NaNs
            if len(fyrirtaeki_sub) > 1:
                # Calculate correlation
                correlation = fyrirtaeki_sub.corr(omxigi_sub)
                
                # Append the column name, correlation, and count of non-NaN observations to the list
                correlation_results.append({
                    'Column': column,
                    'Correlation': correlation,
                    'Count': len(fyrirtaeki_sub) + 1
                })
            else:
                # If not enough data after dropping NaNs, indicate this in the results
                correlation_results.append({
                    'Column': column,
                    'Correlation': 'Not enough data',
                    'Count': 0
                })
        else:
            # If not enough initial data, indicate this in the results
            correlation_results.append({
                'Column': column,
                'Correlation': 'Not enough initial data for correlation',
                'Count': len(common_index)
            })
    
    # Convert the list of dictionaries to a DataFrame
    result_df = pd.DataFrame(correlation_results)
    return result_df

# Calculate correlations for both time periods
correlation_before_2014 = calculate_correlation(fyrirtaeki_before_2014_returns, iceland_indices_before_2014_returns) #? Important variable
correlation_2014 = calculate_correlation(fyrirtaeki_2014_returns, iceland_indices_2014_returns)#? Important variable


#! Ask fink if it is ok to have monthly data for this
def volatility(data):
    result = pd.DataFrame()
    vol = data.std() * np.sqrt(12)
    # Convert the series to DataFrame for consistency with your original function's return type
    result = vol.to_frame(name='Volatility')
    return result

fyrirtaeki_before_2014_vol = volatility(fyrirtaeki_before_2014_returns) #? Important variable
fyrirtaeki_2014_vol = volatility(fyrirtaeki_2014_returns) #? Important variable
indices_before_2014_vol = volatility(indices_before_2014_returns) #? Important variable
indices_2014_vol = volatility(indices_2014_returns)#? Important variable
iceland_indices_before_2014_vol = volatility(iceland_indices_before_2014_returns)#? Important variable
iceland_indices_2014_vol = volatility(iceland_indices_2014_returns)#? Important variable

# with pd.ExcelWriter('correlation_results.xlsx', engine='openpyxl') as writer:
#     correlation_before_2014.to_excel(writer, sheet_name='Before 2014')
#     correlation_2014.to_excel(writer, sheet_name='2014')

# with pd.ExcelWriter('volatility_result1.xlsx', engine='openpyxl') as writer:
#     fyrirtaeki_2014_vol.to_excel(writer, sheet_name='2014')
#     fyrirtaeki_before_2014_vol.to_excel(writer, sheet_name='Before 2014')
#     iceland_indices_2014_vol.to_excel(writer, sheet_name = "2014 1")
#     iceland_indices_before_2014_vol.to_excel(writer, sheet_name = "2014 2")

def calculate_covariance(fyrirtaeki_data, iceland_indices_data):
    omxigi_px_last = iceland_indices_data['OMXIGI Index PX_LAST']
    
    covariance_results = []
    
    for column in fyrirtaeki_data.columns:
        # Calculate the covariance only for overlapping dates
        common_index = fyrirtaeki_data.index.intersection(omxigi_px_last.index)
        
        if len(common_index) > 1:
            fyrirtaeki_sub = fyrirtaeki_data.loc[common_index, column].dropna()
            omxigi_sub = omxigi_px_last.loc[common_index].dropna()
            
            # Find common index after dropping NaN to ensure both series have the same dates
            common_index_no_nan = fyrirtaeki_sub.index.intersection(omxigi_sub.index)
            fyrirtaeki_sub = fyrirtaeki_sub.loc[common_index_no_nan] #Locate W
            omxigi_sub = omxigi_sub.loc[common_index_no_nan]
            
            # Check again if there's enough data after removing NaNs
            if len(fyrirtaeki_sub) > 1:
                # Calculate covariance
                covariance = fyrirtaeki_sub.cov(omxigi_sub)
                
                # Append the column name and covariance to the list
                covariance_results.append({
                    'Column': column,
                    'Covariance': covariance
                })
            else:
                # If not enough data after dropping NaNs, indicate this in the results
                covariance_results.append({
                    'Column': column,
                    'Covariance': 'Not enough data'
                })
        else:
            # If not enough initial data, indicate this in the results
            covariance_results.append({
                'Column': column,
                'Covariance': 'Not enough initial data for covariance'
            })
    
    # Convert the list of dictionaries to a DataFrame
    result_df = pd.DataFrame(covariance_results)
    return result_df


all_iceland = pd.concat([fyrirtaeki_before_2014, fyrirtaeki_2014], axis=0)
all_iceland_ind = pd.concat([iceland_indices_before_2014, iceland_indices_2014], axis=0)
all_iceland_indices = calculate_monthly_returns(all_iceland_ind)
all_iceland_stocks = calculate_monthly_returns(all_iceland)
correlation = calculate_correlation(all_iceland_stocks, all_iceland_indices) #? Important variable
covar_before_2014 = calculate_covariance(fyrirtaeki_before_2014_returns, iceland_indices_before_2014_returns) #? Important variable
covar_2014 = calculate_covariance(fyrirtaeki_2014_returns, iceland_indices_2014_returns) #? Important variable
covariance = calculate_covariance(all_iceland_stocks, all_iceland_indices) #? Important variable


# with pd.ExcelWriter('fm.xlsx', engine='openpyxl') as writer:
#     all_iceland_indices.to_excel(writer, sheet_name='Before 2014')
#     all_iceland_stock.to_excel(writer, sheet_name='2014')

#BETA
def beta(fyrirtaeki_data, market_data):
    omxigi_px_last = market_data['OMXIGI Index PX_LAST']
    beta_result = []

    for column in fyrirtaeki_data.columns:
        # Combine stock and market data into a single DataFrame for overlapping dates
        combined_data = pd.DataFrame({
            'stock': fyrirtaeki_data[column],
            'market': omxigi_px_last
        }).dropna()  # Drop rows where either stock or market data is NaN

        # Ensure there's enough data after dropping NaNs
        if len(combined_data) > 1:
            # Calculate covariance between the stock and market returns
            covariance = combined_data['stock'].cov(combined_data['market'])
            
            # Calculate variance of the market for the same period
            variance = combined_data['market'].var()

            # Calculate the beta using the calculated variance of the market
            beta_value = covariance / variance

            beta_result.append({
                'Column': column,
                'Beta': beta_value
            })
        else:
            # If not enough data after dropping NaNs, indicate this in the results
            beta_result.append({
                'Column': column,
                'Beta': 'Not enough data'
            })

    # Convert the list of dictionaries to a DataFrame
    result_df = pd.DataFrame(beta_result)
    return result_df

  
betas = beta(all_iceland_stocks, all_iceland_indices) #? Important variable
for idx, row in betas.iterrows():
    print(row['Column'], 'beta', row['Beta'])
