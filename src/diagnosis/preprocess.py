import pandas as pd
import numpy as np
import glob
import os

def preprocess(csv_file):
    # Read the CSV file into a dataframe
    df = pd.read_csv(csv_file)

    # Extract the values from the first column using str.split and keep only the first part
    df.iloc[:, 0] = df.iloc[:, 0].str.split("_", n=1).str[0]

    # Convert all values to float and handle invalid values
    for column in df.columns:
        try:
            df[column] = df[column].astype(float)
        except ValueError:
            df[column] = pd.to_numeric(df[column], errors='coerce')

    # Drop columns containing "Unnamed"
    df = df.loc[:, ~df.columns.str.contains("Unnamed")]

    # Remove empty columns
    df = df.dropna(axis=1, how='all')

    # Replace "Factor1:CASE" with 1 and "Factor1:CONTROL" with 0 in the dataframe
    df.replace({"Factor1:CASE": "1", "Factor1:CONTROL": "0"}, inplace=True)

    return df

def replace_in_file(file_path, search_text, replace_text):
    with open(file_path, 'r') as file:
        filedata = file.read()

    newdata = filedata.replace(search_text, replace_text)

    with open(file_path, 'w') as file:
        file.write(newdata)

folder_path = "../data/patient1"
csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

# List to store individual dataframes
dfs = []

for i, csv_file in enumerate(csv_files, 1):
    df = preprocess(csv_file)
    output_file_path = f"Processed{i}.csv"
    df.to_csv(output_file_path, index=False)
    print(f"Updated dataframe has been saved to {output_file_path}")

    # Append the processed dataframe to the list
    df = pd.read_csv(output_file_path, skiprows=1)
    dfs.append(df)

# Combine the dataframes vertically (row-wise)
combined_df = pd.concat(dfs)

# Save the combined dataframe to a new CSV file
output_file_path = "CombinedResults.csv"
combined_df.to_csv(output_file_path, index=False)
print(f"Combined dataframe has been saved to {output_file_path}")

# Sort and reset the index of the combined dataframe
combined_df = pd.read_csv("CombinedResults.csv")
combined_df = combined_df.sort_values(by=combined_df.columns[0], ascending=True)
combined_df = combined_df.reset_index(drop=True)

# Save the sorted dataframe to a new CSV file
sorted_output_file_path = "CombinedResultsSorted.csv"
combined_df.to_csv(sorted_output_file_path, index=False)
print(f"Combined dataframe has been sorted and saved to {sorted_output_file_path}")

# Transpose the dataframe and save it to a new CSV file
df = pd.read_csv('CombinedResultsSorted.csv')
df_transposed = df.transpose()
df_transposed.to_csv('Transposed.csv', header=False)

# Replace values starting with '1' or '0' in the first column
df = pd.read_csv('Transposed.csv')
df['Unnamed: 0'] = df['Unnamed: 0'].apply(lambda x: '1' if str(x).startswith('1') else x)
df['Unnamed: 0'] = df['Unnamed: 0'].apply(lambda x: '0' if str(x).startswith('0') else x)

# Replace the cell at column 1, row 1 with "PD"
df.rename(columns={df.columns[0]: 'PD'}, inplace=True)

# Save the final dataframe to a CSV file
df.to_csv('Final.csv', index=False)
print(f"Final dataframe has been saved to Final.csv")
