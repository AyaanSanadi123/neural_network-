import pandas as pd

# Load the dataset
df = pd.read_csv('lap_time_predictor_dataset.csv')
original_len = len(df)

# The most accurate way to filter F1 outliers is by grouping by 'raceId'. 
# A lap of 110 seconds is normal at Spa, but a Safety Car pace at Monaco.
# We will calculate the median lap time for each race, and drop laps that are 
# more than 20% slower than the race median (which safely covers tire deg and fuel loads, 
# but eliminates crawling Safety Car laps or pit-lane waits).

# Calculate median time per race
race_medians = df.groupby('raceId')['milliseconds'].transform('median')

# Filter laps to keep only those within 120% of the median pace
df_clean = df[df['milliseconds'] <= (race_medians * 1.20)]
cleaned_len = len(df_clean)
dropped_rows = original_len - cleaned_len

# Get new max lap time
new_max = df_clean['milliseconds'].max()
new_stats = df_clean[['lap', 'tire_age', 'milliseconds']].describe().to_dict()

# Overwrite or save to a new file
df_clean.to_csv('lap_time_predictor_dataset_cleaned.csv', index=False)

print(f"Original rows: {original_len}")
print(f"Cleaned rows: {cleaned_len}")
print(f"Dropped rows (Anomalies): {dropped_rows}")
print(f"New Max ms: {new_max}")