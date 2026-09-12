import pandas as pd
import os

# 1. Load the specific required files
print("Loading required CSVs...")
DATA_DIR = 'f1_data'
laps = pd.read_csv(os.path.join(DATA_DIR, 'lap_times.csv'))
pits = pd.read_csv(os.path.join(DATA_DIR, 'pit_stops.csv'))
races = pd.read_csv(os.path.join(DATA_DIR, 'races.csv'))
results = pd.read_csv(os.path.join(DATA_DIR, 'results.csv'))

# 2. Join the base track and car identifiers
print("Merging track and constructor data...")
race_entries = results[['raceId', 'driverId', 'constructorId']].drop_duplicates()
master = pd.merge(laps, races[['raceId', 'year', 'circuitId']], on='raceId', how='left')
master = pd.merge(master, race_entries, on=['raceId', 'driverId'], how='left')

# 3. Engineer the "Tire Age" feature using pit stop data
print("Calculating dynamic tire degradation...")
# Flag the exact laps where a driver pitted
pits['is_pit_stop'] = 1
master = pd.merge(master, pits[['raceId', 'driverId', 'lap', 'is_pit_stop']], 
                  on=['raceId', 'driverId', 'lap'], how='left')
master['is_pit_stop'] = master['is_pit_stop'].fillna(0)

# Sort chronologically to ensure the rolling calculations work correctly
master = master.sort_values(by=['raceId', 'driverId', 'lap'])

# Create a 'stint' ID that increments every time a driver makes a pit stop
master['stint'] = master.groupby(['raceId', 'driverId'])['is_pit_stop'].cumsum()

# Calculate tire age: laps completed since the start of the current stint
master['tire_age'] = master.groupby(['raceId', 'driverId', 'stint']).cumcount() + 1

# 4. Final Cleanup (Strictly numeric columns for matrix operations)
print("Formatting final matrix...")
master = master[['raceId', 'circuitId', 'constructorId', 'driverId', 'year', 'lap', 'tire_age', 'milliseconds']]
master = master.dropna().astype(int) # Force integer types for memory efficiency

# Export the targeted regression dataset
out_name = 'lap_time_predictor_dataset.csv'
master.to_csv(out_name, index=False)
print(f"Success: {out_name} generated with {len(master):,} rows.")