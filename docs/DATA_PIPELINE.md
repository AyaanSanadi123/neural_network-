# Data Pipeline

The engine trains on a single flat CSV, `data/lap_time_predictor_dataset_cleaned.csv`. This document explains where that file comes from, what each column means, and how to rebuild it.

## Source data

`data/f1_data/` contains the Formula 1 results archive as a set of relational CSV tables (the Ergast-derived bundle commonly distributed on Kaggle as "Formula 1 World Championship"). Fourteen tables are included; the pipeline uses four of them:

| Table | Rows | Used columns |
| --- | --- | --- |
| `lap_times.csv` | 589,081 | `raceId, driverId, lap, milliseconds` |
| `pit_stops.csv` | 11,371 | `raceId, driverId, lap` |
| `races.csv` | 1,125 | `raceId, year, circuitId` |
| `results.csv` | 26,759 | `raceId, driverId, constructorId` |

The remaining tables (`circuits`, `constructors`, `constructor_results`, `constructor_standings`, `drivers`, `driver_standings`, `qualifying`, `seasons`, `sprint_results`, `status`) are kept for future feature engineering but are not read by any script.

## Step 1: `data/cleaner.py`, join and feature engineering

Run from inside `data/` (paths are relative):

```bash
cd data
python cleaner.py
```

What it does:

1. **Load** `lap_times`, `pit_stops`, `races`, `results`.
2. **Attach race context**: left-join `lap_times` with `races` on `raceId` to get `year` and `circuitId`.
3. **Attach the car**: build a unique `(raceId, driverId, constructorId)` table from `results` and left-join it on `(raceId, driverId)`.
4. **Flag pit laps**: left-join `pit_stops` on `(raceId, driverId, lap)` and set `is_pit_stop = 1` on matching laps, `0` elsewhere.
5. **Sort** by `raceId, driverId, lap` so the running calculations below are chronological.
6. **Stint number**: `stint = cumulative sum of is_pit_stop` within each `(raceId, driverId)`. It increments every time the driver pits.
7. **Tire age**: `tire_age = position of the lap within its (raceId, driverId, stint) group, starting at 1`. This is the number of laps completed on the current set of tires and is the key engineered feature.
8. **Select and clean**: keep `raceId, circuitId, constructorId, driverId, year, lap, tire_age, milliseconds`, drop rows with any missing value, cast everything to `int`.
9. **Write** `lap_time_predictor_dataset.csv` (this intermediate file is **not** committed).

## Step 2: `data/cleaner1.py`, outlier removal

```bash
python cleaner1.py
```

Lap times are heavily skewed by safety-car periods, red flags and in-lap/out-lap pit-lane time. Because "slow" is track-relative (a normal lap at Spa is far longer than at Monaco), the filter is applied **per race**:

1. Compute the median `milliseconds` of every race.
2. Keep only laps whose time is at most **120%** of their race's median.
3. Write `lap_time_predictor_dataset_cleaned.csv` and print how many rows were dropped.

The 20% margin is loose enough to keep genuine tire degradation and fuel-load effects but removes crawling laps.

## Final dataset schema

`data/lap_time_predictor_dataset_cleaned.csv`, 550,206 samples plus a header row:

| # | Column | Role | Meaning |
| --- | --- | --- | --- |
| 1 | `raceId` | input | Identifier of the Grand Prix (from `races.csv`) |
| 2 | `circuitId` | input | Identifier of the track |
| 3 | `constructorId` | input | Identifier of the team / car |
| 4 | `driverId` | input | Identifier of the driver |
| 5 | `year` | input | Season |
| 6 | `lap` | input | Lap number within the race |
| 7 | `tire_age` | input | Laps completed on the current tire set (engineered) |
| 8 | `milliseconds` | **target** | Lap time in milliseconds |

The C loader is told there are 7 input columns and 1 target (`create_dataset(rows, 7, 1)` in `main.c`), so the column order above is load-bearing.

Note that the first five inputs are **categorical identifiers** that the network currently consumes as raw numbers after min-max scaling. See [FEATURES.md](FEATURES.md) for why this limits the achievable accuracy and what could be done about it.

## How the C engine consumes it

1. `count_csv_dimensions` finds 550,206 rows and 8 columns.
2. `load_csv` parses each line into a `(7×1)` input and `(1×1)` target matrix.
3. `split_dataset_sequential(dataset, 0.7, 0.2, 0.1)` takes the file **in order**:

   | Split | Samples | Position in file |
   | --- | --- | --- |
   | train | 385,144 | first 70% |
   | test | 110,041 | next 20% |
   | validation | 55,021 | last 10% |

   Because the file is sorted by `raceId`, this is effectively a **chronological split**: the model trains on earlier races and is validated/tested on later ones. That is a reasonable choice for a forecasting task, but it also means the distributions differ (newer cars, new circuits, drivers not seen in training).
4. `normalize_dataset` min-max scales all 7 inputs and the target using the training split's ranges.

## Rebuilding from scratch

```bash
# from the repository root
python3 -m venv myenv
source myenv/bin/activate        # Windows: myenv\Scripts\activate
pip install pandas
cd data
python cleaner.py
python cleaner1.py
cd ..
```

Requirements: Python 3.9 or newer and pandas (any recent version; the committed virtualenv used pandas 3.0 on Python 3.14). The scripts take under a minute.

To refresh the raw tables from Kaggle, download the dataset bundle and unzip its CSVs into `data/f1_data/`, keeping the file names above. The Kaggle CLI (`pip install kaggle`) is a convenient way to do that.

## Adapting the pipeline

- **Add a feature**: compute it in `cleaner.py` before the column selection on the final line of step 8, add it to the selected column list **before** `milliseconds`, then update `create_dataset(rows, N, 1)` and the first layer's input size in `main.c`.
- **Change the outlier threshold**: edit the `1.20` multiplier in `cleaner1.py`.
- **Shuffle instead of chronological split**: add `df_clean = df_clean.sample(frac=1, random_state=0)` before writing in `cleaner1.py`. The C engine will then see a random split.
