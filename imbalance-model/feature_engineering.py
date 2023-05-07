import numpy as np
import pandas as pd


def floor_datetime(x: pd.Series, raster: str = "15min") -> pd.Series:
    return x.dt.tz_convert("UTC").dt.floor(raster).dt.tz_convert("Europe/Zurich")


def q90(x):
    print(len(x))
    return x.quantile(0.9)


def q10(x):
    print(len(x))
    return x.quantile(0.1)

def holiday(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(holiday=0)
    # data_ = data.set_index('ts')
    # loop over the list of dates and set weekday to zero for those dates
    for date in ['2022-10-03','2022-12-25','2022-12-26','2023-01-01']:
        mask = df['ts'].dt.date == pd.to_datetime(date).date()
        df.loc[mask, 'holiday'] = 1

    # set the weekends also as holidays
    df.loc[df['ts'].dt.dayofweek == 5, 'holiday'] = 1
    df.loc[df['ts'].dt.dayofweek == 6, 'holiday'] = 1
    return df


def aggregate_indicator(df: pd.DataFrame) -> pd.DataFrame:
    # discretise the grid into an equidistant, 1minute grid from midnight 00:00 to midnight 23:59
    t_index = pd.date_range(
        start=df["ts"].min().floor("1d"),
        end=df["ts"].max().ceil("1d") - pd.Timedelta(minutes=1), #to get 23:59 instead of 00:00 again
        freq="1min",
        name="ts",
    )
    df = df.set_index("ts").reindex(t_index).fillna(0).reset_index()
    df["ts_15min"] = floor_datetime(df["ts"])
    df = df.loc[(df["ts"] - pd.to_timedelta(5, unit="min")) <= df["ts_15min"]]
    # take the average of the one hour grouped by 15minutes
    df = (
        df.groupby("ts_15min", as_index=False)["indicator"]
        .mean()
        .rename(columns={"ts_15min": "ts"})
    )
    return df


def aggregate_frequency(df: pd.DataFrame) -> pd.DataFrame:
    df["ts_15min"] = floor_datetime(df["ts"])
    df["va"] -= 50
    df["va"] = np.where(df["va"].abs() <= 0.02, 0, df["va"])
    df = df.loc[(df["ts"] - pd.to_timedelta(5, unit="min")) <= df["ts_15min"]]
    df = (
        df.groupby("ts_15min", as_index=False)
        .agg(frequency_q90=("va", q90), frequency_q10=("va", q10))
        .rename(columns={"ts_15min": "ts"})
    )
    return df


def aggregate_wind(df: pd.DataFrame) -> pd.DataFrame:
    df["wind"] = df["wind_offshore"] + df["wind_onshore"]
    df["forecast_wind"] = df["forecast_wind_offshore"] + df["forecast_wind_onshore"]
    return df.drop(
        columns=[
            "wind_offshore",
            "wind_onshore",
            "forecast_wind_offshore",
            "forecast_wind_onshore",
        ]
    )


def shift_real_time_data(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_shift = [
        "solar",
        "wind",
        "consumption_actual",
        "prl_up",
        "prl_down",
        "afrr_up",
        "afrr_down",
        "mfrr_up",
        "mfrr_down",
        "imbalance",
    ]
    shifted_columns = [f"{x}_delayed" for x in cols_to_shift]
    df[shifted_columns] = df[cols_to_shift].shift(2)
    cols_to_shift.remove("imbalance")
    return df.drop(columns=cols_to_shift)


def fillna_with_mean(df: pd.DataFrame) -> pd.DataFrame:
    return df.fillna(df.mean())


def feature_pipeline(train_or_test: str = "train") -> pd.DataFrame:
    indicator = pd.read_parquet(f"data/{train_or_test}/indicator.parquet")
    indicator = aggregate_indicator(indicator)
    main = pd.read_parquet(f"data/{train_or_test}/main.parquet")
    main = aggregate_wind(main)
    main = shift_real_time_data(main)
    frequency = pd.read_parquet(f"data/{train_or_test}/frequency.parquet")
    frequency = aggregate_frequency(frequency)
    main = main.merge(frequency, how="left").merge(indicator, how="left")
    main = fillna_with_mean(main)

    main = holiday(main)
    return main
