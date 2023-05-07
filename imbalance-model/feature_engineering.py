import numpy as np
import pandas as pd


def floor_datetime(x: pd.Series, raster: str = "15min") -> pd.Series:
    return x.dt.tz_convert("UTC").dt.floor(raster).dt.tz_convert("Europe/Zurich")


def q80(x):
    return x.quantile(0.8)


def q20(x):
    return x.quantile(0.2)


def aggregate_indicator(df: pd.DataFrame) -> pd.DataFrame:
    t_index = pd.date_range(
        start=df["ts"].min().floor("1d"),
        end=df["ts"].max().ceil("1d") - pd.Timedelta(minutes=1),
        freq="1min",
        name="ts",
    )
    df = df.set_index("ts").reindex(t_index).fillna(0).reset_index()
    df["ts_15min"] = floor_datetime(df["ts"])
    df = df.loc[(df["ts"] - pd.to_timedelta(5, unit="min")) <= df["ts_15min"]]
    df = (
        df.groupby("ts_15min", as_index=False)["indicator"]
        .mean()
        .rename(columns={"ts_15min": "ts"})
    )
    return df


def aggregate_frequency(df: pd.DataFrame) -> pd.DataFrame:
    
    df["va"] -= 50
    df["va"] = np.where(df["va"].abs() <= 0.02, 0, df["va"])
    df = cyclicity_removal(df, time_unit="minute")

    df["ts_15min"] = floor_datetime(df["ts"])
    df = df.loc[(df["ts"] - pd.to_timedelta(5, unit="min")) <= df["ts_15min"]]
    df = (
        df.groupby("ts_15min", as_index=False)
        .agg(   frequency_q80=("va", q80), 
                frequency_q20=("va", q20),
                frequency_mean=("va", np.mean),
                frequency_std=("va", np.std)
            )
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

    HORIZON = 4 # x 15min
    for i in range(HORIZON):
        shifted_columns = [f"{x}_delayed_{i}" for x in cols_to_shift]
        df[shifted_columns] = df[cols_to_shift].shift(2+i)
    # Keep imbalance for the labels
    cols_to_shift.remove("imbalance")
    return df.drop(columns=cols_to_shift)


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

def fillna_with_mean(df: pd.DataFrame) -> pd.DataFrame:
    return df.fillna(df.mean())

def temporal_encoding(df: pd.DataFrame) -> pd.DataFrame:
    df["hour"] = df["ts"].dt.hour
    df["minute"] = df["ts"].dt.minute
    df["dayofweek"] = df["ts"].dt.dayofweek
    return df

def cyclicity_removal(df: pd.DataFrame, time_unit = ['hour','minute']) -> pd.DataFrame:
    # get the 
    if type(time_unit) == list:
        for time in time_unit:
            df[time] = getattr(df['ts'].dt, time)
    elif type(time_unit) == str:
        df[time_unit] = getattr(df['ts'].dt, time_unit)
    else:
        raise("ERROR: time_unit must be either a string of list of strings")

    # get the average for each time of the day
    df_ave_time = df.groupby(time_unit).mean()
    df_ave_time = df_ave_time.reset_index()
    # divide the data by its mean
    merged_df = pd.merge(df, df_ave_time, on=time_unit)
    merged_df= merged_df.set_index('ts_x').drop(columns='ts_y')
    cols_x = merged_df.filter(like='_x').columns
    cols_y = merged_df.filter(like='_y').columns
    eps = 1e-1
    merged_df[cols_x] = merged_df[cols_x].div(merged_df[cols_y].values+eps)
    output_df = merged_df.drop(columns=cols_y).reset_index()
    names_without_x = {col: col.replace('_x', '') for col in output_df.columns}
    output_df = output_df.rename(columns=names_without_x)
    output_df = output_df.drop(columns = time_unit)

    keep_original_columns = output_df.filter(like='imbalance').columns
    output_df[keep_original_columns] = df[keep_original_columns]

    return output_df

def feature_pipeline(train_or_test: str = "train") -> pd.DataFrame:
    indicator = pd.read_parquet(f"../{train_or_test}/indicator.parquet")
    indicator = aggregate_indicator(indicator)
    main = pd.read_parquet(f"../{train_or_test}/main.parquet")
    main = aggregate_wind(main)
    main = shift_real_time_data(main)
    
    main = cyclicity_removal(main)
    main = cyclicity_removal(main, time_unit="dayofweek")
    
    main = holiday(main)

    frequency = pd.read_parquet(f"../{train_or_test}/frequency.parquet")
    frequency = aggregate_frequency(frequency)
    main = main.merge(frequency, how="left").merge(indicator, how="left")
    main = fillna_with_mean(main)
    
    main = temporal_encoding(main)
    return main
