import os
import logging
import tempfile

import numpy as np
import polars as pl

from time import time
from datetime import datetime
from itertools import product


def collect(dataframe, engine="streaming"):
    """
    Collects a Polars LazyFrame into a DataFrame if it is lazy.

    Args:
        dataframe (pl.DataFrame or pl.LazyFrame): The input frame to be collected.
        engine (str, optional): The engine to use for collection. Defaults to "streaming".

    Returns:
        pl.DataFrame: A collected DataFrame if the input was LazyFrame; otherwise, 
            returns the input unchanged.
    """
    if isinstance(dataframe, pl.LazyFrame):
        start = time()
        dataframe = dataframe.collect(engine=engine)
        logging.debug(f"Lazy frame collected in {time()-start}s.")
        return dataframe
    return dataframe


def collect_all(dataframe_list, engine="streaming"):
    """
    Collects a list of Polars LazyFrames into DataFrames if they are lazy.

    Args:
        dataframe_list (list): A list of LazyFrames or DataFrames.
        engine (str, optional): The engine to use for collection. Defaults to "streaming".

    Returns:
        list: A list of collected DataFrames if inputs were LazyFrames; otherwise, 
            returns the input list unchanged.
    """
    if isinstance(dataframe_list[0], pl.LazyFrame):
        start = time()
        dataframe_list = pl.collect_all(dataframe_list, engine=engine)
        logging.debug(f"Lazy frames collected in {time()-start}s.")
        return dataframe_list
    return dataframe_list


def get_example_forecast_paths(n_files=2):
    """
    Checks if example forecast files exist in a temp directory.
    If not, generates synthetic forecast data and saves them.
    Column names: model_time, valid_time, station_id, step, wind_speed, 
        unseen_sta, run_id, predq{0-9}

    Args:
        n_files (int): Number of forecast files to generate if they don't exist.
        n_rows (int): Number of rows per file.

    Returns:
        List[str]: Paths to forecast files.
    """
    # Use system temp directory
    temp_dir = os.path.join(tempfile.gettempdir(), "pfceval_example_forecasts")
    os.makedirs(temp_dir, exist_ok=True)

    file_paths = [
        os.path.join(temp_dir, f"example_forecast_{i}.parquet")
        for i in range(n_files)
    ]

    # Check if files already exist
    if all(os.path.exists(path) for path in file_paths):
        return file_paths

    # Generate synthetic forecast data
    stations = [101, 102, 103, 104, 105]
    unseen_sta = {101: True, 102: True, 103: False, 104: False, 105: False}
    latitudes = {101: 52.11, 102: 51.52, 103: 53.44, 104: 54.01, 105: 51.75}
    longitudes = {101: 5.18, 102: 4.87, 103: 6.23, 104: 5.26, 105: 4.54}
    run_ids = [1, 2, 3, 4, 5, 6, 7]
    model_run_time = pl.select(pl.date_range(
        start=datetime(year=2022, month=1, day=2), 
        end=datetime(year=2022, month=1, day= 2 + len(run_ids)), 
        interval="1d",
    )).to_numpy().squeeze()
    model_run_time = {
        run_id: run_time for run_id, run_time in zip(run_ids, model_run_time)}
    steps = list(range(1, 49))

    rows = [{
        "model_time": str(model_run_time[run_id]),
        "station_id": station_id,
        "step": step,
        "unseen_sta": unseen_sta[station_id],
        "latitude": latitudes[station_id],
        "longitude": longitudes[station_id],
        "run_id": run_id,
    } for run_id, station_id, step in product(run_ids, stations, steps)]

    df = pl.DataFrame(rows)
    df = df.with_columns(
        valid_time=(pl.col("model_time").str.to_datetime() + pl.duration(hours=pl.col("step"))),
        model_time=pl.col("model_time").str.to_datetime(),
        step=pl.duration(hours=pl.col("step")),
    )
    n_rows = df.shape[0]

    for i , path in enumerate(file_paths):
        
        df = df.with_columns(
            wind_speed=np.random.standard_normal(size=n_rows),
            **{ 
                # Add one of them bias
                f"pred_q{ens}": np.random.normal(loc=13-3*i, scale=5, size=n_rows) 
                for ens in range(50)
            }
        )
        df = df.with_columns(
            wind_speed=pl.col("wind_speed")*(pl.col("step").dt.total_hours()/5 + 3) + 10
        )
        df.write_parquet(path)

    return file_paths


def ensure_duration_lead_time(table, lead_time_col):
    if (
        lead_time_col in table.schema 
        and table.schema[lead_time_col] != pl.Duration
    ):
        table = table.with_columns(
            pl.duration(hours=pl.col(lead_time_col)).alias(lead_time_col)
        )
    return table
