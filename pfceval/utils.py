import polars as pl
import logging
from time import time


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
