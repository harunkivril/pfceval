import polars as pl
import logging
from time import time


def collect(dataframe, engine="streaming"):
    if isinstance(dataframe, pl.LazyFrame):
        start = time()
        dataframe = dataframe.collect(engine=engine)
        logging.debug(f"Lazy frame collected in {time()-start}s.")
        return dataframe
    return dataframe


def collect_all(dataframe_list, engine="streaming"):
    if isinstance(dataframe_list[0], pl.LazyFrame):
        start = time()
        dataframe_list = pl.collect_all(dataframe_list, engine=engine)
        logging.debug(f"Lazy frames collected in {time()-start}s.")
        return dataframe_list
    return dataframe_list
