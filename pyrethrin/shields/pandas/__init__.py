"""Pyrethrin shields for Pandas.

This module provides shielded versions of Pandas operations that can fail at
runtime due to data-dependent conditions (file I/O, parsing, etc.).

Shield version: 0.1.0
Pandas version: 2.3.3

Usage:
    from pyrethrin.shields.pandas import read_csv, DataFrame
    from pyrethrin import Ok, Err

    # Safe file reading
    match read_csv("data.csv"):
        case Ok(df):
            # use df
        case Err(OSError() as e):
            print(f"File error: {e}")
        case Err(ParserError() as e):
            print(f"Parse error: {e}")
"""

from __future__ import annotations

__shield_version__ = "0.1.0"
__pandas_version__ = "2.3.3"

from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping, Sequence
from typing import Any, Literal

import pandas as _pd

from pyrethrin.decorators import raises

# =============================================================================
# EXCEPTIONS - Re-export pandas exceptions
# =============================================================================

from pandas.errors import (
    AbstractMethodError as AbstractMethodError,
    DuplicateLabelError as DuplicateLabelError,
    EmptyDataError as EmptyDataError,
    IntCastingNaNError as IntCastingNaNError,
    InvalidIndexError as InvalidIndexError,
    MergeError as MergeError,
    NullFrequencyError as NullFrequencyError,
    NumbaUtilError as NumbaUtilError,
    OptionError as OptionError,
    OutOfBoundsDatetime as OutOfBoundsDatetime,
    OutOfBoundsTimedelta as OutOfBoundsTimedelta,
    ParserError as ParserError,
    PerformanceWarning as PerformanceWarning,
    UndefinedVariableError as UndefinedVariableError,
    UnsortedIndexError as UnsortedIndexError,
    UnsupportedFunctionCall as UnsupportedFunctionCall,
)

# =============================================================================
# CORE DATA STRUCTURES - Re-export unchanged
# =============================================================================

from pandas import DataFrame as DataFrame
from pandas import Series as Series
from pandas import Index as Index
from pandas import RangeIndex as RangeIndex
from pandas import MultiIndex as MultiIndex
from pandas import CategoricalIndex as CategoricalIndex
from pandas import DatetimeIndex as DatetimeIndex
from pandas import TimedeltaIndex as TimedeltaIndex
from pandas import PeriodIndex as PeriodIndex
from pandas import IntervalIndex as IntervalIndex
from pandas import Categorical as Categorical
from pandas import Grouper as Grouper
from pandas import NamedAgg as NamedAgg
from pandas import Flags as Flags

# =============================================================================
# DTYPE TYPES - Re-export unchanged
# =============================================================================

from pandas import ArrowDtype as ArrowDtype
from pandas import BooleanDtype as BooleanDtype
from pandas import CategoricalDtype as CategoricalDtype
from pandas import DatetimeTZDtype as DatetimeTZDtype
from pandas import Float32Dtype as Float32Dtype
from pandas import Float64Dtype as Float64Dtype
from pandas import Int8Dtype as Int8Dtype
from pandas import Int16Dtype as Int16Dtype
from pandas import Int32Dtype as Int32Dtype
from pandas import Int64Dtype as Int64Dtype
from pandas import PeriodDtype as PeriodDtype
from pandas import SparseDtype as SparseDtype
from pandas import StringDtype as StringDtype
from pandas import UInt8Dtype as UInt8Dtype
from pandas import UInt16Dtype as UInt16Dtype
from pandas import UInt32Dtype as UInt32Dtype
from pandas import UInt64Dtype as UInt64Dtype
from pandas import IntervalDtype as IntervalDtype

# =============================================================================
# SPECIAL VALUES - Re-export unchanged
# =============================================================================

from pandas import NA as NA
from pandas import NaT as NaT
from pandas import Timestamp as Timestamp
from pandas import Timedelta as Timedelta
from pandas import Period as Period
from pandas import Interval as Interval
from pandas import DateOffset as DateOffset

# =============================================================================
# UTILITY CLASSES - Re-export unchanged
# =============================================================================

from pandas import ExcelFile as ExcelFile
from pandas import ExcelWriter as ExcelWriter
from pandas import HDFStore as HDFStore
from pandas import IndexSlice as IndexSlice

# =============================================================================
# RANGE GENERATORS - Re-export unchanged
# =============================================================================

from pandas import date_range as date_range
from pandas import bdate_range as bdate_range
from pandas import period_range as period_range
from pandas import timedelta_range as timedelta_range
from pandas import interval_range as interval_range

# =============================================================================
# DATA MANIPULATION - Re-export unchanged (some shielded below)
# =============================================================================

# concat, merge, pivot, cut, qcut are shielded below
from pandas import merge_asof as merge_asof
from pandas import merge_ordered as merge_ordered
from pandas import pivot_table as pivot_table
from pandas import melt as melt
from pandas import lreshape as lreshape
from pandas import wide_to_long as wide_to_long
from pandas import crosstab as crosstab
from pandas import factorize as factorize
from pandas import get_dummies as get_dummies
from pandas import from_dummies as from_dummies
from pandas import unique as unique
from pandas import value_counts as value_counts

# =============================================================================
# NULL CHECKING - Re-export unchanged
# =============================================================================

from pandas import isna as isna
from pandas import isnull as isnull
from pandas import notna as notna
from pandas import notnull as notnull

# =============================================================================
# ARRAY CREATION - Re-export unchanged
# =============================================================================

from pandas import array as array

# =============================================================================
# FREQUENCY INFERENCE - Re-export unchanged
# =============================================================================

from pandas import infer_freq as infer_freq

# =============================================================================
# JSON NORMALIZATION - Shielded below
# =============================================================================

# json_normalize is shielded below

# =============================================================================
# OPTIONS - Re-export unchanged
# =============================================================================

from pandas import options as options
from pandas import get_option as get_option
from pandas import set_option as set_option
from pandas import reset_option as reset_option
from pandas import describe_option as describe_option
from pandas import option_context as option_context

# =============================================================================
# DISPLAY - Re-export unchanged
# =============================================================================

from pandas import set_eng_float_format as set_eng_float_format
from pandas import show_versions as show_versions

# =============================================================================
# SUBMODULES - Re-export unchanged
# =============================================================================

from pandas import api as api
from pandas import arrays as arrays
from pandas import errors as errors
from pandas import io as io
from pandas import offsets as offsets
from pandas import plotting as plotting
from pandas import testing as testing
from pandas import tseries as tseries

# =============================================================================
# FILE I/O - Shielded functions (can fail due to file/parsing errors)
# =============================================================================


@raises(OSError, ParserError, EmptyDataError, ValueError, TypeError, KeyError, UnicodeDecodeError)
def read_csv(
    filepath_or_buffer: str | Any,
    *,
    sep: str | None = None,
    delimiter: str | None = None,
    header: int | Sequence[int] | Literal["infer"] | None = "infer",
    names: Sequence[Hashable] | None = None,
    index_col: int | str | Sequence[int | str] | Literal[False] | None = None,
    usecols: Sequence[Hashable] | Callable | None = None,
    dtype: dict | None = None,
    engine: Literal["c", "python", "pyarrow"] | None = None,
    converters: dict | None = None,
    true_values: list | None = None,
    false_values: list | None = None,
    skipinitialspace: bool = False,
    skiprows: int | Sequence[int] | Callable | None = None,
    skipfooter: int = 0,
    nrows: int | None = None,
    na_values: Sequence[str] | Mapping[str, Sequence[str]] | None = None,
    keep_default_na: bool = True,
    na_filter: bool = True,
    skip_blank_lines: bool = True,
    parse_dates: bool | Sequence[int] | Sequence[str] | Sequence[Sequence] | dict | None = None,
    date_format: str | dict | None = None,
    dayfirst: bool = False,
    cache_dates: bool = True,
    iterator: bool = False,
    chunksize: int | None = None,
    compression: str | dict | None = "infer",
    thousands: str | None = None,
    decimal: str = ".",
    lineterminator: str | None = None,
    quotechar: str = '"',
    quoting: int = 0,
    doublequote: bool = True,
    escapechar: str | None = None,
    comment: str | None = None,
    encoding: str | None = None,
    encoding_errors: str | None = "strict",
    dialect: str | None = None,
    on_bad_lines: Literal["error", "warn", "skip"] | Callable = "error",
    low_memory: bool = True,
    memory_map: bool = False,
    float_precision: Literal["high", "legacy", "round_trip"] | None = None,
    storage_options: dict | None = None,
    dtype_backend: Literal["numpy_nullable", "pyarrow"] = "numpy_nullable",
) -> _pd.DataFrame:
    """Read a comma-separated values (csv) file into DataFrame.

    Returns Result[DataFrame, OSError | ParserError | EmptyDataError | ValueError | TypeError | KeyError | UnicodeDecodeError]
    """
    return _pd.read_csv(
        filepath_or_buffer,
        sep=sep,
        delimiter=delimiter,
        header=header,
        names=names,
        index_col=index_col,
        usecols=usecols,
        dtype=dtype,
        engine=engine,
        converters=converters,
        true_values=true_values,
        false_values=false_values,
        skipinitialspace=skipinitialspace,
        skiprows=skiprows,
        skipfooter=skipfooter,
        nrows=nrows,
        na_values=na_values,
        keep_default_na=keep_default_na,
        na_filter=na_filter,
        skip_blank_lines=skip_blank_lines,
        parse_dates=parse_dates,
        date_format=date_format,
        dayfirst=dayfirst,
        cache_dates=cache_dates,
        iterator=iterator,
        chunksize=chunksize,
        compression=compression,
        thousands=thousands,
        decimal=decimal,
        lineterminator=lineterminator,
        quotechar=quotechar,
        quoting=quoting,
        doublequote=doublequote,
        escapechar=escapechar,
        comment=comment,
        encoding=encoding,
        encoding_errors=encoding_errors,
        dialect=dialect,
        on_bad_lines=on_bad_lines,
        low_memory=low_memory,
        memory_map=memory_map,
        float_precision=float_precision,
        storage_options=storage_options,
        dtype_backend=dtype_backend,
    )


@raises(OSError, ParserError, EmptyDataError, ValueError, TypeError, KeyError, UnicodeDecodeError)
def read_table(
    filepath_or_buffer: str | Any,
    *,
    sep: str = "\t",
    delimiter: str | None = None,
    header: int | Sequence[int] | Literal["infer"] | None = "infer",
    names: Sequence[Hashable] | None = None,
    index_col: int | str | Sequence[int | str] | Literal[False] | None = None,
    usecols: Sequence[Hashable] | Callable | None = None,
    dtype: dict | None = None,
    engine: Literal["c", "python", "pyarrow"] | None = None,
    converters: dict | None = None,
    true_values: list | None = None,
    false_values: list | None = None,
    skipinitialspace: bool = False,
    skiprows: int | Sequence[int] | Callable | None = None,
    skipfooter: int = 0,
    nrows: int | None = None,
    na_values: Sequence[str] | Mapping[str, Sequence[str]] | None = None,
    keep_default_na: bool = True,
    na_filter: bool = True,
    skip_blank_lines: bool = True,
    parse_dates: bool | Sequence[int] | Sequence[str] | Sequence[Sequence] | dict | None = None,
    date_format: str | dict | None = None,
    dayfirst: bool = False,
    cache_dates: bool = True,
    iterator: bool = False,
    chunksize: int | None = None,
    compression: str | dict | None = "infer",
    thousands: str | None = None,
    decimal: str = ".",
    lineterminator: str | None = None,
    quotechar: str = '"',
    quoting: int = 0,
    doublequote: bool = True,
    escapechar: str | None = None,
    comment: str | None = None,
    encoding: str | None = None,
    encoding_errors: str | None = "strict",
    dialect: str | None = None,
    on_bad_lines: Literal["error", "warn", "skip"] | Callable = "error",
    low_memory: bool = True,
    memory_map: bool = False,
    float_precision: Literal["high", "legacy", "round_trip"] | None = None,
    storage_options: dict | None = None,
    dtype_backend: Literal["numpy_nullable", "pyarrow"] = "numpy_nullable",
) -> _pd.DataFrame:
    """Read general delimited file into DataFrame.

    Returns Result[DataFrame, OSError | ParserError | EmptyDataError | ValueError | TypeError | KeyError | UnicodeDecodeError]
    """
    return _pd.read_table(
        filepath_or_buffer,
        sep=sep,
        delimiter=delimiter,
        header=header,
        names=names,
        index_col=index_col,
        usecols=usecols,
        dtype=dtype,
        engine=engine,
        converters=converters,
        true_values=true_values,
        false_values=false_values,
        skipinitialspace=skipinitialspace,
        skiprows=skiprows,
        skipfooter=skipfooter,
        nrows=nrows,
        na_values=na_values,
        keep_default_na=keep_default_na,
        na_filter=na_filter,
        skip_blank_lines=skip_blank_lines,
        parse_dates=parse_dates,
        date_format=date_format,
        dayfirst=dayfirst,
        cache_dates=cache_dates,
        iterator=iterator,
        chunksize=chunksize,
        compression=compression,
        thousands=thousands,
        decimal=decimal,
        lineterminator=lineterminator,
        quotechar=quotechar,
        quoting=quoting,
        doublequote=doublequote,
        escapechar=escapechar,
        comment=comment,
        encoding=encoding,
        encoding_errors=encoding_errors,
        dialect=dialect,
        on_bad_lines=on_bad_lines,
        low_memory=low_memory,
        memory_map=memory_map,
        float_precision=float_precision,
        storage_options=storage_options,
        dtype_backend=dtype_backend,
    )


@raises(OSError, ParserError, EmptyDataError, ValueError, TypeError)
def read_fwf(
    filepath_or_buffer: str | Any,
    *,
    colspecs: Sequence[tuple[int, int]] | Literal["infer"] | None = "infer",
    widths: Sequence[int] | None = None,
    infer_nrows: int = 100,
    dtype_backend: Literal["numpy_nullable", "pyarrow"] = "numpy_nullable",
    **kwargs: Any,
) -> _pd.DataFrame:
    """Read a table of fixed-width formatted lines into DataFrame.

    Returns Result[DataFrame, OSError | ParserError | EmptyDataError | ValueError | TypeError]
    """
    return _pd.read_fwf(
        filepath_or_buffer,
        colspecs=colspecs,
        widths=widths,
        infer_nrows=infer_nrows,
        dtype_backend=dtype_backend,
        **kwargs,
    )


@raises(OSError, ValueError, TypeError, KeyError, ImportError)
def read_excel(
    io: str | Any,
    sheet_name: str | int | list | None = 0,
    *,
    header: int | Sequence[int] | None = 0,
    names: Sequence[Hashable] | None = None,
    index_col: int | str | Sequence[int] | None = None,
    usecols: str | Sequence[int] | Sequence[str] | Callable | None = None,
    dtype: dict | None = None,
    engine: Literal["xlrd", "openpyxl", "odf", "pyxlsb", "calamine"] | None = None,
    converters: dict | None = None,
    true_values: Iterable[Hashable] | None = None,
    false_values: Iterable[Hashable] | None = None,
    skiprows: int | Sequence[int] | Callable | None = None,
    nrows: int | None = None,
    na_values: Sequence[str] | dict | None = None,
    keep_default_na: bool = True,
    na_filter: bool = True,
    verbose: bool = False,
    parse_dates: bool | Sequence | dict = False,
    date_format: str | dict | None = None,
    thousands: str | None = None,
    decimal: str = ".",
    comment: str | None = None,
    skipfooter: int = 0,
    storage_options: dict | None = None,
    dtype_backend: Literal["numpy_nullable", "pyarrow"] = "numpy_nullable",
) -> _pd.DataFrame | dict[str, _pd.DataFrame]:
    """Read an Excel file into a DataFrame.

    Returns Result[DataFrame | dict[str, DataFrame], OSError | ValueError | TypeError | KeyError | ImportError]
    """
    return _pd.read_excel(
        io,
        sheet_name=sheet_name,
        header=header,
        names=names,
        index_col=index_col,
        usecols=usecols,
        dtype=dtype,
        engine=engine,
        converters=converters,
        true_values=true_values,
        false_values=false_values,
        skiprows=skiprows,
        nrows=nrows,
        na_values=na_values,
        keep_default_na=keep_default_na,
        na_filter=na_filter,
        verbose=verbose,
        parse_dates=parse_dates,
        date_format=date_format,
        thousands=thousands,
        decimal=decimal,
        comment=comment,
        skipfooter=skipfooter,
        storage_options=storage_options,
        dtype_backend=dtype_backend,
    )


@raises(OSError, ValueError, TypeError, UnicodeDecodeError)
def read_json(
    path_or_buf: str | Any,
    *,
    orient: Literal["split", "records", "index", "columns", "values", "table"] | None = None,
    typ: Literal["frame", "series"] = "frame",
    dtype: dict | bool | None = None,
    convert_axes: bool | None = None,
    convert_dates: bool | list[str] = True,
    keep_default_dates: bool = True,
    precise_float: bool = False,
    date_unit: Literal["s", "ms", "us", "ns"] | None = None,
    encoding: str | None = None,
    encoding_errors: str | None = "strict",
    lines: bool = False,
    chunksize: int | None = None,
    compression: str | dict | None = "infer",
    nrows: int | None = None,
    storage_options: dict | None = None,
    dtype_backend: Literal["numpy_nullable", "pyarrow"] = "numpy_nullable",
    engine: Literal["ujson", "pyarrow"] = "ujson",
) -> _pd.DataFrame | _pd.Series:
    """Read JSON file into DataFrame or Series.

    Returns Result[DataFrame | Series, OSError | ValueError | TypeError | UnicodeDecodeError]
    """
    return _pd.read_json(
        path_or_buf,
        orient=orient,
        typ=typ,
        dtype=dtype,
        convert_axes=convert_axes,
        convert_dates=convert_dates,
        keep_default_dates=keep_default_dates,
        precise_float=precise_float,
        date_unit=date_unit,
        encoding=encoding,
        encoding_errors=encoding_errors,
        lines=lines,
        chunksize=chunksize,
        compression=compression,
        nrows=nrows,
        storage_options=storage_options,
        dtype_backend=dtype_backend,
        engine=engine,
    )


@raises(OSError, ValueError, TypeError, ImportError)
def read_parquet(
    path: str | Any,
    engine: Literal["auto", "pyarrow", "fastparquet"] = "auto",
    columns: list[str] | None = None,
    storage_options: dict | None = None,
    dtype_backend: Literal["numpy_nullable", "pyarrow"] = "numpy_nullable",
    filesystem: Any = None,
    filters: list[tuple] | list[list[tuple]] | None = None,
    **kwargs: Any,
) -> _pd.DataFrame:
    """Read a Parquet file into a DataFrame.

    Returns Result[DataFrame, OSError | ValueError | TypeError | ImportError]
    """
    return _pd.read_parquet(
        path,
        engine=engine,
        columns=columns,
        storage_options=storage_options,
        dtype_backend=dtype_backend,
        filesystem=filesystem,
        filters=filters,
        **kwargs,
    )


@raises(OSError, ValueError, TypeError, ImportError)
def read_feather(
    path: str | Any,
    columns: Sequence[str] | None = None,
    use_threads: bool = True,
    storage_options: dict | None = None,
    dtype_backend: Literal["numpy_nullable", "pyarrow"] = "numpy_nullable",
) -> _pd.DataFrame:
    """Read a Feather file into a DataFrame.

    Returns Result[DataFrame, OSError | ValueError | TypeError | ImportError]
    """
    return _pd.read_feather(
        path,
        columns=columns,
        use_threads=use_threads,
        storage_options=storage_options,
        dtype_backend=dtype_backend,
    )


@raises(OSError, ValueError, TypeError, ImportError)
def read_orc(
    path: str | Any,
    columns: list[str] | None = None,
    dtype_backend: Literal["numpy_nullable", "pyarrow"] = "numpy_nullable",
    filesystem: Any = None,
    **kwargs: Any,
) -> _pd.DataFrame:
    """Read an ORC file into a DataFrame.

    Returns Result[DataFrame, OSError | ValueError | TypeError | ImportError]
    """
    return _pd.read_orc(
        path,
        columns=columns,
        dtype_backend=dtype_backend,
        filesystem=filesystem,
        **kwargs,
    )


@raises(OSError, ValueError, TypeError)
def read_pickle(
    filepath_or_buffer: str | Any,
    compression: str | dict | None = "infer",
    storage_options: dict | None = None,
) -> Any:
    """Read pickled pandas object from file.

    Returns Result[Any, OSError | ValueError | TypeError]
    """
    return _pd.read_pickle(
        filepath_or_buffer,
        compression=compression,
        storage_options=storage_options,
    )


@raises(OSError, ValueError, TypeError)
def to_pickle(
    obj: Any,
    filepath_or_buffer: str | Any,
    compression: str | dict | None = "infer",
    protocol: int = 5,
    storage_options: dict | None = None,
) -> None:
    """Pickle (serialize) object to file.

    Returns Result[None, OSError | ValueError | TypeError]
    """
    return _pd.to_pickle(
        obj,
        filepath_or_buffer,
        compression=compression,
        protocol=protocol,
        storage_options=storage_options,
    )


@raises(OSError, ValueError, TypeError, KeyError, ImportError)
def read_hdf(
    path_or_buf: str | Any,
    key: str | None = None,
    mode: Literal["r", "r+", "a"] = "r",
    errors: str = "strict",
    where: str | list | None = None,
    start: int | None = None,
    stop: int | None = None,
    columns: list[str] | None = None,
    iterator: bool = False,
    chunksize: int | None = None,
    **kwargs: Any,
) -> _pd.DataFrame | _pd.Series:
    """Read from an HDF5 file.

    Returns Result[DataFrame | Series, OSError | ValueError | TypeError | KeyError | ImportError]
    """
    return _pd.read_hdf(
        path_or_buf,
        key=key,
        mode=mode,
        errors=errors,
        where=where,
        start=start,
        stop=stop,
        columns=columns,
        iterator=iterator,
        chunksize=chunksize,
        **kwargs,
    )


@raises(OSError, ValueError, TypeError)
def read_stata(
    filepath_or_buffer: str | Any,
    *,
    convert_dates: bool = True,
    convert_categoricals: bool = True,
    index_col: str | None = None,
    convert_missing: bool = False,
    preserve_dtypes: bool = True,
    columns: list[str] | None = None,
    order_categoricals: bool = True,
    chunksize: int | None = None,
    iterator: bool = False,
    compression: str | dict | None = "infer",
    storage_options: dict | None = None,
) -> _pd.DataFrame:
    """Read Stata file into DataFrame.

    Returns Result[DataFrame, OSError | ValueError | TypeError]
    """
    return _pd.read_stata(
        filepath_or_buffer,
        convert_dates=convert_dates,
        convert_categoricals=convert_categoricals,
        index_col=index_col,
        convert_missing=convert_missing,
        preserve_dtypes=preserve_dtypes,
        columns=columns,
        order_categoricals=order_categoricals,
        chunksize=chunksize,
        iterator=iterator,
        compression=compression,
        storage_options=storage_options,
    )


@raises(OSError, ValueError, TypeError, ImportError)
def read_sas(
    filepath_or_buffer: str | Any,
    *,
    format: Literal["xport", "sas7bdat"] | None = None,
    index: str | None = None,
    encoding: str | None = None,
    chunksize: int | None = None,
    iterator: bool = False,
    compression: str | dict | None = "infer",
) -> _pd.DataFrame:
    """Read SAS file into DataFrame.

    Returns Result[DataFrame, OSError | ValueError | TypeError | ImportError]
    """
    return _pd.read_sas(
        filepath_or_buffer,
        format=format,
        index=index,
        encoding=encoding,
        chunksize=chunksize,
        iterator=iterator,
        compression=compression,
    )


@raises(OSError, ValueError, TypeError, ImportError)
def read_spss(
    path: str | Any,
    usecols: Sequence[str] | None = None,
    convert_categoricals: bool = True,
    dtype_backend: Literal["numpy_nullable", "pyarrow"] = "numpy_nullable",
) -> _pd.DataFrame:
    """Read SPSS file into DataFrame.

    Returns Result[DataFrame, OSError | ValueError | TypeError | ImportError]
    """
    return _pd.read_spss(
        path,
        usecols=usecols,
        convert_categoricals=convert_categoricals,
        dtype_backend=dtype_backend,
    )


@raises(OSError, ValueError, TypeError, KeyError, ImportError)
def read_sql(
    sql: str,
    con: Any,
    index_col: str | list[str] | None = None,
    coerce_float: bool = True,
    params: list | tuple | Mapping | None = None,
    parse_dates: list | dict | None = None,
    columns: list[str] | None = None,
    chunksize: int | None = None,
    dtype_backend: Literal["numpy_nullable", "pyarrow"] = "numpy_nullable",
    dtype: dict | None = None,
) -> _pd.DataFrame:
    """Read SQL query or table into DataFrame.

    Returns Result[DataFrame, OSError | ValueError | TypeError | KeyError | ImportError]
    """
    return _pd.read_sql(
        sql,
        con,
        index_col=index_col,
        coerce_float=coerce_float,
        params=params,
        parse_dates=parse_dates,
        columns=columns,
        chunksize=chunksize,
        dtype_backend=dtype_backend,
        dtype=dtype,
    )


@raises(OSError, ValueError, TypeError, KeyError, ImportError)
def read_sql_query(
    sql: str,
    con: Any,
    index_col: str | list[str] | None = None,
    coerce_float: bool = True,
    params: list | tuple | Mapping | None = None,
    parse_dates: list | dict | None = None,
    chunksize: int | None = None,
    dtype: dict | None = None,
    dtype_backend: Literal["numpy_nullable", "pyarrow"] = "numpy_nullable",
) -> _pd.DataFrame:
    """Read SQL query into DataFrame.

    Returns Result[DataFrame, OSError | ValueError | TypeError | KeyError | ImportError]
    """
    return _pd.read_sql_query(
        sql,
        con,
        index_col=index_col,
        coerce_float=coerce_float,
        params=params,
        parse_dates=parse_dates,
        chunksize=chunksize,
        dtype=dtype,
        dtype_backend=dtype_backend,
    )


@raises(OSError, ValueError, TypeError, KeyError, ImportError)
def read_sql_table(
    table_name: str,
    con: Any,
    schema: str | None = None,
    index_col: str | list[str] | None = None,
    coerce_float: bool = True,
    parse_dates: list | dict | None = None,
    columns: list[str] | None = None,
    chunksize: int | None = None,
    dtype_backend: Literal["numpy_nullable", "pyarrow"] = "numpy_nullable",
) -> _pd.DataFrame:
    """Read SQL table into DataFrame.

    Returns Result[DataFrame, OSError | ValueError | TypeError | KeyError | ImportError]
    """
    return _pd.read_sql_table(
        table_name,
        con,
        schema=schema,
        index_col=index_col,
        coerce_float=coerce_float,
        parse_dates=parse_dates,
        columns=columns,
        chunksize=chunksize,
        dtype_backend=dtype_backend,
    )


@raises(OSError, ValueError, TypeError, ImportError)
def read_gbq(
    query: str,
    project_id: str | None = None,
    index_col: str | None = None,
    col_order: list[str] | None = None,
    reauth: bool = False,
    auth_local_webserver: bool = True,
    dialect: Literal["legacy", "standard"] | None = None,
    location: str | None = None,
    configuration: dict | None = None,
    credentials: Any = None,
    use_bqstorage_api: bool | None = None,
    max_results: int | None = None,
    progress_bar_type: Literal["tqdm", "tqdm_notebook", "tqdm_gui"] | None = None,
) -> _pd.DataFrame:
    """Read Google BigQuery table into DataFrame.

    Returns Result[DataFrame, OSError | ValueError | TypeError | ImportError]
    """
    return _pd.read_gbq(
        query,
        project_id=project_id,
        index_col=index_col,
        col_order=col_order,
        reauth=reauth,
        auth_local_webserver=auth_local_webserver,
        dialect=dialect,
        location=location,
        configuration=configuration,
        credentials=credentials,
        use_bqstorage_api=use_bqstorage_api,
        max_results=max_results,
        progress_bar_type=progress_bar_type,
    )


@raises(OSError, ValueError, TypeError, ImportError)
def read_html(
    io: str | Any,
    *,
    match: str | None = ".+",
    flavor: str | None = None,
    header: int | Sequence[int] | None = None,
    index_col: int | Sequence[int] | None = None,
    skiprows: int | Sequence[int] | slice | None = None,
    attrs: dict | None = None,
    parse_dates: bool = False,
    thousands: str | None = ",",
    encoding: str | None = None,
    decimal: str = ".",
    converters: dict | None = None,
    na_values: Iterable[object] | None = None,
    keep_default_na: bool = True,
    displayed_only: bool = True,
    extract_links: Literal["header", "footer", "body", "all"] | None = None,
    dtype_backend: Literal["numpy_nullable", "pyarrow"] = "numpy_nullable",
    storage_options: dict | None = None,
) -> list[_pd.DataFrame]:
    """Read HTML tables into list of DataFrame objects.

    Returns Result[list[DataFrame], OSError | ValueError | TypeError | ImportError]
    """
    return _pd.read_html(
        io,
        match=match,
        flavor=flavor,
        header=header,
        index_col=index_col,
        skiprows=skiprows,
        attrs=attrs,
        parse_dates=parse_dates,
        thousands=thousands,
        encoding=encoding,
        decimal=decimal,
        converters=converters,
        na_values=na_values,
        keep_default_na=keep_default_na,
        displayed_only=displayed_only,
        extract_links=extract_links,
        dtype_backend=dtype_backend,
        storage_options=storage_options,
    )


@raises(OSError, ValueError, TypeError, ImportError)
def read_xml(
    path_or_buffer: str | Any,
    *,
    xpath: str = "./*",
    namespaces: dict | None = None,
    elems_only: bool = False,
    attrs_only: bool = False,
    names: Sequence[str] | None = None,
    dtype: dict | None = None,
    converters: dict | None = None,
    parse_dates: bool | Sequence | dict | None = None,
    encoding: str | None = "utf-8",
    parser: Literal["lxml", "etree"] = "lxml",
    stylesheet: str | Any = None,
    iterparse: dict[str, list[str]] | None = None,
    compression: str | dict | None = "infer",
    storage_options: dict | None = None,
    dtype_backend: Literal["numpy_nullable", "pyarrow"] = "numpy_nullable",
) -> _pd.DataFrame:
    """Read XML document into DataFrame.

    Returns Result[DataFrame, OSError | ValueError | TypeError | ImportError]
    """
    return _pd.read_xml(
        path_or_buffer,
        xpath=xpath,
        namespaces=namespaces,
        elems_only=elems_only,
        attrs_only=attrs_only,
        names=names,
        dtype=dtype,
        converters=converters,
        parse_dates=parse_dates,
        encoding=encoding,
        parser=parser,
        stylesheet=stylesheet,
        iterparse=iterparse,
        compression=compression,
        storage_options=storage_options,
        dtype_backend=dtype_backend,
    )


@raises(OSError, ParserError, EmptyDataError, ValueError, TypeError)
def read_clipboard(
    sep: str = r"\s+",
    dtype_backend: Literal["numpy_nullable", "pyarrow"] = "numpy_nullable",
    **kwargs: Any,
) -> _pd.DataFrame:
    """Read text from clipboard into DataFrame.

    Returns Result[DataFrame, OSError | ParserError | EmptyDataError | ValueError | TypeError]
    """
    return _pd.read_clipboard(sep=sep, dtype_backend=dtype_backend, **kwargs)


# =============================================================================
# PARSING - Shielded functions (can fail with invalid data)
# =============================================================================


@raises(ValueError, TypeError, OutOfBoundsDatetime)
def to_datetime(
    arg: Any,
    errors: Literal["raise", "coerce"] = "raise",
    dayfirst: bool = False,
    yearfirst: bool = False,
    utc: bool = False,
    format: str | None = None,
    exact: bool = True,
    unit: str | None = None,
    origin: str = "unix",
    cache: bool = True,
) -> _pd.DatetimeIndex | _pd.Series | _pd.Timestamp:
    """Convert argument to datetime.

    Returns Result[DatetimeIndex | Series | Timestamp, ValueError | TypeError | OutOfBoundsDatetime]
    """
    return _pd.to_datetime(
        arg,
        errors=errors,
        dayfirst=dayfirst,
        yearfirst=yearfirst,
        utc=utc,
        format=format,
        exact=exact,
        unit=unit,
        origin=origin,
        cache=cache,
    )


@raises(ValueError, TypeError)
def to_numeric(
    arg: Any,
    errors: Literal["raise", "coerce"] = "raise",
    downcast: Literal["integer", "signed", "unsigned", "float"] | None = None,
    dtype_backend: Literal["numpy_nullable", "pyarrow"] = "numpy_nullable",
) -> _pd.Series | Any:
    """Convert argument to numeric type.

    Returns Result[Series | scalar, ValueError | TypeError]
    """
    return _pd.to_numeric(
        arg,
        errors=errors,
        downcast=downcast,
        dtype_backend=dtype_backend,
    )


@raises(ValueError, TypeError, OutOfBoundsTimedelta)
def to_timedelta(
    arg: Any,
    unit: str | None = None,
    errors: Literal["raise", "coerce"] = "raise",
) -> _pd.TimedeltaIndex | _pd.Series | _pd.Timedelta:
    """Convert argument to timedelta.

    Returns Result[TimedeltaIndex | Series | Timedelta, ValueError | TypeError | OutOfBoundsTimedelta]
    """
    return _pd.to_timedelta(arg, unit=unit, errors=errors)


@raises(ValueError, SyntaxError, UndefinedVariableError)
def eval(
    expr: str,
    parser: Literal["pandas", "python"] = "pandas",
    engine: Literal["python", "numexpr"] | None = None,
    local_dict: dict | None = None,
    global_dict: dict | None = None,
    resolvers: list | None = None,
    level: int = 0,
    target: Any = None,
    inplace: bool = False,
) -> Any:
    """Evaluate a Python expression as a string.

    Returns Result[Any, ValueError | SyntaxError | UndefinedVariableError]
    """
    return _pd.eval(
        expr,
        parser=parser,
        engine=engine,
        local_dict=local_dict,
        global_dict=global_dict,
        resolvers=resolvers,
        level=level,
        target=target,
        inplace=inplace,
    )


# =============================================================================
# DATA MANIPULATION - Shielded functions (can fail with incompatible data)
# =============================================================================


@raises(TypeError, ValueError)
def concat(
    objs: Iterable[_pd.DataFrame | _pd.Series] | Mapping[Hashable, _pd.DataFrame | _pd.Series],
    *,
    axis: Literal[0, 1, "index", "columns"] = 0,
    join: Literal["inner", "outer"] = "outer",
    ignore_index: bool = False,
    keys: Sequence[Any] | None = None,
    levels: list[Sequence[Any]] | None = None,
    names: list[Hashable] | None = None,
    verify_integrity: bool = False,
    sort: bool = False,
    copy: bool = True,
) -> _pd.DataFrame | _pd.Series:
    """Concatenate pandas objects along a particular axis.

    Returns Result[DataFrame | Series, TypeError | ValueError]

    Raises:
        TypeError: If objects are not all DataFrame or Series
        ValueError: If verify_integrity=True and indexes overlap
    """
    return _pd.concat(
        objs,
        axis=axis,
        join=join,
        ignore_index=ignore_index,
        keys=keys,
        levels=levels,
        names=names,
        verify_integrity=verify_integrity,
        sort=sort,
        copy=copy,
    )


@raises(MergeError, KeyError, ValueError, TypeError)
def merge(
    left: _pd.DataFrame | _pd.Series,
    right: _pd.DataFrame | _pd.Series,
    how: Literal["left", "right", "outer", "inner", "cross"] = "inner",
    on: Hashable | Sequence[Hashable] | None = None,
    left_on: Hashable | Sequence[Hashable] | None = None,
    right_on: Hashable | Sequence[Hashable] | None = None,
    left_index: bool = False,
    right_index: bool = False,
    sort: bool = False,
    suffixes: tuple[str | None, str | None] = ("_x", "_y"),
    copy: bool = True,
    indicator: bool | str = False,
    validate: Literal["one_to_one", "one_to_many", "many_to_one", "many_to_many"] | None = None,
) -> _pd.DataFrame:
    """Merge DataFrame or named Series objects with a database-style join.

    Returns Result[DataFrame, MergeError | KeyError | ValueError | TypeError]

    Raises:
        MergeError: If merge keys are not unique when validate is specified
        KeyError: If join keys are not found in both DataFrames
        ValueError: If merge specification is invalid
        TypeError: If objects are not DataFrame/Series
    """
    return _pd.merge(
        left,
        right,
        how=how,
        on=on,
        left_on=left_on,
        right_on=right_on,
        left_index=left_index,
        right_index=right_index,
        sort=sort,
        suffixes=suffixes,
        copy=copy,
        indicator=indicator,
        validate=validate,
    )


@raises(ValueError, KeyError, TypeError)
def pivot(
    data: _pd.DataFrame,
    *,
    columns: Hashable | Sequence[Hashable] | None = None,
    index: Hashable | Sequence[Hashable] | None = None,
    values: Hashable | Sequence[Hashable] | None = None,
) -> _pd.DataFrame:
    """Reshape data by given index/column values.

    Returns Result[DataFrame, ValueError | KeyError | TypeError]

    Raises:
        ValueError: If there are duplicate entries for the index/columns combination
        KeyError: If columns/index/values not found in DataFrame
        TypeError: If data is not a DataFrame
    """
    return _pd.pivot(data, columns=columns, index=index, values=values)


@raises(ValueError, TypeError)
def cut(
    x: Any,
    bins: int | Sequence[float] | _pd.IntervalIndex,
    right: bool = True,
    labels: Sequence[Hashable] | bool | None = None,
    retbins: bool = False,
    precision: int = 3,
    include_lowest: bool = False,
    duplicates: Literal["raise", "drop"] = "raise",
    ordered: bool = True,
) -> _pd.Categorical | _pd.Series | tuple[_pd.Categorical | _pd.Series, Any]:
    """Bin values into discrete intervals.

    Returns Result[Categorical | Series | tuple, ValueError | TypeError]

    Raises:
        ValueError: If bins edges are not unique, or labels don't match bins
        TypeError: If x cannot be converted to numeric
    """
    return _pd.cut(
        x,
        bins,
        right=right,
        labels=labels,
        retbins=retbins,
        precision=precision,
        include_lowest=include_lowest,
        duplicates=duplicates,
        ordered=ordered,
    )


@raises(ValueError, TypeError)
def qcut(
    x: Any,
    q: int | Sequence[float],
    labels: Sequence[Hashable] | bool | None = None,
    retbins: bool = False,
    precision: int = 3,
    duplicates: Literal["raise", "drop"] = "raise",
) -> _pd.Categorical | _pd.Series | tuple[_pd.Categorical | _pd.Series, Any]:
    """Quantile-based discretization function.

    Returns Result[Categorical | Series | tuple, ValueError | TypeError]

    Raises:
        ValueError: If quantile bins are not unique (too few unique values in data)
        TypeError: If x cannot be converted to numeric
    """
    return _pd.qcut(x, q, labels=labels, retbins=retbins, precision=precision, duplicates=duplicates)


@raises(ValueError, TypeError, KeyError)
def json_normalize(
    data: dict | list[dict],
    record_path: str | list[str] | None = None,
    meta: str | list[str | list[str]] | None = None,
    meta_prefix: str | None = None,
    record_prefix: str | None = None,
    errors: Literal["raise", "ignore"] = "raise",
    sep: str = ".",
    max_level: int | None = None,
) -> _pd.DataFrame:
    """Normalize semi-structured JSON data into a flat table.

    Returns Result[DataFrame, ValueError | TypeError | KeyError]

    Raises:
        ValueError: If data structure is invalid or inconsistent
        TypeError: If data is not a dict or list of dicts
        KeyError: If record_path or meta keys not found in data
    """
    return _pd.json_normalize(
        data,
        record_path=record_path,
        meta=meta,
        meta_prefix=meta_prefix,
        record_prefix=record_prefix,
        errors=errors,
        sep=sep,
        max_level=max_level,
    )


# =============================================================================
# __all__ - Public API
# =============================================================================

__all__ = [
    # Version info
    "__shield_version__",
    "__pandas_version__",
    # Exceptions
    "AbstractMethodError",
    "DuplicateLabelError",
    "EmptyDataError",
    "IntCastingNaNError",
    "InvalidIndexError",
    "MergeError",
    "NullFrequencyError",
    "NumbaUtilError",
    "OptionError",
    "OutOfBoundsDatetime",
    "OutOfBoundsTimedelta",
    "ParserError",
    "PerformanceWarning",
    "UndefinedVariableError",
    "UnsortedIndexError",
    "UnsupportedFunctionCall",
    # Core data structures
    "DataFrame",
    "Series",
    "Index",
    "RangeIndex",
    "MultiIndex",
    "CategoricalIndex",
    "DatetimeIndex",
    "TimedeltaIndex",
    "PeriodIndex",
    "IntervalIndex",
    "Categorical",
    "Grouper",
    "NamedAgg",
    "Flags",
    # Dtype types
    "ArrowDtype",
    "BooleanDtype",
    "CategoricalDtype",
    "DatetimeTZDtype",
    "Float32Dtype",
    "Float64Dtype",
    "Int8Dtype",
    "Int16Dtype",
    "Int32Dtype",
    "Int64Dtype",
    "PeriodDtype",
    "SparseDtype",
    "StringDtype",
    "UInt8Dtype",
    "UInt16Dtype",
    "UInt32Dtype",
    "UInt64Dtype",
    "IntervalDtype",
    # Special values
    "NA",
    "NaT",
    "Timestamp",
    "Timedelta",
    "Period",
    "Interval",
    "DateOffset",
    # Utility classes
    "ExcelFile",
    "ExcelWriter",
    "HDFStore",
    "IndexSlice",
    # Range generators
    "date_range",
    "bdate_range",
    "period_range",
    "timedelta_range",
    "interval_range",
    # Data manipulation
    "concat",
    "merge",
    "merge_asof",
    "merge_ordered",
    "pivot",
    "pivot_table",
    "melt",
    "lreshape",
    "wide_to_long",
    "crosstab",
    "cut",
    "qcut",
    "factorize",
    "get_dummies",
    "from_dummies",
    "unique",
    "value_counts",
    # Null checking
    "isna",
    "isnull",
    "notna",
    "notnull",
    # Array creation
    "array",
    # Frequency inference
    "infer_freq",
    # JSON normalization
    "json_normalize",
    # Options
    "options",
    "get_option",
    "set_option",
    "reset_option",
    "describe_option",
    "option_context",
    # Display
    "set_eng_float_format",
    "show_versions",
    # Submodules
    "api",
    "arrays",
    "errors",
    "io",
    "offsets",
    "plotting",
    "testing",
    "tseries",
    # Shielded file I/O
    "read_csv",
    "read_table",
    "read_fwf",
    "read_excel",
    "read_json",
    "read_parquet",
    "read_feather",
    "read_orc",
    "read_pickle",
    "to_pickle",
    "read_hdf",
    "read_stata",
    "read_sas",
    "read_spss",
    "read_sql",
    "read_sql_query",
    "read_sql_table",
    "read_gbq",
    "read_html",
    "read_xml",
    "read_clipboard",
    # Shielded parsing
    "to_datetime",
    "to_numeric",
    "to_timedelta",
    "eval",
]
