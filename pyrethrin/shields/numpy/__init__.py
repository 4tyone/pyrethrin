"""Pyrethrin shields for NumPy.

This module provides shielded versions of NumPy operations that can fail at
runtime due to data-dependent conditions (singular matrices, missing files, etc.).

Shield version: 0.1.0
NumPy version: 2.4.0

Note: NumPy is a C extension, so Arbor cannot trace into its implementation.
The exception declarations are based on NumPy's documented behavior and testing.

Usage:
    from pyrethrin.shields.numpy import linalg_inv, load_npy, save_npy
    from pyrethrin import Ok, Err

    # Safe matrix inversion
    match linalg_inv(matrix):
        case Ok(inverse):
            # use inverse
        case Err(LinAlgError()):
            print("Matrix is singular")
        case Err(ValueError()):
            print("Invalid matrix shape")
"""

from __future__ import annotations

__shield_version__ = "0.1.0"
__numpy_version__ = "2.4.0"

from collections.abc import Sequence
from typing import Any, Literal

import numpy as _np
from numpy import ndarray
from numpy.linalg import LinAlgError

from pyrethrin.decorators import raises

# =============================================================================
# EXCEPTIONS - Re-export numpy exceptions
# =============================================================================

from numpy.linalg import LinAlgError as LinAlgError
from numpy.exceptions import AxisError as AxisError
from numpy.exceptions import DTypePromotionError as DTypePromotionError

# =============================================================================
# ARRAY TYPES - Re-export unchanged
# =============================================================================

from numpy import ndarray as ndarray
from numpy import dtype as dtype
from numpy import generic as generic
from numpy import number as number
from numpy import integer as integer
from numpy import floating as floating
from numpy import complexfloating as complexfloating

# =============================================================================
# ARRAY CREATION - Re-export unchanged (rarely fail at runtime)
# =============================================================================

from numpy import array as array
from numpy import asarray as asarray
from numpy import zeros as zeros
from numpy import ones as ones
from numpy import empty as empty
from numpy import full as full
from numpy import arange as arange
from numpy import linspace as linspace
from numpy import logspace as logspace
from numpy import geomspace as geomspace
from numpy import eye as eye
from numpy import identity as identity
from numpy import diag as diag
from numpy import diagflat as diagflat
from numpy import tri as tri
from numpy import tril as tril
from numpy import triu as triu
from numpy import vander as vander
from numpy import zeros_like as zeros_like
from numpy import ones_like as ones_like
from numpy import empty_like as empty_like
from numpy import full_like as full_like
from numpy import copy as copy
from numpy import frombuffer as frombuffer
from numpy import fromfunction as fromfunction
from numpy import fromiter as fromiter
from numpy import fromstring as fromstring
from numpy import ascontiguousarray as ascontiguousarray
from numpy import asfortranarray as asfortranarray
from numpy import asanyarray as asanyarray
from numpy import asarray_chkfinite as asarray_chkfinite
from numpy import require as require
from numpy import indices as indices
from numpy import from_dlpack as from_dlpack

# =============================================================================
# ARRAY MANIPULATION - Re-export unchanged
# =============================================================================

from numpy import reshape as reshape
from numpy import ravel as ravel
from numpy import transpose as transpose
from numpy import swapaxes as swapaxes
from numpy import moveaxis as moveaxis
from numpy import rollaxis as rollaxis
from numpy import expand_dims as expand_dims
from numpy import squeeze as squeeze
from numpy import concatenate as concatenate
from numpy import stack as stack
from numpy import vstack as vstack
from numpy import hstack as hstack
from numpy import dstack as dstack
from numpy import column_stack as column_stack
from numpy import row_stack as row_stack
from numpy import split as split
from numpy import hsplit as hsplit
from numpy import vsplit as vsplit
from numpy import dsplit as dsplit
from numpy import array_split as array_split
from numpy import tile as tile
from numpy import repeat as repeat
from numpy import flip as flip
from numpy import fliplr as fliplr
from numpy import flipud as flipud
from numpy import roll as roll
from numpy import rot90 as rot90
from numpy import atleast_1d as atleast_1d
from numpy import atleast_2d as atleast_2d
from numpy import atleast_3d as atleast_3d
from numpy import broadcast_to as broadcast_to
from numpy import broadcast_arrays as broadcast_arrays
from numpy import append as append
from numpy import insert as insert
from numpy import delete as delete
from numpy import resize as resize
from numpy import trim_zeros as trim_zeros
from numpy import pad as pad
from numpy import shape as shape
from numpy import size as size
from numpy import ndim as ndim
from numpy import block as block
from numpy import copyto as copyto
from numpy import putmask as putmask
from numpy import permute_dims as permute_dims
from numpy import unstack as unstack
from numpy import matrix_transpose as matrix_transpose
from numpy import astype as astype

# =============================================================================
# MATHEMATICAL OPERATIONS - Re-export unchanged
# =============================================================================

from numpy import add as add
from numpy import subtract as subtract
from numpy import multiply as multiply
from numpy import divide as divide
from numpy import true_divide as true_divide
from numpy import floor_divide as floor_divide
from numpy import mod as mod
from numpy import fmod as fmod
from numpy import remainder as remainder
from numpy import divmod as divmod
from numpy import power as power
from numpy import float_power as float_power
from numpy import sqrt as sqrt
from numpy import cbrt as cbrt
from numpy import square as square
from numpy import reciprocal as reciprocal
from numpy import positive as positive
from numpy import negative as negative
from numpy import abs as abs
from numpy import absolute as absolute
from numpy import fabs as fabs
from numpy import sign as sign
from numpy import heaviside as heaviside
from numpy import exp as exp
from numpy import exp2 as exp2
from numpy import expm1 as expm1
from numpy import log as log
from numpy import log2 as log2
from numpy import log10 as log10
from numpy import log1p as log1p
from numpy import logaddexp as logaddexp
from numpy import logaddexp2 as logaddexp2
from numpy import sin as sin
from numpy import cos as cos
from numpy import tan as tan
from numpy import arcsin as arcsin
from numpy import arccos as arccos
from numpy import arctan as arctan
from numpy import arctan2 as arctan2
from numpy import hypot as hypot
from numpy import sinh as sinh
from numpy import cosh as cosh
from numpy import tanh as tanh
from numpy import arcsinh as arcsinh
from numpy import arccosh as arccosh
from numpy import arctanh as arctanh
from numpy import degrees as degrees
from numpy import radians as radians
from numpy import deg2rad as deg2rad
from numpy import rad2deg as rad2deg
from numpy import unwrap as unwrap
from numpy import floor as floor
from numpy import ceil as ceil
from numpy import trunc as trunc
from numpy import round as round
from numpy import around as around
from numpy import rint as rint
from numpy import fix as fix
from numpy import clip as clip
from numpy import minimum as minimum
from numpy import maximum as maximum
from numpy import fmin as fmin
from numpy import fmax as fmax
from numpy import modf as modf
from numpy import ldexp as ldexp
from numpy import frexp as frexp
from numpy import copysign as copysign
from numpy import nextafter as nextafter
from numpy import spacing as spacing
from numpy import signbit as signbit
from numpy import sinc as sinc
from numpy import i0 as i0
from numpy import nan_to_num as nan_to_num
from numpy import real_if_close as real_if_close
from numpy import interp as interp
from numpy import angle as angle
from numpy import real as real
from numpy import imag as imag
from numpy import conj as conj
from numpy import conjugate as conjugate
# Aliases for trig functions (numpy 2.x names)
from numpy import acos as acos
from numpy import acosh as acosh
from numpy import asin as asin
from numpy import asinh as asinh
from numpy import atan as atan
from numpy import atanh as atanh
from numpy import atan2 as atan2
# Additional math functions
from numpy import gcd as gcd
from numpy import lcm as lcm
from numpy import pow as pow
from numpy import bitwise_count as bitwise_count
from numpy import trapezoid as trapezoid

# =============================================================================
# AGGREGATION - Re-export unchanged
# =============================================================================

from numpy import sum as sum
from numpy import prod as prod
from numpy import mean as mean
from numpy import std as std
from numpy import var as var
from numpy import min as min
from numpy import max as max
from numpy import ptp as ptp
from numpy import argmin as argmin
from numpy import argmax as argmax
from numpy import nansum as nansum
from numpy import nanprod as nanprod
from numpy import nanmean as nanmean
from numpy import nanstd as nanstd
from numpy import nanvar as nanvar
from numpy import nanmin as nanmin
from numpy import nanmax as nanmax
from numpy import nanargmin as nanargmin
from numpy import nanargmax as nanargmax
from numpy import cumsum as cumsum
from numpy import cumprod as cumprod
from numpy import nancumsum as nancumsum
from numpy import nancumprod as nancumprod
from numpy import diff as diff
from numpy import ediff1d as ediff1d
from numpy import gradient as gradient
from numpy import cross as cross
from numpy import percentile as percentile
from numpy import nanpercentile as nanpercentile
from numpy import quantile as quantile
from numpy import nanquantile as nanquantile
from numpy import median as median
from numpy import nanmedian as nanmedian
from numpy import average as average
from numpy import histogram as histogram
from numpy import histogram2d as histogram2d
from numpy import histogramdd as histogramdd
from numpy import bincount as bincount
from numpy import digitize as digitize
from numpy import cov as cov
from numpy import corrcoef as corrcoef
from numpy import correlate as correlate
from numpy import convolve as convolve
from numpy import cumulative_sum as cumulative_sum
from numpy import cumulative_prod as cumulative_prod
from numpy import amax as amax
from numpy import amin as amin
from numpy import histogram_bin_edges as histogram_bin_edges

# =============================================================================
# COMPARISON - Re-export unchanged
# =============================================================================

from numpy import equal as equal
from numpy import not_equal as not_equal
from numpy import less as less
from numpy import less_equal as less_equal
from numpy import greater as greater
from numpy import greater_equal as greater_equal
from numpy import isnan as isnan
from numpy import isinf as isinf
from numpy import isfinite as isfinite
from numpy import isneginf as isneginf
from numpy import isposinf as isposinf
from numpy import isclose as isclose
from numpy import allclose as allclose
from numpy import array_equal as array_equal
from numpy import array_equiv as array_equiv
from numpy import iscomplex as iscomplex
from numpy import isreal as isreal
from numpy import isnat as isnat
from numpy import isfortran as isfortran
from numpy import iterable as iterable

# =============================================================================
# LOGIC - Re-export unchanged
# =============================================================================

from numpy import all as all
from numpy import any as any
from numpy import logical_and as logical_and
from numpy import logical_or as logical_or
from numpy import logical_not as logical_not
from numpy import logical_xor as logical_xor
from numpy import where as where
from numpy import nonzero as nonzero
from numpy import argwhere as argwhere
from numpy import flatnonzero as flatnonzero
from numpy import count_nonzero as count_nonzero

# =============================================================================
# BITWISE OPERATIONS - Re-export unchanged
# =============================================================================

from numpy import bitwise_and as bitwise_and
from numpy import bitwise_or as bitwise_or
from numpy import bitwise_xor as bitwise_xor
from numpy import invert as invert
from numpy import bitwise_invert as bitwise_invert
from numpy import bitwise_not as bitwise_not
from numpy import left_shift as left_shift
from numpy import right_shift as right_shift
from numpy import bitwise_left_shift as bitwise_left_shift
from numpy import bitwise_right_shift as bitwise_right_shift
from numpy import packbits as packbits
from numpy import unpackbits as unpackbits
from numpy import binary_repr as binary_repr

# =============================================================================
# INDEXING AND SLICING - Re-export unchanged
# =============================================================================

from numpy import take as take
from numpy import put as put
from numpy import choose as choose
from numpy import select as select
from numpy import compress as compress
from numpy import extract as extract
from numpy import place as place
from numpy import put_along_axis as put_along_axis
from numpy import take_along_axis as take_along_axis
from numpy import diagonal as diagonal
from numpy import diag_indices as diag_indices
from numpy import diag_indices_from as diag_indices_from
from numpy import tril_indices as tril_indices
from numpy import tril_indices_from as tril_indices_from
from numpy import triu_indices as triu_indices
from numpy import triu_indices_from as triu_indices_from
from numpy import mask_indices as mask_indices
from numpy import fill_diagonal as fill_diagonal
from numpy import unravel_index as unravel_index
from numpy import ravel_multi_index as ravel_multi_index
from numpy import ix_ as ix_

# =============================================================================
# MATRIX OPERATIONS - Re-export unchanged
# =============================================================================

from numpy import dot as dot
from numpy import vdot as vdot
from numpy import inner as inner
from numpy import outer as outer
from numpy import matmul as matmul
from numpy import tensordot as tensordot
from numpy import einsum as einsum
from numpy import einsum_path as einsum_path
from numpy import kron as kron
from numpy import trace as trace
from numpy import matvec as matvec
from numpy import vecdot as vecdot
from numpy import vecmat as vecmat

# =============================================================================
# WINDOW FUNCTIONS - Re-export unchanged
# =============================================================================

from numpy import bartlett as bartlett
from numpy import blackman as blackman
from numpy import hamming as hamming
from numpy import hanning as hanning
from numpy import kaiser as kaiser

# =============================================================================
# MESH/GRID FUNCTIONS - Re-export unchanged
# =============================================================================

from numpy import meshgrid as meshgrid
from numpy import mgrid as mgrid
from numpy import ogrid as ogrid

# =============================================================================
# SORTING AND SEARCHING - Re-export unchanged
# =============================================================================

from numpy import sort as sort
from numpy import argsort as argsort
from numpy import lexsort as lexsort
from numpy import partition as partition
from numpy import argpartition as argpartition
from numpy import searchsorted as searchsorted
from numpy import unique as unique
from numpy import unique_all as unique_all
from numpy import unique_counts as unique_counts
from numpy import unique_inverse as unique_inverse
from numpy import unique_values as unique_values
from numpy import sort_complex as sort_complex

# =============================================================================
# SET OPERATIONS - Re-export unchanged
# =============================================================================

from numpy import intersect1d as intersect1d
from numpy import union1d as union1d
from numpy import setdiff1d as setdiff1d
from numpy import setxor1d as setxor1d
from numpy import isin as isin

# =============================================================================
# TYPE FUNCTIONS - Re-export unchanged
# =============================================================================

from numpy import can_cast as can_cast
from numpy import promote_types as promote_types
from numpy import result_type as result_type
from numpy import min_scalar_type as min_scalar_type
from numpy import common_type as common_type
from numpy import issubdtype as issubdtype
from numpy import iscomplexobj as iscomplexobj
from numpy import isrealobj as isrealobj
from numpy import isscalar as isscalar
from numpy import isdtype as isdtype
from numpy import mintypecode as mintypecode

# =============================================================================
# DTYPE - Re-export unchanged
# =============================================================================

from numpy import bool_ as bool_
from numpy import int8 as int8
from numpy import int16 as int16
from numpy import int32 as int32
from numpy import int64 as int64
from numpy import uint8 as uint8
from numpy import uint16 as uint16
from numpy import uint32 as uint32
from numpy import uint64 as uint64
from numpy import float16 as float16
from numpy import float32 as float32
from numpy import float64 as float64
from numpy import complex64 as complex64
from numpy import complex128 as complex128
from numpy import intp as intp
from numpy import uintp as uintp
from numpy import intc as intc
from numpy import uintc as uintc
from numpy import longlong as longlong
from numpy import ulonglong as ulonglong
from numpy import single as single
from numpy import double as double
from numpy import longdouble as longdouble
from numpy import csingle as csingle
from numpy import cdouble as cdouble
from numpy import clongdouble as clongdouble
from numpy import object_ as object_
from numpy import bytes_ as bytes_
from numpy import str_ as str_
from numpy import void as void
from numpy import datetime64 as datetime64
from numpy import timedelta64 as timedelta64
# Additional scalar types
from numpy import byte as byte
from numpy import ubyte as ubyte
from numpy import short as short
from numpy import ushort as ushort
from numpy import int_ as int_
from numpy import uint as uint
from numpy import long as long
from numpy import ulong as ulong
from numpy import half as half
# Abstract types
from numpy import character as character
from numpy import flexible as flexible
from numpy import inexact as inexact
from numpy import signedinteger as signedinteger
from numpy import unsignedinteger as unsignedinteger

# =============================================================================
# MEMORY AND INFO - Re-export unchanged
# =============================================================================

from numpy import shares_memory as shares_memory
from numpy import may_share_memory as may_share_memory
from numpy import finfo as finfo
from numpy import iinfo as iinfo
from numpy import base_repr as base_repr

# =============================================================================
# FUNCTIONAL OPERATIONS - Re-export unchanged
# =============================================================================

from numpy import apply_along_axis as apply_along_axis
from numpy import apply_over_axes as apply_over_axes
from numpy import vectorize as vectorize
from numpy import frompyfunc as frompyfunc
from numpy import piecewise as piecewise

# =============================================================================
# RANDOM - Re-export unchanged (use numpy.random module directly)
# =============================================================================

from numpy import random as random

# =============================================================================
# FFT - Re-export unchanged (use numpy.fft module directly)
# =============================================================================

from numpy import fft as fft

# =============================================================================
# POLYNOMIAL MODULE - Re-export unchanged
# =============================================================================

from numpy import polynomial as polynomial

# =============================================================================
# INDEXING HELPERS - Re-export unchanged
# =============================================================================

from numpy import c_ as c_
from numpy import r_ as r_
from numpy import s_ as s_
from numpy import index_exp as index_exp

# =============================================================================
# ITERATOR TYPES - Re-export unchanged
# =============================================================================

from numpy import nditer as nditer
from numpy import ndenumerate as ndenumerate
from numpy import ndindex as ndindex
from numpy import flatiter as flatiter
from numpy import broadcast as broadcast
from numpy import broadcast_shapes as broadcast_shapes

# =============================================================================
# ERROR HANDLING - Re-export unchanged
# =============================================================================

from numpy import errstate as errstate
from numpy import seterr as seterr
from numpy import geterr as geterr
from numpy import seterrcall as seterrcall
from numpy import geterrcall as geterrcall

# =============================================================================
# PRINT OPTIONS - Re-export unchanged
# =============================================================================

from numpy import set_printoptions as set_printoptions
from numpy import get_printoptions as get_printoptions
from numpy import printoptions as printoptions
from numpy import array2string as array2string
from numpy import array_repr as array_repr
from numpy import array_str as array_str
from numpy import format_float_positional as format_float_positional
from numpy import format_float_scientific as format_float_scientific

# =============================================================================
# DATETIME - Re-export unchanged
# =============================================================================

from numpy import datetime_as_string as datetime_as_string
from numpy import datetime_data as datetime_data
from numpy import busdaycalendar as busdaycalendar
from numpy import busday_offset as busday_offset
from numpy import busday_count as busday_count
from numpy import is_busday as is_busday

# =============================================================================
# SPECIAL ARRAY TYPES - Re-export unchanged
# =============================================================================

from numpy import memmap as memmap
from numpy import matrix as matrix
from numpy import recarray as recarray
from numpy import record as record

# =============================================================================
# POLYNOMIAL FUNCTIONS - Re-export unchanged
# =============================================================================

from numpy import poly1d as poly1d
from numpy import polyval as polyval
from numpy import polyadd as polyadd
from numpy import polysub as polysub
from numpy import polymul as polymul
from numpy import polydiv as polydiv
from numpy import polyder as polyder
from numpy import polyint as polyint

# =============================================================================
# ADDITIONAL MODULES - Re-export unchanged
# =============================================================================

from numpy import ma as ma
from numpy import rec as rec
from numpy import lib as lib
from numpy import testing as testing
from numpy import exceptions as exceptions
from numpy import dtypes as dtypes

# =============================================================================
# CONSTANTS - Re-export unchanged
# =============================================================================

from numpy import pi as pi
from numpy import e as e
from numpy import euler_gamma as euler_gamma
from numpy import inf as inf
from numpy import nan as nan
from numpy import newaxis as newaxis

# =============================================================================
# LINEAR ALGEBRA - Shielded functions
# =============================================================================


@raises(LinAlgError, ValueError, TypeError)
def linalg_inv(a: ndarray) -> ndarray:
    """Compute the inverse of a matrix.

    Returns Result[ndarray, LinAlgError | ValueError | TypeError]

    Raises:
        LinAlgError: If the matrix is singular or not square
        ValueError: If the input is not a valid matrix
        TypeError: If the input type is invalid
    """
    return _np.linalg.inv(a)


@raises(LinAlgError, ValueError, TypeError)
def linalg_solve(a: ndarray, b: ndarray) -> ndarray:
    """Solve a linear matrix equation.

    Returns Result[ndarray, LinAlgError | ValueError | TypeError]

    Raises:
        LinAlgError: If the matrix is singular
        ValueError: If the shapes are incompatible
        TypeError: If the input types are invalid
    """
    return _np.linalg.solve(a, b)


@raises(LinAlgError, ValueError, TypeError)
def linalg_det(a: ndarray) -> Any:
    """Compute the determinant of a matrix.

    Returns Result[scalar, LinAlgError | ValueError | TypeError]

    Raises:
        LinAlgError: If the matrix is not square
        ValueError: If the input is not a valid matrix
        TypeError: If the input type is invalid
    """
    return _np.linalg.det(a)


@raises(LinAlgError, ValueError, TypeError)
def linalg_eig(a: ndarray) -> tuple[ndarray, ndarray]:
    """Compute eigenvalues and eigenvectors of a square matrix.

    Returns Result[tuple[eigenvalues, eigenvectors], LinAlgError | ValueError | TypeError]

    Raises:
        LinAlgError: If the eigenvalue computation does not converge
        ValueError: If the matrix is not square
        TypeError: If the input type is invalid
    """
    return _np.linalg.eig(a)


@raises(LinAlgError, ValueError, TypeError)
def linalg_eigh(a: ndarray, UPLO: Literal["L", "U"] = "L") -> tuple[ndarray, ndarray]:
    """Compute eigenvalues and eigenvectors of a Hermitian matrix.

    Returns Result[tuple[eigenvalues, eigenvectors], LinAlgError | ValueError | TypeError]
    """
    return _np.linalg.eigh(a, UPLO=UPLO)


@raises(LinAlgError, ValueError, TypeError)
def linalg_eigvals(a: ndarray) -> ndarray:
    """Compute eigenvalues of a square matrix.

    Returns Result[ndarray, LinAlgError | ValueError | TypeError]
    """
    return _np.linalg.eigvals(a)


@raises(LinAlgError, ValueError, TypeError)
def linalg_eigvalsh(a: ndarray, UPLO: Literal["L", "U"] = "L") -> ndarray:
    """Compute eigenvalues of a Hermitian matrix.

    Returns Result[ndarray, LinAlgError | ValueError | TypeError]
    """
    return _np.linalg.eigvalsh(a, UPLO=UPLO)


@raises(LinAlgError, ValueError, TypeError)
def linalg_svd(
    a: ndarray,
    full_matrices: bool = True,
    compute_uv: bool = True,
    hermitian: bool = False,
) -> tuple[ndarray, ndarray, ndarray] | ndarray:
    """Compute the singular value decomposition.

    Returns Result[tuple[U, S, Vh] | S, LinAlgError | ValueError | TypeError]
    """
    return _np.linalg.svd(a, full_matrices=full_matrices, compute_uv=compute_uv, hermitian=hermitian)


@raises(LinAlgError, ValueError, TypeError)
def linalg_pinv(a: ndarray, rcond: float = 1e-15, hermitian: bool = False) -> ndarray:
    """Compute the Moore-Penrose pseudo-inverse.

    Returns Result[ndarray, LinAlgError | ValueError | TypeError]
    """
    return _np.linalg.pinv(a, rcond=rcond, hermitian=hermitian)


@raises(LinAlgError, ValueError, TypeError)
def linalg_cholesky(a: ndarray, *, upper: bool = False) -> ndarray:
    """Compute the Cholesky decomposition.

    Returns Result[ndarray, LinAlgError | ValueError | TypeError]

    Raises:
        LinAlgError: If the matrix is not positive definite
    """
    return _np.linalg.cholesky(a, upper=upper)


@raises(LinAlgError, ValueError, TypeError)
def linalg_qr(a: ndarray, mode: Literal["reduced", "complete", "r", "raw"] = "reduced") -> tuple[ndarray, ndarray] | ndarray:
    """Compute the QR decomposition.

    Returns Result[tuple[Q, R] | R, LinAlgError | ValueError | TypeError]
    """
    return _np.linalg.qr(a, mode=mode)


@raises(LinAlgError, ValueError, TypeError)
def linalg_lstsq(a: ndarray, b: ndarray, rcond: float | None = None) -> tuple[ndarray, ndarray, int, ndarray]:
    """Compute the least-squares solution to a linear matrix equation.

    Returns Result[tuple[solution, residuals, rank, singular_values], LinAlgError | ValueError | TypeError]
    """
    return _np.linalg.lstsq(a, b, rcond=rcond)


@raises(LinAlgError, ValueError, TypeError)
def linalg_matrix_rank(a: ndarray, tol: float | None = None, hermitian: bool = False) -> int:
    """Compute the matrix rank.

    Returns Result[int, LinAlgError | ValueError | TypeError]
    """
    return _np.linalg.matrix_rank(a, tol=tol, hermitian=hermitian)


@raises(ValueError, TypeError)
def linalg_norm(
    x: ndarray,
    ord: int | float | Literal["fro", "nuc"] | None = None,
    axis: int | tuple[int, int] | None = None,
    keepdims: bool = False,
) -> ndarray | float:
    """Compute the matrix or vector norm.

    Returns Result[ndarray | float, ValueError | TypeError]
    """
    return _np.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)


@raises(ValueError, TypeError)
def linalg_cond(x: ndarray, p: int | float | Literal["fro", "nuc"] | None = None) -> float:
    """Compute the condition number of a matrix.

    Returns Result[float, ValueError | TypeError]
    """
    return _np.linalg.cond(x, p=p)


# =============================================================================
# FILE I/O - Shielded functions
# =============================================================================


@raises(FileNotFoundError, ValueError, IOError, PermissionError)
def load_npy(
    file: str,
    *,
    mmap_mode: Literal["r+", "r", "w+", "c"] | None = None,
    allow_pickle: bool = False,
    fix_imports: bool = True,
    encoding: str = "ASCII",
) -> Any:
    """Load arrays or pickled objects from .npy, .npz or pickled files.

    Returns Result[Any, FileNotFoundError | ValueError | IOError | PermissionError]

    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the file format is invalid
        IOError: If there's an I/O error
        PermissionError: If the file cannot be read
    """
    return _np.load(file, mmap_mode=mmap_mode, allow_pickle=allow_pickle, fix_imports=fix_imports, encoding=encoding)


@raises(IOError, ValueError, PermissionError, TypeError)
def save_npy(file: str, arr: ndarray, allow_pickle: bool = True) -> None:
    """Save an array to a binary file in NumPy .npy format.

    Returns Result[None, IOError | ValueError | PermissionError | TypeError]

    Raises:
        IOError: If there's an I/O error
        ValueError: If the array cannot be saved
        PermissionError: If the file cannot be written
        TypeError: If the array type is invalid
    """
    return _np.save(file, arr, allow_pickle=allow_pickle)


@raises(IOError, ValueError, PermissionError)
def savez_npy(file: str, *args: ndarray, **kwargs: ndarray) -> None:
    """Save several arrays into a single file in uncompressed .npz format.

    Returns Result[None, IOError | ValueError | PermissionError]
    """
    return _np.savez(file, *args, **kwargs)


@raises(IOError, ValueError, PermissionError)
def savez_compressed_npy(file: str, *args: ndarray, **kwargs: ndarray) -> None:
    """Save several arrays into a single file in compressed .npz format.

    Returns Result[None, IOError | ValueError | PermissionError]
    """
    return _np.savez_compressed(file, *args, **kwargs)


@raises(FileNotFoundError, ValueError, IOError, PermissionError)
def loadtxt_file(
    fname: str,
    *,
    dtype: Any = float,
    comments: str | Sequence[str] = "#",
    delimiter: str | None = None,
    converters: dict[int, Any] | None = None,
    skiprows: int = 0,
    usecols: int | Sequence[int] | None = None,
    unpack: bool = False,
    ndmin: int = 0,
    encoding: str = "bytes",
    max_rows: int | None = None,
) -> ndarray:
    """Load data from a text file.

    Returns Result[ndarray, FileNotFoundError | ValueError | IOError | PermissionError]

    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the data cannot be parsed
        IOError: If there's an I/O error
        PermissionError: If the file cannot be read
    """
    return _np.loadtxt(
        fname,
        dtype=dtype,
        comments=comments,
        delimiter=delimiter,
        converters=converters,
        skiprows=skiprows,
        usecols=usecols,
        unpack=unpack,
        ndmin=ndmin,
        encoding=encoding,
        max_rows=max_rows,
    )


@raises(IOError, ValueError, PermissionError)
def savetxt_file(
    fname: str,
    X: ndarray,
    fmt: str = "%.18e",
    delimiter: str = " ",
    newline: str = "\n",
    header: str = "",
    footer: str = "",
    comments: str = "# ",
    encoding: str | None = None,
) -> None:
    """Save an array to a text file.

    Returns Result[None, IOError | ValueError | PermissionError]

    Raises:
        IOError: If there's an I/O error
        ValueError: If the array cannot be formatted
        PermissionError: If the file cannot be written
    """
    return _np.savetxt(
        fname, X, fmt=fmt, delimiter=delimiter, newline=newline, header=header, footer=footer, comments=comments, encoding=encoding
    )


@raises(FileNotFoundError, ValueError, IOError, PermissionError)
def genfromtxt_file(
    fname: str,
    *,
    dtype: Any = float,
    comments: str = "#",
    delimiter: str | int | Sequence[int] | None = None,
    skip_header: int = 0,
    skip_footer: int = 0,
    converters: dict[int, Any] | None = None,
    missing_values: str | Sequence[str] | None = None,
    filling_values: Any = None,
    usecols: int | Sequence[int] | None = None,
    names: bool | str | Sequence[str] | None = None,
    excludelist: Sequence[str] | None = None,
    deletechars: str = r" !#$%&'()*+,-./:;<=>?@[\]^{|}~",
    replace_space: str = "_",
    autostrip: bool = False,
    case_sensitive: bool | Literal["upper", "lower"] = True,
    defaultfmt: str = "f%i",
    unpack: bool | None = None,
    usemask: bool = False,
    loose: bool = True,
    invalid_raise: bool = True,
    max_rows: int | None = None,
    encoding: str = "bytes",
    ndmin: int = 0,
) -> ndarray:
    """Load data from a text file with missing values handled.

    Returns Result[ndarray, FileNotFoundError | ValueError | IOError | PermissionError]
    """
    return _np.genfromtxt(
        fname,
        dtype=dtype,
        comments=comments,
        delimiter=delimiter,
        skip_header=skip_header,
        skip_footer=skip_footer,
        converters=converters,
        missing_values=missing_values,
        filling_values=filling_values,
        usecols=usecols,
        names=names,
        excludelist=excludelist,
        deletechars=deletechars,
        replace_space=replace_space,
        autostrip=autostrip,
        case_sensitive=case_sensitive,
        defaultfmt=defaultfmt,
        unpack=unpack,
        usemask=usemask,
        loose=loose,
        invalid_raise=invalid_raise,
        max_rows=max_rows,
        encoding=encoding,
        ndmin=ndmin,
    )


@raises(FileNotFoundError, ValueError, IOError, PermissionError)
def fromfile_binary(
    file: str,
    dtype: Any = float,
    count: int = -1,
    sep: str = "",
    offset: int = 0,
) -> ndarray:
    """Construct an array from data in a file.

    Returns Result[ndarray, FileNotFoundError | ValueError | IOError | PermissionError]
    """
    return _np.fromfile(file, dtype=dtype, count=count, sep=sep, offset=offset)


# =============================================================================
# POLYNOMIAL - Shielded functions (can fail with ill-conditioned data)
# =============================================================================


@raises(LinAlgError, ValueError, TypeError)
def polyfit(x: ndarray, y: ndarray, deg: int, rcond: float | None = None, full: bool = False, w: ndarray | None = None, cov: bool | Literal["unscaled"] = False) -> ndarray | tuple[ndarray, ...]:
    """Fit a polynomial to data.

    Returns Result[ndarray | tuple, LinAlgError | ValueError | TypeError]

    Raises:
        LinAlgError: If the fit is poorly conditioned
        ValueError: If the input arrays are invalid
        TypeError: If the input types are invalid
    """
    return _np.polyfit(x, y, deg, rcond=rcond, full=full, w=w, cov=cov)


@raises(LinAlgError, ValueError)
def roots(p: ndarray) -> ndarray:
    """Return the roots of a polynomial with given coefficients.

    Returns Result[ndarray, LinAlgError | ValueError]

    Raises:
        LinAlgError: If the eigenvalue computation fails
        ValueError: If the polynomial is empty or invalid
    """
    return _np.roots(p)


@raises(ValueError, TypeError)
def poly(seq_of_zeros: ndarray) -> ndarray:
    """Find the coefficients of a polynomial with given roots.

    Returns Result[ndarray, ValueError | TypeError]
    """
    return _np.poly(seq_of_zeros)


# =============================================================================
# STRING/BYTES - Re-export char module
# =============================================================================

from numpy import char as char
from numpy import strings as strings


# =============================================================================
# ORIGINAL MODULES - Escape hatch for direct access
# =============================================================================

linalg = _np.linalg

# =============================================================================
# __all__ - Public API
# =============================================================================

__all__ = [
    # Version info
    "__shield_version__",
    "__numpy_version__",
    # Exceptions
    "LinAlgError",
    "AxisError",
    "DTypePromotionError",
    # Types
    "ndarray",
    "dtype",
    "generic",
    "number",
    "integer",
    "floating",
    "complexfloating",
    # Array creation
    "array",
    "asarray",
    "asanyarray",
    "asarray_chkfinite",
    "zeros",
    "ones",
    "empty",
    "full",
    "arange",
    "linspace",
    "logspace",
    "geomspace",
    "eye",
    "identity",
    "diag",
    "diagflat",
    "tri",
    "tril",
    "triu",
    "vander",
    "zeros_like",
    "ones_like",
    "empty_like",
    "full_like",
    "copy",
    "frombuffer",
    "fromfunction",
    "fromiter",
    "fromstring",
    "ascontiguousarray",
    "asfortranarray",
    "require",
    "indices",
    "from_dlpack",
    # Array manipulation
    "reshape",
    "ravel",
    "transpose",
    "swapaxes",
    "moveaxis",
    "rollaxis",
    "expand_dims",
    "squeeze",
    "concatenate",
    "stack",
    "vstack",
    "hstack",
    "dstack",
    "column_stack",
    "row_stack",
    "split",
    "hsplit",
    "vsplit",
    "dsplit",
    "array_split",
    "tile",
    "repeat",
    "flip",
    "fliplr",
    "flipud",
    "roll",
    "rot90",
    "atleast_1d",
    "atleast_2d",
    "atleast_3d",
    "broadcast_to",
    "broadcast_arrays",
    "append",
    "insert",
    "delete",
    "resize",
    "trim_zeros",
    "pad",
    "shape",
    "size",
    "ndim",
    "block",
    "copyto",
    "putmask",
    "permute_dims",
    "unstack",
    "matrix_transpose",
    "astype",
    # Math operations
    "add",
    "subtract",
    "multiply",
    "divide",
    "true_divide",
    "floor_divide",
    "mod",
    "fmod",
    "remainder",
    "divmod",
    "power",
    "pow",
    "float_power",
    "sqrt",
    "cbrt",
    "square",
    "reciprocal",
    "positive",
    "negative",
    "abs",
    "absolute",
    "fabs",
    "sign",
    "heaviside",
    "exp",
    "exp2",
    "expm1",
    "log",
    "log2",
    "log10",
    "log1p",
    "logaddexp",
    "logaddexp2",
    "sin",
    "cos",
    "tan",
    "arcsin",
    "arccos",
    "arctan",
    "arctan2",
    "asin",
    "acos",
    "atan",
    "atan2",
    "hypot",
    "sinh",
    "cosh",
    "tanh",
    "arcsinh",
    "arccosh",
    "arctanh",
    "asinh",
    "acosh",
    "atanh",
    "degrees",
    "radians",
    "deg2rad",
    "rad2deg",
    "unwrap",
    "floor",
    "ceil",
    "trunc",
    "round",
    "around",
    "rint",
    "fix",
    "clip",
    "minimum",
    "maximum",
    "fmin",
    "fmax",
    "modf",
    "ldexp",
    "frexp",
    "copysign",
    "nextafter",
    "spacing",
    "signbit",
    "sinc",
    "i0",
    "nan_to_num",
    "real_if_close",
    "interp",
    "angle",
    "real",
    "imag",
    "conj",
    "conjugate",
    "gcd",
    "lcm",
    "bitwise_count",
    "trapezoid",
    # Aggregation
    "sum",
    "prod",
    "mean",
    "std",
    "var",
    "min",
    "max",
    "ptp",
    "amax",
    "amin",
    "argmin",
    "argmax",
    "nansum",
    "nanprod",
    "nanmean",
    "nanstd",
    "nanvar",
    "nanmin",
    "nanmax",
    "nanargmin",
    "nanargmax",
    "cumsum",
    "cumprod",
    "cumulative_sum",
    "cumulative_prod",
    "nancumsum",
    "nancumprod",
    "diff",
    "ediff1d",
    "gradient",
    "cross",
    "percentile",
    "nanpercentile",
    "quantile",
    "nanquantile",
    "median",
    "nanmedian",
    "average",
    "histogram",
    "histogram2d",
    "histogramdd",
    "histogram_bin_edges",
    "bincount",
    "digitize",
    "cov",
    "corrcoef",
    "correlate",
    "convolve",
    # Comparison
    "equal",
    "not_equal",
    "less",
    "less_equal",
    "greater",
    "greater_equal",
    "isnan",
    "isinf",
    "isfinite",
    "isneginf",
    "isposinf",
    "isclose",
    "allclose",
    "array_equal",
    "array_equiv",
    "iscomplex",
    "isreal",
    "isnat",
    "isfortran",
    "iterable",
    # Logic
    "all",
    "any",
    "logical_and",
    "logical_or",
    "logical_not",
    "logical_xor",
    "where",
    "nonzero",
    "argwhere",
    "flatnonzero",
    "count_nonzero",
    # Bitwise operations
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "invert",
    "bitwise_invert",
    "bitwise_not",
    "left_shift",
    "right_shift",
    "bitwise_left_shift",
    "bitwise_right_shift",
    "packbits",
    "unpackbits",
    "binary_repr",
    # Indexing and slicing
    "take",
    "put",
    "choose",
    "select",
    "compress",
    "extract",
    "place",
    "put_along_axis",
    "take_along_axis",
    "diagonal",
    "diag_indices",
    "diag_indices_from",
    "tril_indices",
    "tril_indices_from",
    "triu_indices",
    "triu_indices_from",
    "mask_indices",
    "fill_diagonal",
    "unravel_index",
    "ravel_multi_index",
    "ix_",
    # Matrix operations
    "dot",
    "vdot",
    "inner",
    "outer",
    "matmul",
    "tensordot",
    "einsum",
    "einsum_path",
    "kron",
    "trace",
    "matvec",
    "vecdot",
    "vecmat",
    # Window functions
    "bartlett",
    "blackman",
    "hamming",
    "hanning",
    "kaiser",
    # Mesh/grid functions
    "meshgrid",
    "mgrid",
    "ogrid",
    # Sorting
    "sort",
    "argsort",
    "lexsort",
    "partition",
    "argpartition",
    "searchsorted",
    "unique",
    "unique_all",
    "unique_counts",
    "unique_inverse",
    "unique_values",
    "sort_complex",
    # Set operations
    "intersect1d",
    "union1d",
    "setdiff1d",
    "setxor1d",
    "isin",
    # Type functions
    "can_cast",
    "promote_types",
    "result_type",
    "min_scalar_type",
    "common_type",
    "issubdtype",
    "iscomplexobj",
    "isrealobj",
    "isscalar",
    "isdtype",
    "mintypecode",
    # Dtype types
    "bool_",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float16",
    "float32",
    "float64",
    "complex64",
    "complex128",
    "intp",
    "uintp",
    "intc",
    "uintc",
    "longlong",
    "ulonglong",
    "single",
    "double",
    "longdouble",
    "csingle",
    "cdouble",
    "clongdouble",
    "object_",
    "bytes_",
    "str_",
    "void",
    "datetime64",
    "timedelta64",
    "byte",
    "ubyte",
    "short",
    "ushort",
    "int_",
    "uint",
    "long",
    "ulong",
    "half",
    "character",
    "flexible",
    "inexact",
    "signedinteger",
    "unsignedinteger",
    # Memory and info
    "shares_memory",
    "may_share_memory",
    "finfo",
    "iinfo",
    "base_repr",
    # Functional operations
    "apply_along_axis",
    "apply_over_axes",
    "vectorize",
    "frompyfunc",
    "piecewise",
    # Indexing helpers
    "c_",
    "r_",
    "s_",
    "index_exp",
    # Iterator types
    "nditer",
    "ndenumerate",
    "ndindex",
    "flatiter",
    "broadcast",
    "broadcast_shapes",
    # Error handling
    "errstate",
    "seterr",
    "geterr",
    "seterrcall",
    "geterrcall",
    # Print options
    "set_printoptions",
    "get_printoptions",
    "printoptions",
    "array2string",
    "array_repr",
    "array_str",
    "format_float_positional",
    "format_float_scientific",
    # Datetime
    "datetime_as_string",
    "datetime_data",
    "busdaycalendar",
    "busday_offset",
    "busday_count",
    "is_busday",
    # Special array types
    "memmap",
    "matrix",
    "recarray",
    "record",
    # Polynomial functions
    "poly1d",
    "polyval",
    "polyadd",
    "polysub",
    "polymul",
    "polydiv",
    "polyder",
    "polyint",
    # Submodules
    "random",
    "fft",
    "linalg",
    "polynomial",
    "char",
    "strings",
    "ma",
    "rec",
    "lib",
    "testing",
    "exceptions",
    "dtypes",
    # Constants
    "pi",
    "e",
    "euler_gamma",
    "inf",
    "nan",
    "newaxis",
    # Shielded linear algebra
    "linalg_inv",
    "linalg_solve",
    "linalg_det",
    "linalg_eig",
    "linalg_eigh",
    "linalg_eigvals",
    "linalg_eigvalsh",
    "linalg_svd",
    "linalg_pinv",
    "linalg_cholesky",
    "linalg_qr",
    "linalg_lstsq",
    "linalg_matrix_rank",
    "linalg_norm",
    "linalg_cond",
    # Shielded file I/O
    "load_npy",
    "save_npy",
    "savez_npy",
    "savez_compressed_npy",
    "loadtxt_file",
    "savetxt_file",
    "genfromtxt_file",
    "fromfile_binary",
    # Shielded polynomial
    "polyfit",
    "roots",
    "poly",
]
