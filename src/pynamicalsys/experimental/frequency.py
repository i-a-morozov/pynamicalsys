"""
Frequency
---------

Window weighted frequency estimation for symplectic mappings

I.M., 2025

"""
from typing import Callable
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from numba import njit

@njit
def window(
    length:int,
    degree:float=1.0
) -> NDArray[np.float64]:
    """
    Generate exponential window

    Parameters
    ----------
    length: int
        window length
    degree: float, default=1.0
        window degree

    Returns
    -------
    NDArray[np.float64]

    """
    ts = np.linspace(0.0, (length - 1.0)/length, length)
    ws = np.exp(-1.0/((1.0 - ts)**degree*ts**degree))
    return ws/np.sum(ws)


@njit
def frequency(
    weights: NDArray[np.float64],
    mapping: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    state: NDArray[np.float64],
    parameters: NDArray[np.float64]
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Estimate rotation numbers

    Parameters
    ----------
    weights: NDArray[np.float64]
        window weights
    mapping: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]
        mapping
    state: NDArray[np.float64]
        state
    parameters: NDArray[np.float64]
        additional mapping parameters

    Returns
    -------
    Tuple[NDArray[np.float64], NDArray[np.float64]]

    """
    factor = 2.0*np.pi
    _, size = state.shape
    qs, ps = np.reshape(state, (2, -1, size))
    initial = np.arctan2(qs, ps)
    total = np.zeros_like(initial)
    for weight in weights:
        state = mapping(state, parameters)
        qs, ps = np.reshape(state, (2, -1, size))
        current = np.arctan2(qs, ps)
        total += weight*((current - initial) % factor)
        initial = current
    return state, total/factor


@njit
def frequencies(
    intervals: int,
    weights: NDArray[np.float64],
    mapping: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    state: NDArray[np.float64],
    parameters: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Compute frequencies over several non-overlapping intervals

    Parameters
    ----------
    intervals: int
        number of intervals
    weights: NDArray[np.float64]
        window weights
    mapping: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]
        mapping
    state: NDArray[np.float64]
        state
    parameters: NDArray[np.float64]
        additional mapping parameters

    Returns
    -------
    NDArray[np.float64]

    """
    dimension, size = state.shape
    frequencies = np.empty((intervals, dimension // 2, size), dtype=np.float64)
    for i in range(intervals):
        state, frequencies[i] = frequency(weights, mapping, state, parameters)
    return frequencies
