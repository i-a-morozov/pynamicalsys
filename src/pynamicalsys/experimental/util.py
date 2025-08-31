"""
FP utils
--------

"""
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from numba import njit


@njit
def trajectory(
    length:int, 
    mapping:Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    state:NDArray[np.float64],
    parameters:NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Generate trajectories

    Parameters
    ----------
    
    length: int
        trajectory length
    mapping: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]
        mapping
    state: NDArray[np.float64]
        state
    parameters: NDArray[np.float64]
        additional mapping parameters

    Returns
    -------
    NDArray[np.float64]
        trajectory

    """
    local = np.copy(state)
    table = np.empty((length, *state.shape), dtype=np.float64)
    for i in range(length):
        local = mapping(local, parameters)
        table[i] = local
    return table


def exact(
    order: int,
    mapping: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    points: NDArray[np.float64],
    parameters: NDArray[np.float64],
    tolerance: float=1.0E-9
) -> NDArray[np.bool_]:
    """
    Exact fixed point test

    Parameters
    ----------
    order: int, positive
        fixed point order
    mapping: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]
        mapping
    points: NDArray[np.float64]
        fixed points
    parameters: NDArray[np.float64]
        additional mapping parameters
    tolerance: float, default=1.0E-9
        tolerance

    Returns
    -------
    NDArray[np.bool_]
        logical mask

    """
    orbits = trajectory(order, mapping, points.T, parameters).T
    counts = np.empty(len(points), dtype=np.int64)
    for i in range(len(points)):
        counts[i] = np.isclose(orbits[i].T, points[i], rtol=tolerance, atol=tolerance).all(axis=-1).sum()
    return counts == 1


def canonize(
    chain:NDArray[np.float64],
    tolerance:float=1.0E-9,
    reverse:bool=True
) -> NDArray[np.float64]:
    """
    Periodic chain canonization (canonical starting point)

    Parameters
    ----------
    chain: NDArray[np.float64]
        chain
    tol: float, default=1.0E-9
        tolerance
    reverse: bool, default=True
        reverse

    Returns
    -------
    NDArray[np.float64]

    """
    length, _ = chain.shape
    rotations = np.empty(((1 + reverse)*length, *chain.shape), dtype=np.float64)
    local = np.copy(chain)
    for i in range(length):
        rotations[i] = local
        local = np.roll(local, -1, axis=0)
    if reverse:
        local = np.flip(chain, axis=0)
        for i in range(length):
            rotations[i + length] = local
            local = np.roll(local, -1, axis=0)
    size, *_ = rotations.shape
    flat = rotations.reshape(size, -1)
    keys = np.round(flat/tolerance).astype(np.int64)
    idx, *_ = np.lexsort(keys.T[::-1])
    start, *_ = rotations[idx]
    return start


def unique(
    order: int,
    mapping: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
    points: NDArray[np.float64],
    parameters: NDArray[np.float64],
    tolerance: float=1.0E-9,
    reverse:bool=True
) -> NDArray[np.bool_]:
    """
    Create unique mask

    Parameters
    ----------
    order: int, positive
        fixed point order
    mapping: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]
        mapping
    points: NDArray[np.float64]
        fixed points
    parameters: NDArray[np.float64]
        additional mapping parameters
    tolerance: float, default=1.0E-9
        tolerance
    reverse: bool, default=True
        include reverse rotations flag

    Returns
    -------
    NDArray[np.bool_]
        logical mask

    """
    chains = trajectory(order, mapping, points.T, parameters).T
    starts = np.stack([canonize(chain.T, tolerance=tolerance, reverse=reverse) for chain in chains])
    matrix = (starts * starts).sum(-1)
    matrix = matrix.reshape(-1, 1) + matrix - 2.0*(starts @ starts.T)
    return np.logical_not(np.any(np.triu(matrix <= tolerance, k=1), axis=0))
