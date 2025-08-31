import numpy as np
from numpy.typing import NDArray

from numba import njit

@njit
def bb_map_forward(
    x: NDArray[np.float64],
    parameters: NDArray[np.float64],
    epsilon: float = 1.0e-18
) -> NDArray[np.float64]:
    """
    Beam-beam map (forward)
    
    """
    q, p = x + epsilon
    nu, xi, ks = parameters
    c = np.cos(2.0*np.pi*nu)
    s = np.sin(2.0*np.pi*nu)
    Q = p
    P = -q + 2.0*c*p + 8.0*np.pi*xi*s/p*(np.exp(-0.5*p*p) - 1.0) + s*ks*p*p
    return np.stack((Q, P))


@njit
def bb_map_inverse(
    x: NDArray[np.float64],
    parameters: NDArray[np.float64],
    epsilon: float = 1.0e-18
) -> NDArray[np.float64]:
    """
    Beam-beam map (inverse)
    
    """
    q, p = x + epsilon
    nu, xi, ks = parameters
    c = np.cos(2.0*np.pi*nu)
    s = np.sin(2.0*np.pi*nu)
    Q = -p + 2.0*c*q + 8.0*np.pi*xi*s/q*(np.exp(-0.5*q*q) - 1.0) + s*ks*q*q
    P = q
    return np.stack((Q, P))


@njit
def bb_map_diagonal_symmetry(
    q: NDArray[np.float64],
    parameters: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Diagonal symmetry line
    
    """
    nu, xi, ks = parameters
    p = q
    return p


@njit
def bb_map_force_symmetry(
    q: NDArray[np.float64],
    parameters: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Force symmetry line
    
    """
    nu, xi, ks = parameters
    nu, xi, ks = parameters
    c = np.cos(2.0*np.pi*nu)
    s = np.sin(2.0*np.pi*nu)    
    p = 0.5*(2.0*q*c + (ks*q*q + (8.0*np.pi*xi)*(np.exp(-0.5*q*q) - 1.0)/q)*s)
    return p
