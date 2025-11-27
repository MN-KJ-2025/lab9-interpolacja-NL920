import pickle
from typing import Any

import main
import numpy as np
import pytest

try:
    with open("expected", "rb") as f:
        expected = pickle.load(f)
except FileNotFoundError:
    print(
        "Error: The 'expected' file was not found. Please ensure it is in the correct directory."
    )
    expected = {
        "chebyshev_nodes": [],
        "bar_cheb_weights": [],
        "barycentric_inte": [],
        "L_inf": [],
    }


# --- Data Preparation ---

valid_chebyshev_nodes = [
    (n, res) for n, res in expected["chebyshev_nodes"] if res is not None
]
valid_bar_cheb_weights = [
    (n, res) for n, res in expected["bar_cheb_weights"] if res is not None
]
valid_barycentric_inte = [
    (xi, yi, wi, x, res) for xi, yi, wi, x, res in expected["barycentric_inte"] if res is not None
]
valid_L_inf = [
    (xr, x, res) for xr, x, res in expected["L_inf"] if res is not None
]


# --- Tests for chebyshev_nodes ---

@pytest.mark.parametrize("n, expected_result", valid_chebyshev_nodes)
def test_chebyshev_nodes_correct_solution(n: int, expected_result: np.ndarray):
    """Tests if chebyshev_nodes calculates the correct nodes for valid inputs."""
    actual_result = main.chebyshev_nodes(n)
    assert actual_result == pytest.approx(expected_result), (
        f"Chebyshev nodes are incorrect for n={n}."
    )


# --- Tests for bar_cheb_weights ---

@pytest.mark.parametrize("n, expected_result", valid_bar_cheb_weights)
def test_bar_cheb_weights_correct_solution(n: int, expected_result: np.ndarray):
    """Tests if bar_cheb_weights calculates the correct weights for valid inputs."""
    actual_result = main.bar_cheb_weights(n)
    assert actual_result == pytest.approx(expected_result), (
        f"Barycentric Chebyshev weights are incorrect for n={n}."
    )


# --- Tests for barycentric_inte ---

@pytest.mark.parametrize("xi, yi, wi, x, expected_result", valid_barycentric_inte)
def test_barycentric_inte_correct_solution(
    xi: np.ndarray, yi: np.ndarray, wi: np.ndarray, x: np.ndarray, expected_result: np.ndarray
):
    """Tests if barycentric_inte calculates the correct interpolated values."""
    actual_result = main.barycentric_inte(xi, yi, wi, x)
    assert actual_result == pytest.approx(expected_result), (
        "The interpolated values are incorrect."
    )


# --- Tests for L_inf ---

@pytest.mark.parametrize("xr, x, expected_result", valid_L_inf)
def test_L_inf_correct_solution(
    xr: Any, x: Any, expected_result: float
):
    """Tests if L_inf calculates the correct norm for various valid inputs."""
    actual_result = main.L_inf(xr, x)
    assert actual_result == pytest.approx(expected_result), (
        f"L_inf norm is incorrect for inputs xr={xr} and x={x}."
    )