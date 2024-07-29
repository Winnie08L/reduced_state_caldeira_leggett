from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
from matplotlib import pyplot as plt
from surface_potential_analysis.basis.stacked_basis import TupleBasisWithLengthLike
from surface_potential_analysis.kernel.gaussian import (
    get_effective_gaussian_parameters,
    get_gaussian_isotropic_noise_kernel,
)
from surface_potential_analysis.kernel.kernel import (
    IsotropicNoiseKernel,
    as_diagonal_kernel,
    as_isotropic_kernel,
    as_noise_kernel,
    get_diagonal_noise_kernel,
)
from surface_potential_analysis.kernel.kernel import (
    get_noise_kernel as get_noise_kernel_generic,
)
from surface_potential_analysis.kernel.plot import (
    plot_diagonal_kernel,
    plot_kernel_truncation_error,
)
from surface_potential_analysis.kernel.plot import plot_kernel as plot_kernel_generic
from surface_potential_analysis.operator.operator import as_operator
from surface_potential_analysis.operator.operator_list import (
    select_operator,
    select_operator_diagonal,
)
from surface_potential_analysis.operator.plot import (
    plot_eigenstate_occupations,
    plot_operator_2d,
    plot_operator_along_diagonal,
)
from surface_potential_analysis.potential.plot import (
    plot_potential_1d_x,
    plot_potential_2d_x,
)

from surface_potential_analysis.stacked_basis.conversion import (
    stacked_basis_as_fundamental_position_basis,
)
from surface_potential_analysis.stacked_basis.util import get_x_coordinates_in_axes
from surface_potential_analysis.state_vector.conversion import (
    convert_state_vector_list_to_basis,
)

from surface_potential_analysis.state_vector.eigenstate_calculation import (
    calculate_eigenvectors_hermitian,
)
from surface_potential_analysis.state_vector.plot import (
    animate_state_over_list_1d_x,
    plot_average_band_occupation,
    plot_state_1d_k,
    plot_state_1d_x,
)
from surface_potential_analysis.state_vector.state_vector_list import (
    state_vector_list_into_iter,
)
from surface_potential_analysis.util.plot import Scale, build_animation, plot_data_1d
from surface_potential_analysis.util.util import (
    Measure,
    get_data_in_axes,
    get_measured_data,
)

from reduced_state_caldeira_leggett.dynamics import (
    get_initial_state,
    get_stochastic_evolution,
)
from reduced_state_caldeira_leggett.system import (
    PeriodicSystem,
    SimulationConfig,

    _get_full_hamiltonian,
    get_2d_111_potential,
    get_extended_interpolated_potential,

    get_hamiltonian,
    get_lorentzian_isotropic_noise_kernel,
    get_noise_kernel,
    get_noise_operators,

    solve_linear_general_isotropic_noise,
    solve_linear_lorentzian_isotropic_noise,

)

if TYPE_CHECKING:
    from matplotlib.animation import ArtistAnimation
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from surface_potential_analysis.basis.stacked_basis import TupleBasisLike
    from surface_potential_analysis.state_vector.state_vector import StateVector
    from surface_potential_analysis.types import SingleStackedIndexLike


def plot_system_eigenstates(
    system: PeriodicSystem,
    config: SimulationConfig,
) -> None:
    """Plot the potential against position."""
    potential = get_potential_1d(
        system,
        config.shape,
        config.resolution,
    )
    fig, ax, _ = plot_potential_1d_x(potential)

    hamiltonian = get_hamiltonian(system, config)
    eigenvectors = calculate_eigenvectors_hermitian(hamiltonian)

    ax1 = ax.twinx()
    fig2, ax2 = plt.subplots()
    for _i, state in enumerate(state_vector_list_into_iter(eigenvectors)):
        plot_state_1d_x(state, ax=ax1)
        plot_state_1d_k(state, ax=ax2)

    fig.show()
    fig2.show()
    input()


def plot_basis_states(
    system: PeriodicSystem,
    config: SimulationConfig,
) -> None:
    """Plot the potential against position."""
    potential = get_potential_1d(
        system,
        config.shape,
        config.resolution,
    )
    fig, ax, line = plot_potential_1d_x(potential)
    line.set_color("black")
    line.set_linewidth(3)

    hamiltonian = get_hamiltonian(system, config)
    states = hamiltonian["basis"][0].vectors

    ax1 = ax.twinx()
    fig2, ax2 = plt.subplots()
    for i, state in enumerate(state_vector_list_into_iter(states)):
        _, _, line = plot_state_1d_x(state, ax=ax1)
        line.set_label(f"state {i}")

        plot_state_1d_k(state, ax=ax2)

    fig.show()
    fig2.show()
    input()


def plot_thermal_occupation(
    system: PeriodicSystem,
    config: SimulationConfig,
) -> None:
    hamiltonian = get_hamiltonian(system, config)
    fig, _, _ = plot_eigenstate_occupations(hamiltonian, 150)

    fig.show()
    input()


def plot_kernel(
    system: PeriodicSystem,
    config: SimulationConfig,
) -> None:
    kernel = get_noise_kernel(system, config)
    diagonal = as_diagonal_kernel(kernel)

    fig, _, _ = plot_diagonal_kernel(diagonal)
    fig.show()
    input()

    fig, _, _ = plot_kernel_generic(as_noise_kernel(diagonal))
    fig.show()

    fig, _ = plot_kernel_truncation_error(kernel)
    fig.show()

    corrected_operators = get_noise_operators(system, config)
    kernel_full = get_noise_kernel_generic(corrected_operators)

    fig, _, _ = plot_kernel_generic(kernel_full)
    fig.show()

    diagonal = as_diagonal_kernel(kernel_full)
    fig, _, _ = plot_diagonal_kernel(diagonal)
    fig.show()

    input()


def plot_lindblad_operator(
    system: PeriodicSystem,
    config: SimulationConfig,
) -> None:
    operators = get_noise_operators(system, config)

    args = np.argsort(np.abs(operators["eigenvalue"]))[::-1]

    for idx in args[:10]:
        operator = select_operator(operators, idx=idx)

        fig, ax, _ = plot_operator_2d(operator)
        ax.set_title("Operator")
        fig.show()

    input()


def plot_state_against_t(
    system: PeriodicSystem,
    config: SimulationConfig,
    *,
    n: int,
    step: int,
    dt_ratio: float = 500,
) -> None:
    potential = get_potential_1d(
        system,
        config.shape,
        config.resolution,
    )
    fig, ax, line = plot_potential_1d_x(potential)
    line.set_color("black")
    line.set_linewidth(3)

    states = get_stochastic_evolution(system, config, n=n, step=step, dt_ratio=dt_ratio)

    _fig, _, _animnation_ = animate_state_over_list_1d_x(states, ax=ax.twinx())

    fig.show()
    input()


def plot_stochastic_occupation(
    system: PeriodicSystem,
    config: SimulationConfig,
    *,
    dt_ratio: float = 500,
    n: int = 800,
    step: int = 4000,
) -> None:
    states = get_stochastic_evolution(
        system,
        config,
        n=n,
        step=step,
        dt_ratio=dt_ratio,
    )
    hamiltonian = get_hamiltonian(system, config)

    fig2, ax2, line = plot_average_band_occupation(hamiltonian, states)

    for ax in [ax2]:
        _, _, line = plot_eigenstate_occupations(hamiltonian, 150, ax=ax)
        line.set_linestyle("--")
        line.set_label("Expected")

        ax.legend([line], ["Boltzmann occupation"])

    fig2.show()
    input()


def plot_initial_state(system: PeriodicSystem, config: SimulationConfig) -> None:
    initial = get_initial_state(system, config)
    fig, _ax, _ = plot_state_1d_x(initial)

    fig.show()
    input()


def plot_noise_operator(
    system: PeriodicSystem,
    config: SimulationConfig,
) -> None:
    operator = select_operator(get_noise_operators(system, config), 0)
    fig, _ax, _ = plot_operator_along_diagonal(operator)

    fig.show()
    input()


def plot_2d_111_potential(
    system: PeriodicSystem,
    config: SimulationConfig,
) -> None:

    potential = get_2d_111_potential(system, config.shape, config.resolution)
    fig, _, _ = plot_potential_2d_x(potential)
    fig.show()
    input()


_L0Inv = TypeVar("_L0Inv", bound=int)


def animate_data_2d_x(
    basis: TupleBasisLike[*tuple[Any, ...]],
    data: np.ndarray[tuple[_L0Inv], np.dtype[np.complex128]],
    axes: tuple[int, int] = (0, 1),
    idx: SingleStackedIndexLike | None = None,
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
    clim: tuple[float | None, float | None] = (None, None),
    measure: Measure = "abs",
) -> tuple[Figure, Axes, ArtistAnimation]:
    idx = tuple(0 for _ in range(basis.ndim - len(axes))) if idx is None else idx
    clim = (0.0, clim[1]) if clim[0] is None and measure == "abs" else clim

    coordinates = get_x_coordinates_in_axes(basis, axes, idx)
    data_in_axis = get_data_in_axes(data.reshape(basis.shape), axes, idx)
    measured_data = get_measured_data(data_in_axis, measure)

    fig, ax, ani = build_animation(
        lambda i, ax: ax.pcolormesh(
            *coordinates[:2, :, :, i],
            measured_data[:, :, i],
            shading="nearest",
        ),
        data.shape[2],
        ax=ax,
        scale=scale,
        clim=clim,
    )
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(ax.collections[0], ax=ax, format="%4.1e")

    ax.set_xlabel(f"x{axes[0]} axis")
    ax.set_ylabel(f"x{axes[1]} axis")
    return fig, ax, ani


def animate_state_2d_x(
    state: StateVector[TupleBasisLike[*tuple[Any, ...]]],
    axes: tuple[int, int] = (0, 1),
    idx: SingleStackedIndexLike | None = None,
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
) -> tuple[Figure, Axes, ArtistAnimation]:
    converted = convert_state_vector_list_to_basis(
        state,
        stacked_basis_as_fundamental_position_basis(state["basis"]),
    )
    return animate_data_2d_x(
        converted["basis"],
        converted["data"].reshape(converted["basis"].shape),
        axes,
        idx,
        ax=ax,
        scale=scale,
        measure="real",
    )


# def plot_2d_111_state_against_t(
#     system: PeriodicSystem,
#     config: SimulationConfig,
#     *,
#     n: int,
#     step: int,
#     dt_ratio: float = 500,
# ) -> None:
#     potential = get_2d_111_potential(system, config.shape, config.resolution)
#     fig, ax, _ = plot_potential_2d_x(potential)
#     states = get_stochastic_evolution(system, config, n=n, step=step, dt_ratio=dt_ratio)
#     _fig, _, _animation_ = animate_state_2d_x(states, ax=ax.twinx())
#     fig.show()
#     input()


_B0 = TypeVar("_B0", bound=TupleBasisWithLengthLike[Any, Any])


def plot_new_noise_operators(
    kernel: IsotropicNoiseKernel[_B0],
    *,
    n: int = 1,
) -> None:
    """Plot the noise operators generated."""
    operators = solve_linear_general_isotropic_noise(kernel, n=n)
    op = select_operator_diagonal(operators, idx=1)
    fig1, ax1, _ = plot_operator_along_diagonal(as_operator(op), measure="real")
    ax1.set_title("fitted noise operator")
    fig1.show()
    input()


def plot_isotropic_noise_kernel(
    system: PeriodicSystem,
    config: SimulationConfig,
) -> None:
    """Plot 1d general isotropic noise kernel, comparing the true one and the fitted one,
    gaussian noise is used here for testing.
    """
    hamiltonian = _get_full_hamiltonian(system, config.shape, config.resolution)
    a, lambda_ = get_effective_gaussian_parameters(
        hamiltonian["basis"][0],
        system.eta,
        config.temperature,
        lambda_factor=2 * np.sqrt(2),
    )

    basis_x = stacked_basis_as_fundamental_position_basis(hamiltonian["basis"][0])
    kernel_real = get_gaussian_isotropic_noise_kernel(basis_x, a, lambda_)
    data = kernel_real["data"].reshape(kernel_real["basis"].shape)
    fig, ax, line = plot_data_1d(
        data,
        np.arange(data.size),
        scale="linear",
        measure="real",
    )
    fig, _, line1 = plot_data_1d(
        data,
        np.arange(data.size),
        ax=ax,
        scale="linear",
        measure="imag",
    )
    line.set_label("true noise, real")
    line1.set_label("true noise, imag")
    ax.set_title("noise kernel")
    fig.show()

    # fitted noise kernel-----------------------------
    # operators = solve_linear_gaussian_isotropic_noise(system, config, n=1)
    operators = solve_linear_general_isotropic_noise(kernel_real, n=20)
    kernel = get_diagonal_noise_kernel(operators)
    kernel_isotropic = as_isotropic_kernel(kernel)
    data = kernel_isotropic["data"]
    fig, _, line2 = plot_data_1d(
        data,
        np.arange(data.size),
        ax=ax,
        scale="linear",
        measure="real",
    )
    fig, _, line3 = plot_data_1d(
        data,
        np.arange(data.size),
        ax=ax,
        scale="linear",
        measure="imag",
    )
    line2.set_label("fitted noise, real")
    line3.set_label("fitted noise, imag")
    ax.legend()
    fig.show()
    input()


def plot_isotropic_lorentzian_noise(
    system: PeriodicSystem,
    config: SimulationConfig,
) -> None:
    # 1d lorentzian---------------------------
    hamiltonian = _get_full_hamiltonian(system, config.shape, config.resolution)
    basis_x = stacked_basis_as_fundamental_position_basis(hamiltonian["basis"][0])
    kernel_lorentz = get_lorentzian_isotropic_noise_kernel(basis_x)
    data = kernel_lorentz["data"].reshape(kernel_lorentz["basis"].shape)
    fig, ax, line = plot_data_1d(
        data,
        np.arange(data.size),
        scale="linear",
        measure="real",
    )
    fig, _, line1 = plot_data_1d(
        data,
        np.arange(data.size),
        ax=ax,
        scale="linear",
        measure="imag",
    )
    line.set_label("true noise, real")
    line1.set_label("true noise, imag")
    ax.set_title("noise kernel")
    fig.show()

    operators = solve_linear_lorentzian_isotropic_noise(system, config, n=10)
    kernel = get_diagonal_noise_kernel(operators)
    kernel_isotropic = as_isotropic_kernel(kernel)
    data = kernel_isotropic["data"]
    fig, _, line2 = plot_data_1d(
        data,
        np.arange(data.size),
        ax=ax,
        scale="linear",
        measure="real",
    )
    fig, _, line3 = plot_data_1d(
        data,
        np.arange(data.size),
        ax=ax,
        scale="linear",
        measure="imag",
    )
    line2.set_label("fitted noise, real")
    line3.set_label("fitted noise, imag")
    ax.legend()

    fig.show()
    input()
