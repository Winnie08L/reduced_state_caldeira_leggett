from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from surface_potential_analysis.basis.explicit_basis import (
    explicit_stacked_basis_as_fundamental,
)
from surface_potential_analysis.kernel.kernel import as_diagonal_kernel
from surface_potential_analysis.kernel.plot import (
    plot_diagonal_kernel,
    plot_kernel_truncation_error,
)
from surface_potential_analysis.operator.operator_list import select_operator
from surface_potential_analysis.operator.plot import (
    plot_eigenstate_occupations,
    plot_operator_2d,
)
from surface_potential_analysis.potential.plot import plot_potential_1d_x
from surface_potential_analysis.state_vector.conversion import (
    convert_state_vector_list_to_basis,
)
from surface_potential_analysis.state_vector.eigenstate_calculation import (
    calculate_eigenvectors_hermitian,
)
from surface_potential_analysis.state_vector.plot import (
    plot_state_1d_k,
    plot_state_1d_x,
)
from surface_potential_analysis.state_vector.state_vector_list import (
    state_vector_list_into_iter,
)

from reduced_state_caldeira_leggett.system import (
    PeriodicSystem,
    SimulationConfig,
    get_extended_interpolated_potential,
    get_hamiltonian,
    get_noise_kernel,
    get_noise_operators,
)


def plot_system_eigenstates(
    system: PeriodicSystem,
    config: SimulationConfig,
) -> None:
    """Plot the potential against position."""
    potential = get_extended_interpolated_potential(
        system,
        config.shape,
        config.resolution,
    )
    fig, ax, _ = plot_potential_1d_x(potential)

    hamiltonian = get_hamiltonian(system, config)
    eigenvectors = calculate_eigenvectors_hermitian(hamiltonian)
    basis = explicit_stacked_basis_as_fundamental(hamiltonian["basis"][0])
    converted = convert_state_vector_list_to_basis(eigenvectors, basis)

    ax1 = ax.twinx()
    fig2, ax2 = plt.subplots()
    for _i, state in enumerate(state_vector_list_into_iter(converted)):
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
    potential = get_extended_interpolated_potential(
        system,
        config.shape,
        config.resolution,
    )
    fig, ax, _ = plot_potential_1d_x(potential)

    hamiltonian = get_hamiltonian(system, config)
    states = hamiltonian["basis"][0].vectors

    ax1 = ax.twinx()
    fig2, ax2 = plt.subplots()
    for i, state in enumerate(state_vector_list_into_iter(states)):
        _, _, line = plot_state_1d_x(state, ax=ax1)
        line.set_label(f"state {i}")

        plot_state_1d_k(state, ax=ax2)

    fig.show()
    fig.legend()
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
    kernel = get_noise_kernel(system, config, 150)
    diagonal = as_diagonal_kernel(kernel)

    fig, _, _ = plot_diagonal_kernel(diagonal)
    fig.show()

    fig, _ = plot_kernel_truncation_error(kernel)
    fig.show()

    input()


def plot_lindblad_operator(
    system: PeriodicSystem,
    config: SimulationConfig,
    *,
    temperature: float = 155,
) -> None:
    operators = get_noise_operators(system, config, temperature)

    args = np.argsort(np.abs(operators["eigenvalue"]))[::-1]

    for idx in args[:10]:
        operator = select_operator(operators, idx=idx)

        fig, ax, _ = plot_operator_2d(operator)
        ax.set_title("Operator")
        fig.show()

    input()
