from reduced_state_caldeira_leggett.plot import (
    plot_basis_states,
    plot_kernel,
    plot_lindblad_operator,
    plot_system_eigenstates,
    plot_thermal_occupation,
)
from reduced_state_caldeira_leggett.system import (
    HYDROGEN_NICKEL_SYSTEM,
    SimulationConfig,
)

if __name__ == "__main__":
    system = HYDROGEN_NICKEL_SYSTEM
    config = SimulationConfig(shape=(3,), resolution=(31,), n_bands=3, type="wannier")

    plot_lindblad_operator(system, config)
    plot_kernel(system, config)
    plot_thermal_occupation(system, config)
    plot_basis_states(system, config)
    plot_system_eigenstates(system, config)
