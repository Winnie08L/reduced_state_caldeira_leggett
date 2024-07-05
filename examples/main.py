from reduced_state_caldeira_leggett.plot import (
    plot_basis_states,
    plot_initial_state,
    plot_state_against_t,
)
from reduced_state_caldeira_leggett.system import (
    FREE_LITHIUM_SYSTEM,
    SimulationConfig,
)

if __name__ == "__main__":
    system = FREE_LITHIUM_SYSTEM
    config = SimulationConfig(shape=(2,), resolution=(31,), n_bands=3, type="bloch")

    plot_basis_states(system, config)
    plot_state_against_t(system, config, n=1000, step=500)
    # plot_kernel(system, config, temperature=0.0001)
    # plot_lindblad_operator(system, config)
    # plot_thermal_occupation(system, config)
    # plot_system_eigenstates(system, config)
    # plot_stochastic_occupation(system, config, n=1000, step=500)
    plot_initial_state(system, config)
    # plot_noise_operator(system, config, temperature=150.0)
