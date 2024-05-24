from reduced_state_caldeira_leggett.plot import (
    plot_system_eigenstates,
    plot_thermal_occupation,
)
from reduced_state_caldeira_leggett.system import (
    HYDROGEN_NICKEL_SYSTEM,
    SimulationConfig,
)

if __name__ == "__main__":
    system = HYDROGEN_NICKEL_SYSTEM
    config = SimulationConfig(shape=(4,), resolution=(30,), n_bands=2)
    plot_thermal_occupation(system, config)
    plot_system_eigenstates(system, config)
