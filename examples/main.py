from reduced_state_caldeira_leggett.plot import (
    plot_2d_111_potential,
    plot_2d_111_state_against_t,
)
from reduced_state_caldeira_leggett.system import (
    SODIUM_COPPER_SYSTEM,
    SimulationConfig,
    get_2d_111_potential,
)

if __name__ == "__main__":
    system = SODIUM_COPPER_SYSTEM
    config = SimulationConfig(
        shape=(1, 1),
        resolution=(15, 15),
        n_bands=3,
        type="bloch",
        temperature=155,
    )

    # plot_basis_states(system, config)
    # plot_state_against_t(system, config, n=1000, step=500)
    # plot_initial_state(system, config)
    test = get_2d_111_potential(system, config.shape, config.resolution)
    plot_2d_111_potential(test)
    plot_2d_111_state_against_t(system, config, n=100, step=50)
    # print(max(test["data"]))
    # print(min(test["data"]))
