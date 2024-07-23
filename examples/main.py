from reduced_state_caldeira_leggett.plot import (
    plot_isotropic_noise_kernel,
)
from reduced_state_caldeira_leggett.system import (
    HYDROGEN_NICKEL_SYSTEM,
    SimulationConfig,
)

if __name__ == "__main__":
    system = HYDROGEN_NICKEL_SYSTEM
    config = SimulationConfig(
        shape=(2,),
        resolution=(31,),
        n_bands=3,
        type="bloch",
        temperature=150,
    )

    # plot_new_noise_operators(system, config, n=1)
    plot_isotropic_noise_kernel(system, config)
    # plot_kernel(system, config)
    # hamiltonian = _get_full_hamiltonian(system, config.shape, config.resolution)
    # a, lambda_ = get_effective_gaussian_parameters(
    #     hamiltonian["basis"][0],
    #     system.eta,
    #     config.temperature,
    #     lambda_factor=2 * np.sqrt(2),
    # )

    # basis_x = stacked_basis_as_fundamental_position_basis(hamiltonian["basis"][0])
    # correlation = get_gaussian_isotropic_noise_kernel(basis_x, a, lambda_)
    # data = correlation["data"].reshape(correlation["basis"].shape)
    # fig, _, _ = plot_data_1d(data, np.arange(data.size), scale="linear", measure="real")
    # fig.show()
    # said missing arg in pcolormesh
    # input()

    # plot_basis_states(system, config)
    # plot_state_against_t(system, config, n=1000, step=500)
    # plot_initial_state(system, config)
