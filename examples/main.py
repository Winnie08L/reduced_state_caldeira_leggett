from reduced_state_caldeira_leggett.plot import (
    plot_compare_error_1d_gaussian,
    plot_gaussian_noise_kernel,
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

    plot_gaussian_noise_kernel(system, config, fit_method="poly fit", n=5)
    plot_compare_error_1d_gaussian(system, config, n=5)
