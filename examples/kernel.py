import numpy as np

from reduced_state_caldeira_leggett.plot import (
    plot_isotropic_kernel_percentage_error,
    plot_kernel_fit_runtime,
    plot_noise_kernel,
)
from reduced_state_caldeira_leggett.system import (
    HYDROGEN_NICKEL_SYSTEM,
    SimulationConfig,
)

if __name__ == "__kernel__":
    system = HYDROGEN_NICKEL_SYSTEM
    config = SimulationConfig(
        shape=(2,),
        resolution=(31,),
        n_bands=3,
        type="bloch",
        temperature=150,
        FitMethod="poly fit",
        n_polynomial=0,
    )
    config1 = SimulationConfig(
        shape=(3,),
        resolution=(31,),
        n_bands=3,
        type="bloch",
        temperature=150,
        FitMethod="fft",
        n_polynomial=5,
    )
    size = np.array([(3,), (4,), (5,), (6,), (7,)])
    # add 2d example here

    plot_noise_kernel(system, config)
    plot_kernel_fit_runtime(system, config, size)
    plot_isotropic_kernel_percentage_error(system, config, config1=config1)
