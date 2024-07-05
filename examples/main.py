from reduced_state_caldeira_leggett.plot import plot_2d_111_potential
from reduced_state_caldeira_leggett.system import (
    HYDROGEN_NICKEL_SYSTEM,
    SODIUM_COPPER_SYSTEM,
    SimulationConfig,
    get_2d_111_potential,
)

"""# consider Li on Cu 111 plane
# atom site grid
full_grid = StackedBasis(
    FundamentalPositionBasis(np.array([1.0, 0]), 30),
    FundamentalPositionBasis(np.array([np.sqrt(3) / 2, 1 / 2]), 30),
)
sites = BasisUtil(full_grid).x_points_stacked
print(sites.shape)

# position coordinates
vector_x = np.array([5.0, 0])
vector_y = np.array([0, 5.0])
basis_x = FundamentalPositionBasis(vector_x, 50)
basis_y = FundamentalPositionBasis(vector_y, 50)
full_basis = StackedBasis(basis_x, basis_y)
util = BasisUtil(full_basis)
x_points = util.x_points_stacked


def LiCu_potential(coord, basis: StackedBasis[Any, Any]):
    # coord: position of Cu atoms(sites)
    # basis: coord system formed to represent R, the free position variable
    util = BasisUtil(full_basis)
    x_points = util.x_points_stacked

    E_R = np.zeros((basis.n,), dtype=np.float64)
    a = 49600
    b = -148
    alpha = -3.32
    beta = -0.2
    rho_s = 1.91  # A
    for point in coord.T:
        R_r = [
            np.sqrt((np.linalg.norm(point - r_i)) ** 2 + 0.1**2) for r_i in x_points.T
        ]
        # let it be moving in 2d? so delta_z = 0?
        E_i = [
            (a * np.exp(alpha * distance) + b * np.exp(beta * distance))
            * ((rho_s / distance) ** 2)
            for distance in R_r
        ]
        E_R += np.array(E_i)
    return {"basis": basis, "data": E_R}


# coord = np.array([[0, 0], [1, 0], [np.sqrt(3) / 2, 1 / 2]])
test = LiCu_potential(sites, full_basis)
fig, ax, mesh = plot_potential_2d_x(test)
print("max value", np.max(test["data"]))
print("min", np.min(test["data"]))
mesh.set_clim(0, 1322410150)
fig.show()
input()
"""


system = SODIUM_COPPER_SYSTEM
test = get_2d_111_potential(system)
plot_2d_111_potential(test)

if __name__ == "__main__":
    system = HYDROGEN_NICKEL_SYSTEM
    config = SimulationConfig(shape=(4,), resolution=(31,), n_bands=4, type="bloch")

    # plot_basis_states(system, config)
    # plot_state_against_t(system, config, n=1000, step=500)
    # plot_kernel(system, config, temperature=0.0001)
    # plot_lindblad_operator(system, config)
    # plot_thermal_occupation(system, config)
    # plot_system_eigenstates(system, config)
    # plot_stochastic_occupation(system, config, n=1000, step=500)
    # plot_initial_state(system, config)
    # plot_noise_operator(system, config, temperature=150.0)
    # plot_3d_potential(system, config)
