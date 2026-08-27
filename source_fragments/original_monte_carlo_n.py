import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

def theoretical_nd_sphere_volume(dim):
    numerator = np.pi ** (dim / 2)
    denominator = gamma(dim / 2 + 1)
    vol = numerator / denominator
    return vol

def mc_circle_area(N_samples, plot_points=False):
    x = np.random.uniform(low=-1, high=1, size=N_samples)
    y = np.random.uniform(low=-1, high=1, size=N_samples)
    in_circle = (x ** 2 + y ** 2) <= 1
    N_in = np.sum(in_circle)
    mc_area = 4 * (N_in / N_samples)
    true_area = np.pi
    abs_error = np.abs(mc_area - true_area)
    rel_error = (abs_error / true_area) * 100
    if plot_points:
        plt.figure(figsize=(6, 6))
        plt.scatter(x[in_circle], y[in_circle], c='blue', s=1, alpha=0.6, label='Points inside the circle')
        plt.scatter(x[~in_circle], y[~in_circle], c='red', s=1, alpha=0.6, label='Points outside the circle')
        theta = np.linspace(0, 2 * np.pi, 1000)
        plt.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=2, label='Unit circle')
        plt.xlim(-1.1, 1.1)
        plt.ylim(-1.1, 1.1)
        plt.axis('equal')
        plt.title(f'MC method for circle area calculation (Number of samples={N_samples})')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

    return mc_area, abs_error, rel_error

def mc_nd_sphere_volume(dim, N_samples):
    points = np.random.uniform(low=-1, high=1, size=(N_samples, dim))
    sq_distances = np.sum(points ** 2, axis=1)
    N_in = np.sum(sq_distances <= 1)
    cube_vol = 2 ** dim
    mc_vol = cube_vol * (N_in / N_samples)
    true_vol = theoretical_nd_sphere_volume(dim)
    abs_error = np.abs(mc_vol - true_vol)
    rel_error = (abs_error / true_vol) * 100
    return mc_vol, abs_error, rel_error

def grid_2d_circle_area(K):
    x = np.linspace(-1 + 1 / K, 1 - 1 / K, K)
    y = np.linspace(-1 + 1 / K, 1 - 1 / K, K)
    xx, yy = np.meshgrid(x, y)
    in_circle = (xx ** 2 + yy ** 2) <= 1
    N_in = np.sum(in_circle)
    single_grid_area = (2 / K) ** 2
    grid_area = N_in * single_grid_area
    true_area = np.pi
    abs_error = np.abs(grid_area - true_area)
    rel_error = (abs_error / true_area) * 100
    return grid_area, abs_error, rel_error

def grid_3d_sphere_volume(K):
    x = np.linspace(-1 + 1 / K, 1 - 1 / K, K)
    y = np.linspace(-1 + 1 / K, 1 - 1 / K, K)
    z = np.linspace(-1 + 1 / K, 1 - 1 / K, K)
    xx, yy, zz = np.meshgrid(x, y, z)
    in_sphere = (xx ** 2 + yy ** 2 + zz ** 2) <= 1
    N_in = np.sum(in_sphere)
    single_grid_vol = (2 / K) ** 3
    grid_vol = N_in * single_grid_vol
    true_vol = theoretical_nd_sphere_volume(3)
    abs_error = np.abs(grid_vol - true_vol)
    rel_error = (abs_error / true_vol) * 100
    return grid_vol, abs_error, rel_error

if __name__ == "__main__":
    print("=" * 80)
    print("Experiment 1: Monte Carlo method for calculating the area of 2D unit circle")
    print("=" * 80)
    N_list_2d = [100, 1000, 10000, 100000, 1000000]
    for N in N_list_2d:
        mc_area, abs_err, rel_err = mc_circle_area(N, plot_points=(N == 10000))
        print(
            f"Number of samples={N:>7d} | MC estimated area={mc_area:.6f} | Absolute error={abs_err:.6f} | Relative error={rel_err:.4f}% | Theoretical value={np.pi:.6f}")

    print("\n" + "=" * 80)
    print("Experiment 2: Monte Carlo method for calculating the volume of N-dimensional unit sphere (Number of samples=1,000,000)")
    print("=" * 80)
    dim_list = [3, 4, 10]
    N_samples_nd = 1000000
    for dim in dim_list:
        mc_vol, abs_err, rel_err = mc_nd_sphere_volume(dim, N_samples_nd)
        true_vol = theoretical_nd_sphere_volume(dim)
        print(
            f"Dimension={dim:>2d} | MC estimated volume={mc_vol:.6f} | Absolute error={abs_err:.6f} | Relative error={rel_err:.4f}% | Theoretical value={true_vol:.6f}")

    print("\n" + "=" * 80)
    print("Experiment 3: Grid method vs Monte Carlo method (2D unit circle, equivalent computational effort)")
    print("=" * 80)
    K_2d = 100
    grid_area, grid_abs_err, grid_rel_err = grid_2d_circle_area(K_2d)
    mc_area, mc_abs_err, mc_rel_err = mc_circle_area(10000)
    print(f"Grid method (K={K_2d}, Total number of grids={K_2d * K_2d}) : Estimated area={grid_area:.6f} | Relative error={grid_rel_err:.4f}%")
    print(f"MC method (Number of samples=10000) : Estimated area={mc_area:.6f} | Relative error={mc_rel_err:.4f}%")

    print("\n" + "=" * 80)
    print("Experiment 4: Grid method vs Monte Carlo method (3D unit sphere, equivalent computational effort)")
    print("=" * 80)
    K_3d = 50
    grid_vol, grid_abs_err, grid_rel_err = grid_3d_sphere_volume(K_3d)
    mc_vol, mc_abs_err, mc_rel_err = mc_nd_sphere_volume(3, 125000)
    true_vol_3d = theoretical_nd_sphere_volume(3)
    print(f"Grid method (K={K_3d}, Total number of grids={K_3d ** 3}) : Estimated volume={grid_vol:.6f} | Relative error={grid_rel_err:.4f}%")
    print(f"MC method (Number of samples=125000) : Estimated volume={mc_vol:.6f} | Relative error={mc_rel_err:.4f}%")

    print("\n" + "=" * 80)
    print("Experiment 5: Visualization of N-dimensional unit sphere volume variation with dimension")
    print("=" * 80)
    dims_plot = np.arange(1, 21)
    vols_plot = [theoretical_nd_sphere_volume(d) for d in dims_plot]

    plt.figure(figsize=(10, 6))
    plt.plot(dims_plot, vols_plot, 'o-', color='darkblue', linewidth=2, markersize=6)
    plt.axvline(x=5, color='red', linestyle='--', label='Maximum volume (5 dimensions)')
    plt.xlabel('Dimension N', fontsize=12)
    plt.ylabel('Volume of N-dimensional unit sphere', fontsize=12)
    plt.title('Volume variation of N-dimensional unit sphere with dimension', fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.xticks(dims_plot)
    plt.show()

    print("\n" + "=" * 80)
    print("Experiment 6: Visualization of MC method error variation with number of samples (Verifying 1/√N scaling law)")
    print("=" * 80)
    N_list_err = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]
    mc_errors = []
    for N in N_list_err:
        err_list = [mc_circle_area(N)[1] for _ in range(5)]
        avg_err = np.mean(err_list)
        mc_errors.append(avg_err)

    plt.figure(figsize=(10, 6))
    plt.loglog(N_list_err, mc_errors, 'o-', color='darkblue', label='Actual error of MC method', markersize=6)
    ref_err = 0.5 / np.sqrt(N_list_err)
    plt.loglog(N_list_err, ref_err, 'r--', label=r'Theoretical trend of 1/√N', linewidth=2)
    plt.xlabel('Number of samples N (log scale)', fontsize=12)
    plt.ylabel('Absolute error (log scale)', fontsize=12)
    plt.title('MC method for circle area calculation: Error variation with number of samples', fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()