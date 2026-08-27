import numpy as np
import matplotlib.pyplot as plt

COLOR_FIX = '#1f77b4'
COLOR_RAND_UNI = '#2ca02c'
COLOR_THEO = '#ff7f0e'
MARKER = 7


def theoretical_avg(dim, N, sigma_s=1.0):
    if dim == 1:
        C_d = np.sqrt(4 / np.pi)
    elif dim == 2 :
        C_d = np.sqrt(2 / np.pi)
    elif dim == 3:
        C_d = np.sqrt(8 / (3 * np.pi))
    elif dim >= 4:
        C_d = 1.0
    return C_d * sigma_s * np.sqrt(N)


def theoretical_msd(N, sigma_s=1.0):
    return N * sigma_s ** 2


# Optimization 1: reduce the ensemble from 5,000 to 1,000 runs while retaining stable statistics.
def fixed_step(dim, N_steps, N_sim=1000):
    dists = []
    msds = []
    for _ in range(N_sim):
        coords = np.zeros(dim)
        for _ in range(N_steps):
            axis = np.random.randint(0, dim)
            coords[axis] += np.random.choice([-1, 1])
        dists.append(np.sqrt(np.sum(coords ** 2)))
        msds.append(np.sum(coords ** 2))
    return np.mean(dists), np.std(dists), np.mean(msds)


# Optimization 2: reduce the ensemble from 5,000 to 1,000 runs.
def random_step(dim, N_steps, N_sim=1000, type='uniform'):
    dists = []
    msds = []
    if type == 'uniform':
        sigma_s = np.sqrt((1.5 ** 3 - 0.5 ** 3) / (3 * (1.5 - 0.5)))
    for _ in range(N_sim):
        coords = np.zeros(dim)
        for _ in range(N_steps):
            axis = np.random.randint(0, dim)
            if type == 'uniform':
                step_size = np.random.uniform(0.5, 1.5)
                step_sign = np.random.choice([-1, 1])
                step = step_size * step_sign
            coords[axis] += step
        dists.append(np.sqrt(np.sum(coords ** 2)))
        msds.append(np.sum(coords ** 2))
    return np.mean(dists), np.std(dists), np.mean(msds), sigma_s


# Optimization 3: reduce the default horizon and ensemble size for practical runtimes.
def return_prob(dim, max_steps=5000, N_sim=200):
    cnt = 0
    if dim in [1, 2]:
        max_steps = 20000  # Use a longer horizon in one and two dimensions.
        N_sim = 400
    for _ in range(N_sim):
        coords = np.zeros(dim)
        back = False
        for _ in range(max_steps):
            axis = np.random.randint(0, dim)
            coords[axis] += np.random.choice([-1, 1])
            if np.all(coords == 0):
                back = True
                break
        if back:
            cnt += 1
    return cnt / N_sim


# Retain the core sequence needed to verify square-root scaling across four decades.
N_list = [10, 100, 1000, 10000]
avg_fix, std_fix, msd_fix = [], [], []
theo_avg, theo_msd = [], []
for N in N_list:
    a, s, m = fixed_step(2, N)
    avg_fix.append(a), std_fix.append(s), msd_fix.append(m)
    theo_avg.append(theoretical_avg(2, N)), theo_msd.append(theoretical_msd(N))

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
ax1.loglog(N_list, avg_fix, 'o-', c=COLOR_FIX, label='Simulated', ms=MARKER)
ax1.loglog(N_list, theo_avg, 's--', c=COLOR_THEO, label='Theoretical', ms=MARKER)
ax1.set_xlabel('Number of Steps (N)')
ax1.set_ylabel('Average Distance')
ax1.set_title('2D: Average Distance ~ sqrt(N)')
ax1.legend()

ax2.plot(N_list, msd_fix, 'o-', c=COLOR_FIX, label='Simulated', ms=MARKER)
ax2.plot(N_list, theo_msd, 's--', c=COLOR_THEO, label='Theoretical', ms=MARKER)
ax2.set_xlabel('Number of Steps (N)')
ax2.set_ylabel('Mean Square Distance (MSD)')
ax2.set_title('2D: MSD ~ N')
ax2.legend()

ax3.plot(N_list, std_fix, 'o-', c='darkred', ms=MARKER)
ax3.set_xlabel('Number of Steps (N)')
ax3.set_ylabel('Std of Distance (Spread)')
ax3.set_title('2D: Spread of Distance Distribution')

N = 1000
avg_uni, std_uni, msd_uni, sigma_uni = random_step(2, N, type='uniform')
ax4.bar(['Fixed', 'Random(Uniform)'],
        [avg_fix[2], avg_uni],
        color=[COLOR_FIX, COLOR_RAND_UNI])
ax4.set_ylabel('Average Distance (N=1000)')
ax4.set_title('2D: Fixed vs Random Sized Steps')
plt.tight_layout()
plt.show()

dims = [1, 2, 3, 4]
avg_dim, std_dim, msd_dim = [], [], []
for d in dims:
    a, s, m = fixed_step(d, 1000)
    avg_dim.append(a), std_dim.append(s), msd_dim.append(m)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
ax1.plot(dims, avg_dim, 'o-', c=COLOR_FIX, ms=MARKER)
ax1.set_xlabel('Dimension')
ax1.set_ylabel('Average Distance (N=1000)')
ax1.set_title('Avg Distance across Dimensions')

ax2.plot(dims, std_dim, 'o-', c='darkred', ms=MARKER)
ax2.set_xlabel('Dimension')
ax2.set_ylabel('Std of Distance (Spread)')
ax2.set_title('Spread across Dimensions')

ax3.plot(dims, msd_dim, 'o-', c=COLOR_THEO, ms=MARKER)
ax3.set_xlabel('Dimension')
ax3.set_ylabel('MSD (N=1000)')
ax3.set_title('MSD across Dimensions')
plt.tight_layout()
plt.show()

probs = [return_prob(d) for d in dims]
plt.figure(figsize=(7, 4))
plt.plot(dims, probs, 'o-', c='darkblue', ms=MARKER)
plt.axhline(1.0, c='red', ls='--', label='Theory: 1 (1/2D)')
plt.xlabel('Dimension')
plt.ylabel('Return Probability to Origin')
plt.title('Polya\'s Random Walk Constant (Bonus)')
plt.ylim(0, 1.1)
plt.legend()
plt.show()

print("=== 2D Random Walk Results (N=1000) ===")
print(f"Fixed Step | Avg: {avg_fix[2]:.3f} | Spread: {std_fix[2]:.3f} | MSD: {msd_fix[2]:.3f}")
print(f"Random(Uniform) | Avg: {avg_uni:.3f} | Spread: {std_uni:.3f} | MSD: {msd_uni:.3f}")


print("\n=== Different Dimensions (N=1000) ===")
for d, a, s, m in zip(dims, avg_dim, std_dim, msd_dim):
    print(f"Dim {d} | Avg: {a:.3f} | Spread: {s:.3f} | MSD: {m:.3f}")

print("\n=== Bonus: Return Probability ===")
for d, p in zip(dims, probs):
    print(f"Dim {d} | Probability: {p:.3f} (Theory: 1 for 1/2D, <1 for >2D)")
