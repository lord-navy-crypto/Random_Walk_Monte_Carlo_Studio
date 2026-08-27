from __future__ import annotations

STEP_MODEL_LABELS = {
    "fixed": "Fixed step length",
    "uniform": "Uniform length U(a, b)",
    "folded_normal": "Folded normal |N(mu, sigma)|",
    "exponential": "Exponential length (scale theta)",
}

SCAN_LABELS = {
    "n_steps": "Number of steps N",
    "dimension": "Spatial dimension d",
    "n_walkers": "Number of walkers M",
    "fixed_step_size": "Fixed step length s",
    "uniform_lower": "Uniform lower bound a",
    "uniform_upper": "Uniform upper bound b",
}

STAT_DEFINITIONS = {
    "mean_radius": "Ensemble average of endpoint radius R = ||X_N|| across walkers.",
    "std_radius": "Standard deviation of individual endpoint radii; this is spread, not uncertainty of the mean.",
    "msd": "Mean-squared displacement: the ensemble average of R^2.",
    "sem_radius": "Standard error of the estimated mean radius: SD(R) / sqrt(M).",
    "ci_mean_radius": "Two-sided 95% Student-t confidence interval for the ensemble mean radius.",
    "seed_sd": "Standard deviation among independent full simulation runs with different seeds.",
}

SOURCE_STATUS = {
    "assignment": "Source-derived experiment",
    "correction": "Corrected mathematical or statistical interpretation",
    "extension": "Studio extension",
}


def step_parameter_labels(model: str):
    if model == "fixed":
        return ("Step length s", None)
    if model == "uniform":
        return ("Lower bound a", "Upper bound b")
    if model == "folded_normal":
        return ("Underlying normal mean mu", "Underlying normal SD sigma")
    if model == "exponential":
        return ("Exponential scale theta (not rate lambda)", None)
    raise ValueError(model)


SYMBOL_GLOSSARY = {
    "d": "Spatial dimension.",
    "N_steps": "Number of random-walk steps per walker.",
    "M_walkers": "Number of independent walkers in one ensemble.",
    "N_samples": "Number of Monte Carlo samples in one integration trial.",
    "R_trials": "Number of independent repeated Monte Carlo trials.",
    "H": "Finite return-time horizon in steps.",
    "K": "Grid subdivisions per coordinate axis.",
    "S": "Non-negative step length before applying the independent sign.",
    "R": "Endpoint radius ||X_N||; context distinguishes it from repeated-trial count.",
}

GEOMETRY_GLOSSARY = {
    "unit_disk": "In 2D, x^2 + y^2 <= 1 is the closed unit disk. Its area is pi.",
    "unit_d_ball": "In d dimensions, sum(x_i^2) <= 1 is the unit d-ball. V_d = pi^(d/2) / Gamma(d/2 + 1).",
    "unit_sphere_boundary": "sum(x_i^2) = 1 is the boundary sphere; its surface measure is not estimated here.",
}

THEORY_STATUS = {
    "exact": "Exact identity for the stated stochastic model.",
    "asymptotic": "Large-N central-limit/chi approximation.",
    "finite_horizon": "Simulation quantity measured only through a finite step horizon.",
    "source_reported": "Value stated in the supplied report.",
    "reproduced": "Value obtained by rerunning the supplied rule.",
}
