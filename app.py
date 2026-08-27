from __future__ import annotations

import math
from numbers import Integral, Real

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from rw_mc_studio.advanced import (
    bootstrap_mean_ci,
    first_passage_1d,
    multi_seed_random_walk,
    repeated_scrambled_qmc,
)
from rw_mc_studio.monte_carlo import (
    circle_area_mc,
    grid_circle_area,
    grid_sphere_volume,
    nd_ball_volume_mc,
    repeated_circle_trials,
    theoretical_nd_ball_volume,
)
from rw_mc_studio.presets import DEFAULT_PRESET, dumps_preset, loads_preset
from rw_mc_studio.random_walk import (
    return_probability,
    simulate_endpoints,
    simulate_trajectory,
    step_second_moment,
    summarize_endpoints,
    theoretical_mean_radius,
    theoretical_msd,
)
from rw_mc_studio.scans import fit_power_law, mc_convergence_scan, random_walk_scan
from rw_mc_studio.semantics import (
    GEOMETRY_GLOSSARY,
    SCAN_LABELS,
    STAT_DEFINITIONS,
    STEP_MODEL_LABELS,
    SYMBOL_GLOSSARY,
    THEORY_STATUS,
    step_parameter_labels,
)


APP_VERSION = "4.0.0"
RESULT_ORDER_NOTE = "Every result follows the same reading order: visualization, key results, then complete data."


def format_number(value: object, significant: int = 6) -> str:
    """Readable display formatting without changing exported numeric precision."""
    if value is None:
        return "Not observed"
    if isinstance(value, (bool, np.bool_)):
        return "Yes" if bool(value) else "No"
    if isinstance(value, (Integral, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (Real, np.floating)):
        x = float(value)
        if math.isnan(x):
            return "NaN"
        if math.isinf(x):
            return "+inf" if x > 0 else "-inf"
        if x == 0:
            return "0"
        if abs(x) >= 1_000_000 or abs(x) < 0.0001:
            return f"{x:.{significant - 1}e}"
        return f"{x:,.{significant}g}"
    return str(value)


def display_table(df: pd.DataFrame, *, hide_index: bool = True) -> None:
    shown = df.copy()
    for column in shown.columns:
        if pd.api.types.is_numeric_dtype(shown[column]):
            shown[column] = shown[column].map(format_number)
    st.dataframe(shown, use_container_width=True, hide_index=hide_index)


def download_csv(df: pd.DataFrame, filename: str, label: str = "Download complete data (CSV)") -> None:
    st.download_button(
        label,
        df.to_csv(index=False),
        filename,
        "text/csv",
        use_container_width=True,
    )


def result_section(number: int, title: str) -> None:
    st.markdown(f"### {number}. {title}")


def show_figure(fig: plt.Figure) -> None:
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def scalar_frame(data: dict, excluded: tuple[str, ...] = ()) -> pd.DataFrame:
    row = {}
    for key, value in data.items():
        if key in excluded or isinstance(value, (np.ndarray, list, tuple, dict)):
            continue
        row[key] = value
    return pd.DataFrame([row])


def parse_number_list(text: str, integer: bool = False) -> list[float] | list[int]:
    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("Enter at least one comma-separated value.")
    parsed = [float(item) for item in values]
    if not all(math.isfinite(value) for value in parsed):
        raise ValueError("Scan values must be finite.")
    if integer:
        if any(value != int(value) for value in parsed):
            raise ValueError("All values in this field must be integers.")
        return [int(value) for value in parsed]
    return parsed


def progress_callback(progress_bar):
    def update(fraction: float, message: str) -> None:
        progress_bar.progress(min(max(int(fraction * 100), 0), 100), text=message)

    return update


st.set_page_config(
    page_title="Random Walk and Monte Carlo Studio",
    page_icon="📊",
    layout="wide",
)
st.title("Random Walk and Monte Carlo Simulation Studio")
st.caption(
    f"Version {APP_VERSION} | Reproducible stochastic simulation, uncertainty analysis, "
    "parameter scans, and memory-aware computation."
)
st.info(RESULT_ORDER_NOTE)

with st.sidebar:
    st.header("Global settings")
    seed = st.number_input(
        "Random seed",
        min_value=0,
        max_value=2_147_483_647,
        value=12_345,
        step=1,
    )
    st.caption("The same seed and parameters reproduce the same stochastic result.")
    st.divider()
    st.markdown("**Uncertainty terminology**")
    st.caption(
        "SD measures spread among observations. SEM measures uncertainty in an estimated mean. "
        "Seed-to-seed SD measures variation among complete independent runs."
    )
    with st.expander("Symbol glossary"):
        for symbol, meaning in SYMBOL_GLOSSARY.items():
            st.markdown(f"- **{symbol}** - {meaning}")

(
    overview_tab,
    walk_tab,
    scan_tab,
    mc_tab,
    ball_tab,
    return_tab,
    grid_tab,
    audit_tab,
    preset_tab,
) = st.tabs(
    [
        "Study Map",
        "Random Walk Ensemble",
        "Parameter Scan",
        "Repeated Monte Carlo",
        "High-Dimensional Ball",
        "Return and First Passage",
        "Grid vs Monte Carlo",
        "Reproducibility Audit",
        "Presets and Glossary",
    ]
)


with overview_tab:
    st.subheader("Reconstructed study map")
    audit_df = pd.DataFrame(
        {
            "dimension": [1, 2, 3, 4],
            "correct_asymptotic_coefficient": [0.7979, 0.8862, 0.9213, 0.9400],
            "legacy_source_coefficient": [1.1284, 0.7979, 0.9213, 1.0000],
        }
    )

    result_section(1, "Visualization")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        audit_df["dimension"],
        audit_df["correct_asymptotic_coefficient"],
        "o-",
        label="Correct CLT/chi coefficient",
    )
    ax.plot(
        audit_df["dimension"],
        audit_df["legacy_source_coefficient"],
        "s--",
        label="Legacy source coefficient",
    )
    ax.set_xlabel("Spatial dimension d")
    ax.set_ylabel("Coefficient in E[R] approximately C_d sqrt(N)")
    ax.set_xticks(audit_df["dimension"])
    ax.grid(alpha=0.25)
    ax.legend()
    show_figure(fig)

    result_section(2, "Key study points")
    c1, c2, c3 = st.columns(3)
    c1.metric("Source-derived experiment groups", "2")
    c2.metric("Core research modules", "8")
    c3.metric("Uncertainty layers", "5")
    st.markdown(
        """
- **Random walks:** fixed or random step lengths, endpoint radius, radial spread, MSD,
  dimension comparisons, finite-horizon return, and first passage.
- **Monte Carlo integration:** unit-disk area, unit d-ball volume, midpoint-grid comparison,
  convergence scaling, and repeated scrambled Sobol QMC.
- **Semantic corrections:** MSD uses E[S^2]; mean-radius theory is labeled asymptotic;
  return probability is finite-horizon; zero-hit high-dimensional estimates retain exact-binomial intervals.
"""
    )
    st.warning(
        "The supplied report states a 2D fixed-step mean radius near 17.7 at N=1,000. "
        "Direct reproduction of the supplied rule gives a value near 28.0. "
        "The studio preserves this as a documented discrepancy."
    )

    result_section(3, "Complete audit data")
    display_table(audit_df)
    download_csv(audit_df, "study_map_coefficient_audit.csv")


with walk_tab:
    st.subheader("Ensemble endpoint statistics")
    c1, c2, c3, c4 = st.columns(4)
    dimension = c1.number_input("Dimension d", 1, 200, 2, key="rw_dimension")
    steps = c2.number_input("Steps per walker N", 1, 2_000_000, 1_000, key="rw_steps")
    walkers = c3.number_input("Independent walkers M", 10, 500_000, 10_000, key="rw_walkers")
    model = c4.selectbox(
        "Step-length model",
        list(STEP_MODEL_LABELS),
        format_func=lambda item: STEP_MODEL_LABELS[item],
        key="rw_model",
    )
    first_label, second_label = step_parameter_labels(model)
    first_default = 0.5 if model == "uniform" else 1.0
    parameter_1 = st.number_input(first_label, value=float(first_default), key=f"rw_p1_{model}")
    parameter_2 = None
    if second_label:
        second_default = 1.5 if model == "uniform" else 0.2
        parameter_2 = st.number_input(second_label, value=float(second_default), key=f"rw_p2_{model}")

    try:
        second_moment = step_second_moment(model, parameter_1, parameter_2)
        configuration_error = None
        st.caption(f"Selected model second moment E[S^2] = {format_number(second_moment)}.")
    except ValueError as exc:
        configuration_error = str(exc)
        st.error(configuration_error)

    endpoint_mb = int(dimension) * int(walkers) * 8 / 1_000_000
    too_large = endpoint_mb > 80
    st.caption(f"Persistent endpoint array estimate: {endpoint_mb:,.1f} MB.")
    if too_large:
        st.warning("The d x M endpoint array exceeds the 80 MB safety budget. Reduce d or M.")

    if st.button(
        "Run ensemble simulation",
        type="primary",
        disabled=bool(configuration_error or too_large),
    ):
        bar = st.progress(0, text="Starting ensemble simulation...")
        try:
            endpoints, metadata = simulate_endpoints(
                int(dimension),
                int(steps),
                int(walkers),
                model,
                parameter_1,
                parameter_2,
                int(seed),
                progress=progress_callback(bar),
                return_metadata=True,
            )
            summary = summarize_endpoints(endpoints)
            theory_mean = theoretical_mean_radius(
                int(dimension), int(steps), model, parameter_1, parameter_2
            )
            theory_msd = theoretical_msd(int(steps), model, parameter_1, parameter_2)
            radii = np.linalg.norm(endpoints, axis=1)
            complete = summary | {
                "dimension": int(dimension),
                "n_steps": int(steps),
                "step_model": model,
                "step_second_moment": second_moment,
                "mean_radius_theory_asymptotic": theory_mean,
                "mean_radius_difference": summary["mean_radius"] - theory_mean,
                "msd_theory_exact": theory_msd,
                "msd_difference": summary["msd"] - theory_msd,
                "engine": metadata["engine"],
                "seed": int(seed),
            }
            st.session_state["ensemble_result"] = {
                "radii": radii,
                "summary": summary,
                "theory_mean": theory_mean,
                "theory_msd": theory_msd,
                "complete": complete,
            }
        except Exception as exc:
            st.error(str(exc))
        finally:
            bar.empty()

    ensemble_result = st.session_state.get("ensemble_result")
    if ensemble_result:
        result_section(1, "Visualization")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(ensemble_result["radii"], bins=50, density=True, alpha=0.75)
        ax.axvline(
            ensemble_result["summary"]["mean_radius"],
            linestyle="--",
            label="Simulated ensemble mean",
        )
        ax.axvline(
            ensemble_result["theory_mean"],
            linestyle=":",
            label="Asymptotic theory",
        )
        ax.set_xlabel("Endpoint radius R")
        ax.set_ylabel("Probability density")
        ax.grid(alpha=0.2)
        ax.legend()
        show_figure(fig)

        result_section(2, "Key results")
        summary = ensemble_result["summary"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Mean radius",
            format_number(summary["mean_radius"]),
            format_number(summary["mean_radius"] - ensemble_result["theory_mean"]),
        )
        c2.metric("Radial spread SD", format_number(summary["std_radius"]))
        c3.metric(
            "Mean-squared displacement",
            format_number(summary["msd"]),
            format_number(summary["msd"] - ensemble_result["theory_msd"]),
        )
        c4.metric(
            "95% CI for mean radius",
            f"[{format_number(summary['mean_radius_ci_low'])}, "
            f"{format_number(summary['mean_radius_ci_high'])}]",
        )
        st.caption("Mean-radius theory is asymptotic; the MSD identity is exact for the stated model.")

        result_section(3, "Complete data")
        ensemble_df = pd.DataFrame([ensemble_result["complete"]])
        display_table(ensemble_df)
        download_csv(ensemble_df, "random_walk_ensemble_complete.csv")

    with st.expander("Statistic definitions"):
        for name, definition in STAT_DEFINITIONS.items():
            st.markdown(f"- **{name}** - {definition}")

    with st.expander("Single trajectory"):
        trajectory_steps = st.slider("Trajectory steps", 10, 10_000, 500)
        if st.button("Generate trajectory"):
            trajectory = simulate_trajectory(
                int(dimension),
                int(trajectory_steps),
                model,
                parameter_1,
                parameter_2,
                int(seed),
            )
            st.session_state["trajectory_result"] = trajectory
        trajectory_result = st.session_state.get("trajectory_result")
        if trajectory_result is not None:
            result_section(1, "Visualization")
            fig, ax = plt.subplots(figsize=(7, 5))
            if int(dimension) == 1:
                ax.plot(np.arange(len(trajectory_result)), trajectory_result[:, 0])
                ax.set_xlabel("Step")
                ax.set_ylabel("x")
            else:
                ax.plot(trajectory_result[:, 0], trajectory_result[:, 1], linewidth=1)
                ax.scatter([0], [0], label="Origin")
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.axis("equal")
                ax.legend()
            ax.grid(alpha=0.2)
            show_figure(fig)
            st.caption("For d > 2, the chart is the x-y projection of the full trajectory.")
            result_section(2, "Key results")
            final_radius = float(np.linalg.norm(trajectory_result[-1]))
            c1, c2 = st.columns(2)
            c1.metric("Final radius", format_number(final_radius))
            c2.metric("Recorded positions", format_number(len(trajectory_result)))
            result_section(3, "Complete data")
            trajectory_df = pd.DataFrame(
                trajectory_result,
                columns=[f"x_{index + 1}" for index in range(trajectory_result.shape[1])],
            )
            trajectory_df.insert(0, "step", np.arange(len(trajectory_df)))
            display_table(trajectory_df)
            download_csv(trajectory_df, "random_walk_single_trajectory.csv")


with scan_tab:
    st.subheader("Batch parameter scan")
    base_model = st.selectbox(
        "Base step model",
        ["fixed", "uniform"],
        format_func=lambda item: STEP_MODEL_LABELS[item],
        key="scan_model",
    )
    allowed = ["n_steps", "dimension", "n_walkers"]
    allowed += ["fixed_step_size"] if base_model == "fixed" else ["uniform_lower", "uniform_upper"]
    scan_variable = st.selectbox(
        "Scan variable",
        allowed,
        format_func=lambda item: SCAN_LABELS[item],
    )
    defaults = {
        "n_steps": "10,100,1000,10000",
        "dimension": "1,2,3,4,5,8,10",
        "n_walkers": "100,300,1000,3000,10000",
        "fixed_step_size": "0.25,0.5,1,1.5,2",
        "uniform_lower": "0.1,0.25,0.5,0.75,1.0",
        "uniform_upper": "1.0,1.25,1.5,2.0,3.0",
    }
    raw_values = st.text_input("Comma-separated scan values", defaults[scan_variable])
    c1, c2, c3 = st.columns(3)
    base_dimension = c1.number_input("Base dimension d", 1, 100, 2)
    base_steps = c2.number_input("Base steps N", 1, 1_000_000, 1_000)
    base_walkers = c3.number_input("Base walkers M", 10, 200_000, 5_000)
    if base_model == "fixed":
        base_p1 = st.number_input("Base fixed step s", value=1.0)
        base_p2 = None
    else:
        base_p1 = st.number_input("Base uniform lower bound a", value=0.5)
        base_p2 = st.number_input("Base uniform upper bound b", value=1.5)

    if st.button("Run parameter scan", type="primary"):
        bar = st.progress(0, text="Starting parameter scan...")
        try:
            values = parse_number_list(raw_values)
            scan_df = random_walk_scan(
                values,
                scan_variable,
                dim=int(base_dimension),
                n_steps=int(base_steps),
                n_walkers=int(base_walkers),
                model=base_model,
                p1=base_p1,
                p2=base_p2,
                seed=int(seed),
                progress=progress_callback(bar),
            )
            fit = None
            if scan_variable == "n_steps":
                fit = fit_power_law(scan_df["n_steps"], scan_df["mean_radius"])
            st.session_state["scan_result"] = {"data": scan_df, "fit": fit, "variable": scan_variable}
        except Exception as exc:
            st.error(str(exc))
        finally:
            bar.empty()

    scan_result = st.session_state.get("scan_result")
    if scan_result:
        scan_df = scan_result["data"]
        variable = scan_result["variable"]
        result_section(1, "Visualization")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(scan_df[variable], scan_df["mean_radius"], "o-", label="Simulation mean")
        ax.plot(
            scan_df[variable],
            scan_df["theory_mean_radius_asymptotic"],
            "--",
            label="Asymptotic theory",
        )
        ax.set_xlabel(SCAN_LABELS[variable])
        ax.set_ylabel("Mean endpoint radius")
        ax.grid(alpha=0.25)
        ax.legend()
        show_figure(fig)

        result_section(2, "Key results")
        c1, c2, c3 = st.columns(3)
        c1.metric("Scan points", format_number(len(scan_df)))
        c2.metric("Minimum mean radius", format_number(scan_df["mean_radius"].min()))
        c3.metric("Maximum mean radius", format_number(scan_df["mean_radius"].max()))
        if scan_result["fit"]:
            fit = scan_result["fit"]
            st.success(
                f"Observed diffusion exponent alpha = {fit['exponent']:.4f}; "
                f"target asymptotic value = 0.5000; log-space R^2 = {fit['r2_log']:.5f}."
            )

        result_section(3, "Complete data")
        display_table(scan_df)
        download_csv(scan_df, "random_walk_parameter_scan_complete.csv")


with mc_tab:
    st.subheader("Repeated Monte Carlo unit-disk area estimator")
    st.caption("The sampled object is the closed unit disk x^2 + y^2 <= 1, whose area is pi.")
    c1, c2 = st.columns(2)
    samples_per_trial = c1.number_input("Samples per trial N", 100, 100_000_000, 10_000)
    trial_count = c2.number_input("Independent trials R", 2, 100_000, 100)

    if st.button("Run repeated Monte Carlo", type="primary"):
        try:
            output = repeated_circle_trials(int(samples_per_trial), int(trial_count), int(seed))
            st.session_state["repeated_mc_result"] = output
        except Exception as exc:
            st.error(str(exc))

    repeated_result = st.session_state.get("repeated_mc_result")
    if repeated_result:
        result_section(1, "Visualization")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(repeated_result["estimates"], bins=30, alpha=0.78)
        ax.axvline(math.pi, linestyle="--", label="True area pi")
        ax.axvline(repeated_result["mean_estimate"], linestyle=":", label="Trial mean")
        ax.set_xlabel("Estimated unit-disk area")
        ax.set_ylabel("Independent trials")
        ax.grid(alpha=0.2)
        ax.legend()
        show_figure(fig)

        result_section(2, "Key results")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean estimate", format_number(repeated_result["mean_estimate"]))
        c2.metric("Run-to-run SD", format_number(repeated_result["run_to_run_sd"]))
        c3.metric("RMSE", format_number(repeated_result["rmse"]))
        c4.metric(
            "95% CI for trial mean",
            f"[{format_number(repeated_result['mean_estimate_ci_low'])}, "
            f"{format_number(repeated_result['mean_estimate_ci_high'])}]",
        )
        st.caption(
            "The fast engine samples the exact Binomial(N, pi/4) hit-count distribution, "
            "so it preserves the estimator distribution while avoiding point allocation."
        )

        result_section(3, "Complete data")
        trial_df = pd.DataFrame(
            {
                "trial": np.arange(1, len(repeated_result["estimates"]) + 1),
                "estimate": repeated_result["estimates"],
                "error": repeated_result["estimates"] - math.pi,
                "absolute_error": np.abs(repeated_result["estimates"] - math.pi),
            }
        )
        display_table(trial_df)
        download_csv(trial_df, "repeated_monte_carlo_all_trials.csv")
        with st.expander("Complete summary record"):
            display_table(scalar_frame(repeated_result, excluded=("estimates",)))

    st.divider()
    st.subheader("Monte Carlo convergence")
    convergence_values = st.text_input(
        "Sample counts N",
        "100,300,1000,3000,10000,30000,100000",
    )
    convergence_trials = st.number_input("Trials per sample count", 2, 10_000, 100)
    if st.button("Run convergence scan"):
        bar = st.progress(0, text="Running convergence scan...")
        try:
            values = parse_number_list(convergence_values, integer=True)
            convergence_df = mc_convergence_scan(
                values,
                int(convergence_trials),
                int(seed),
                progress=progress_callback(bar),
            )
            convergence_fit = fit_power_law(convergence_df["n_samples"], convergence_df["rmse"])
            st.session_state["convergence_result"] = {
                "data": convergence_df,
                "fit": convergence_fit,
            }
        except Exception as exc:
            st.error(str(exc))
        finally:
            bar.empty()

    convergence_result = st.session_state.get("convergence_result")
    if convergence_result:
        convergence_df = convergence_result["data"]
        result_section(1, "Visualization")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.loglog(convergence_df["n_samples"], convergence_df["rmse"], "o-", label="Observed RMSE")
        reference = convergence_df["rmse"].iloc[0] * np.sqrt(
            convergence_df["n_samples"].iloc[0] / convergence_df["n_samples"]
        )
        ax.loglog(convergence_df["n_samples"], reference, "--", label="N^(-1/2) reference")
        ax.set_xlabel("Samples per trial N")
        ax.set_ylabel("RMSE")
        ax.grid(alpha=0.25, which="both")
        ax.legend()
        show_figure(fig)

        result_section(2, "Key results")
        fit = convergence_result["fit"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Observed RMSE exponent", f"{fit['exponent']:.4f}")
        c2.metric("Expected exponent", "-0.5000")
        c3.metric("Log-space R^2", f"{fit['r2_log']:.5f}")

        result_section(3, "Complete data")
        display_table(convergence_df)
        download_csv(convergence_df, "monte_carlo_convergence_complete.csv")


with ball_tab:
    st.subheader("High-dimensional unit d-ball")
    st.caption(
        "The estimator uses sum(x_i^2) <= 1 and computes d-dimensional volume, "
        "not the surface measure of the boundary sphere."
    )
    c1, c2 = st.columns(2)
    ball_dimension = c1.number_input("Dimension d", 1, 300, 5)
    ball_samples = c2.number_input("Pseudo-random MC samples", 100, 50_000_000, 1_000_000)

    if st.button("Run pseudo-random high-dimensional MC"):
        try:
            st.session_state["ball_mc_result"] = nd_ball_volume_mc(
                int(ball_dimension), int(ball_samples), int(seed)
            )
        except Exception as exc:
            st.error(str(exc))

    ball_result = st.session_state.get("ball_mc_result")
    if ball_result:
        result_section(1, "Visualization")
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
        axes[0].bar(["Estimate", "Theory"], [ball_result["estimate"], ball_result["true_value"]])
        axes[0].set_ylabel("Unit d-ball volume")
        axes[0].set_title("Estimate vs theory")
        axes[1].bar(
            ["Observed", "Expected"],
            [ball_result["hits_inside"], ball_result["expected_hits_if_model_exact"]],
        )
        axes[1].set_ylabel("Points inside the ball")
        axes[1].set_title("Rare-event diagnostic")
        for axis in axes:
            axis.grid(alpha=0.2, axis="y")
        fig.tight_layout()
        show_figure(fig)

        result_section(2, "Key results")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Volume estimate", format_number(ball_result["estimate"]))
        c2.metric("Theoretical volume", format_number(ball_result["true_value"]))
        c3.metric("Absolute error", format_number(ball_result["absolute_error"]))
        c4.metric("Hits inside", format_number(ball_result["hits_inside"]))
        if ball_result["rare_event_status"] != "OK":
            st.warning(
                "Rare-event regime: expected hits are low. Increase N or compare repeated QMC. "
                "A zero-hit estimate does not imply zero uncertainty."
            )

        result_section(3, "Complete data")
        ball_df = scalar_frame(ball_result)
        display_table(ball_df)
        download_csv(ball_df, "high_dimensional_ball_mc_complete.csv")

    st.divider()
    st.subheader("Repeated scrambled Sobol QMC")
    c1, c2 = st.columns(2)
    qmc_power = c1.slider("Samples per replicate = 2^m", 6, 20, 14)
    qmc_replicates = c2.number_input("Independent scrambles", 2, 64, 8)
    if st.button("Run repeated QMC"):
        try:
            st.session_state["qmc_result"] = repeated_scrambled_qmc(
                int(ball_dimension), int(qmc_power), int(qmc_replicates), int(seed)
            )
        except Exception as exc:
            st.error(str(exc))

    qmc_result = st.session_state.get("qmc_result")
    if qmc_result:
        result_section(1, "Visualization")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        replicate_numbers = np.arange(1, len(qmc_result["estimates"]) + 1)
        ax.plot(replicate_numbers, qmc_result["estimates"], "o", label="Scrambled replicate")
        ax.axhline(qmc_result["true_value"], linestyle="--", label="Theory")
        ax.axhline(qmc_result["mean_estimate"], linestyle=":", label="Replicate mean")
        ax.set_xlabel("Independent scramble")
        ax.set_ylabel("Volume estimate")
        ax.set_xticks(replicate_numbers)
        ax.grid(alpha=0.25)
        ax.legend()
        show_figure(fig)

        result_section(2, "Key results")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean QMC estimate", format_number(qmc_result["mean_estimate"]))
        c2.metric("Replicate SD", format_number(qmc_result["replicate_sd"]))
        c3.metric("Absolute error", format_number(qmc_result["absolute_error_of_mean"]))
        c4.metric(
            "95% CI",
            f"[{format_number(qmc_result['ci_low'])}, {format_number(qmc_result['ci_high'])}]",
        )

        result_section(3, "Complete data")
        qmc_df = pd.DataFrame(
            {
                "replicate": replicate_numbers,
                "estimate": qmc_result["estimates"],
                "error": qmc_result["estimates"] - qmc_result["true_value"],
                "absolute_error": np.abs(qmc_result["estimates"] - qmc_result["true_value"]),
            }
        )
        display_table(qmc_df)
        download_csv(qmc_df, "repeated_scrambled_qmc_all_replicates.csv")
        with st.expander("Complete summary record"):
            display_table(scalar_frame(qmc_result, excluded=("estimates",)))

    if st.button("Generate theoretical volume curve for d = 1 to 50"):
        dimensions = np.arange(1, 51)
        volumes = np.array([theoretical_nd_ball_volume(int(value)) for value in dimensions])
        st.session_state["theoretical_ball_curve"] = pd.DataFrame(
            {"dimension": dimensions, "theoretical_unit_ball_volume": volumes}
        )
    curve_df = st.session_state.get("theoretical_ball_curve")
    if curve_df is not None:
        result_section(1, "Visualization")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(curve_df["dimension"], curve_df["theoretical_unit_ball_volume"], "o-")
        ax.set_xlabel("Dimension d")
        ax.set_ylabel("Theoretical unit d-ball volume")
        ax.grid(alpha=0.25)
        show_figure(fig)
        result_section(2, "Key results")
        peak_row = curve_df.loc[curve_df["theoretical_unit_ball_volume"].idxmax()]
        c1, c2 = st.columns(2)
        c1.metric("Peak dimension in d = 1...50", format_number(peak_row["dimension"]))
        c2.metric("Peak theoretical volume", format_number(peak_row["theoretical_unit_ball_volume"]))
        result_section(3, "Complete data")
        display_table(curve_df)
        download_csv(curve_df, "theoretical_unit_ball_volume_d1_to_d50.csv")


with return_tab:
    st.subheader("Finite-horizon return to origin")
    c1, c2, c3 = st.columns(3)
    return_dimension = c1.number_input("Dimension d", 1, 20, 3, key="return_dimension")
    return_horizon = c2.number_input("Horizon H", 10, 1_000_000, 5_000)
    return_trials = c3.number_input("Independent trajectories", 10, 200_000, 2_000)
    if st.button("Estimate finite-horizon return probability", type="primary"):
        bar = st.progress(0, text="Estimating return probability...")
        try:
            st.session_state["return_result"] = return_probability(
                int(return_dimension),
                int(return_horizon),
                int(return_trials),
                int(seed),
                progress=progress_callback(bar),
            )
        except Exception as exc:
            st.error(str(exc))
        finally:
            bar.empty()

    return_result = st.session_state.get("return_result")
    if return_result:
        result_section(1, "Visualization")
        probability = return_result["return_probability_by_horizon"]
        low_error = probability - return_result["probability_ci_low"]
        high_error = return_result["probability_ci_high"] - probability
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.bar(["Returned by H", "Not returned by H"], [probability, 1 - probability])
        ax.errorbar(
            [0],
            [probability],
            yerr=np.array([[low_error], [high_error]]),
            fmt="none",
            capsize=6,
            color="black",
        )
        ax.set_ylim(0, 1)
        ax.set_ylabel("Estimated probability")
        ax.grid(alpha=0.2, axis="y")
        show_figure(fig)

        result_section(2, "Key results")
        c1, c2, c3 = st.columns(3)
        c1.metric("Return probability by H", format_number(probability))
        c2.metric("Returned trajectories", format_number(return_result["returned_trials"]))
        c3.metric(
            "95% exact-binomial CI",
            f"[{format_number(return_result['probability_ci_low'])}, "
            f"{format_number(return_result['probability_ci_high'])}]",
        )
        st.info(return_result["semantic_note"])

        result_section(3, "Complete data")
        return_df = scalar_frame(return_result)
        display_table(return_df)
        download_csv(return_df, "finite_horizon_return_complete.csv")

    st.divider()
    st.subheader("One-dimensional first-passage extension")
    c1, c2, c3 = st.columns(3)
    target = c1.number_input("Boundary magnitude L", 1, 1_000, 20)
    passage_trials = c2.number_input("Trials", 100, 100_000, 5_000, key="passage_trials")
    passage_horizon = c3.number_input("Maximum steps", 10, 1_000_000, 10_000, key="passage_horizon")
    if st.button("Run first-passage experiment"):
        try:
            st.session_state["first_passage_result"] = first_passage_1d(
                int(target), int(passage_trials), int(passage_horizon), int(seed)
            )
        except Exception as exc:
            st.error(str(exc))

    passage_result = st.session_state.get("first_passage_result")
    if passage_result:
        result_section(1, "Visualization")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        if len(passage_result["hit_times"]):
            ax.hist(passage_result["hit_times"], bins=50, alpha=0.78)
            ax.set_xlabel("First-passage step")
            ax.set_ylabel("Observed hits")
        else:
            ax.text(0.5, 0.5, "No first passage observed within the horizon.", ha="center", va="center")
            ax.set_axis_off()
        ax.grid(alpha=0.2)
        show_figure(fig)

        result_section(2, "Key results")
        c1, c2, c3 = st.columns(3)
        c1.metric("Observed hit fraction", format_number(passage_result["hit_fraction"]))
        c2.metric(
            "Mean first-passage step among hits",
            format_number(passage_result["mean_first_passage_among_hits"]),
        )
        c3.metric("Censored fraction", format_number(passage_result["censored_fraction"]))

        result_section(3, "Complete data")
        passage_df = pd.DataFrame(
            {
                "observed_hit": np.arange(1, len(passage_result["hit_times"]) + 1),
                "first_passage_step": passage_result["hit_times"],
            }
        )
        display_table(passage_df)
        download_csv(passage_df, "first_passage_observed_hit_times.csv")
        with st.expander("Complete summary record"):
            display_table(scalar_frame(passage_result, excluded=("hit_times",)))


with grid_tab:
    st.subheader("Midpoint grid vs Monte Carlo")
    c1, c2 = st.columns(2)
    grid_2d = c1.number_input("2D midpoint-grid subdivisions K", 10, 5_000, 100)
    grid_3d = c2.number_input("3D midpoint-grid subdivisions K", 5, 300, 50)
    if st.button("Compare methods at equal evaluation counts", type="primary"):
        try:
            grid_disk = grid_circle_area(int(grid_2d))
            mc_disk = circle_area_mc(int(grid_2d * grid_2d), int(seed))
            grid_sphere = grid_sphere_volume(int(grid_3d))
            mc_sphere = nd_ball_volume_mc(3, int(grid_3d**3), int(seed))
            grid_df = pd.DataFrame(
                [
                    ["2D midpoint grid", grid_disk["points"], grid_disk["estimate"], grid_disk["true_value"], grid_disk["absolute_error"]],
                    ["2D Monte Carlo", int(grid_2d**2), mc_disk["estimate"], mc_disk["true_value"], mc_disk["absolute_error"]],
                    ["3D midpoint grid", grid_sphere["points"], grid_sphere["estimate"], grid_sphere["true_value"], grid_sphere["absolute_error"]],
                    ["3D Monte Carlo", int(grid_3d**3), mc_sphere["estimate"], mc_sphere["true_value"], mc_sphere["absolute_error"]],
                ],
                columns=["method", "evaluation_points", "estimate", "true_value", "absolute_error"],
            )
            st.session_state["grid_result"] = grid_df
        except Exception as exc:
            st.error(str(exc))

    grid_result = st.session_state.get("grid_result")
    if grid_result is not None:
        result_section(1, "Visualization")
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
        axes[0].bar(grid_result["method"], grid_result["estimate"])
        axes[0].scatter(
            np.arange(len(grid_result)),
            grid_result["true_value"],
            marker="_",
            s=500,
            label="Theory",
        )
        axes[0].set_ylabel("Area or volume estimate")
        axes[0].tick_params(axis="x", rotation=25)
        axes[0].legend()
        axes[1].bar(grid_result["method"], grid_result["absolute_error"])
        axes[1].set_ylabel("Absolute error")
        axes[1].tick_params(axis="x", rotation=25)
        for axis in axes:
            axis.grid(alpha=0.2, axis="y")
        fig.tight_layout()
        show_figure(fig)

        result_section(2, "Key results")
        best_row = grid_result.loc[grid_result["absolute_error"].idxmin()]
        c1, c2, c3 = st.columns(3)
        c1.metric("Lowest-error method", str(best_row["method"]))
        c2.metric("Lowest absolute error", format_number(best_row["absolute_error"]))
        c3.metric("Largest evaluation count", format_number(grid_result["evaluation_points"].max()))
        st.caption("Grid cost grows as K^d; the method remains valid but becomes impractical as dimension rises.")

        result_section(3, "Complete data")
        display_table(grid_result)
        download_csv(grid_result, "grid_vs_monte_carlo_complete.csv")


with audit_tab:
    st.subheader("Multi-seed reproducibility audit")
    c1, c2, c3, c4 = st.columns(4)
    audit_dimension = c1.number_input("Dimension d", 1, 50, 2, key="audit_dimension")
    audit_steps = c2.number_input("Steps N", 1, 1_000_000, 1_000, key="audit_steps")
    audit_walkers = c3.number_input("Walkers per seed M", 100, 100_000, 5_000, key="audit_walkers")
    audit_count = c4.number_input("Independent seeds", 2, 100, 10, key="audit_count")
    audit_model = st.selectbox(
        "Step model",
        ["fixed", "uniform"],
        format_func=lambda item: STEP_MODEL_LABELS[item],
        key="audit_model",
    )
    if audit_model == "fixed":
        audit_p1 = st.number_input("Fixed step s", value=1.0, key="audit_p1")
        audit_p2 = None
    else:
        audit_p1 = st.number_input("Uniform lower bound a", value=0.5, key="audit_uniform_a")
        audit_p2 = st.number_input("Uniform upper bound b", value=1.5, key="audit_uniform_b")

    if st.button("Run reproducibility audit", type="primary"):
        seeds = np.arange(int(seed), int(seed) + int(audit_count))
        bar = st.progress(0, text="Running independent seeds...")
        try:
            audit_df, summary = multi_seed_random_walk(
                int(audit_dimension),
                int(audit_steps),
                int(audit_walkers),
                seeds,
                audit_model,
                audit_p1,
                audit_p2,
                progress=progress_callback(bar),
            )
            bootstrap = bootstrap_mean_ci(audit_df["mean_radius"], n_boot=2_000, seed=int(seed))
            st.session_state["audit_result"] = {
                "data": audit_df,
                "summary": summary,
                "bootstrap": bootstrap,
            }
        except Exception as exc:
            st.error(str(exc))
        finally:
            bar.empty()

    audit_result = st.session_state.get("audit_result")
    if audit_result:
        audit_df = audit_result["data"]
        result_section(1, "Visualization")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.errorbar(
            audit_df["seed"],
            audit_df["mean_radius"],
            yerr=audit_df["sem_radius"],
            fmt="o",
            capsize=3,
        )
        ax.axhline(audit_result["summary"]["mean_radius_mean"], linestyle="--", label="Mean of run means")
        ax.set_xlabel("Random seed")
        ax.set_ylabel("Run mean radius with within-run SEM")
        ax.grid(alpha=0.25)
        ax.legend()
        show_figure(fig)

        result_section(2, "Key results")
        summary = audit_result["summary"]
        bootstrap = audit_result["bootstrap"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean of run means", format_number(summary["mean_radius_mean"]))
        c2.metric("Seed-to-seed SD", format_number(summary["mean_radius_sd_across_seeds"]))
        c3.metric(
            "Bootstrap 95% CI",
            f"[{format_number(bootstrap['ci_low'])}, {format_number(bootstrap['ci_high'])}]",
        )
        c4.metric("Mean runtime per seed", format_number(summary["runtime_seconds_mean"]) + " s")

        result_section(3, "Complete data")
        display_table(audit_df)
        download_csv(audit_df, "multi_seed_reproducibility_complete.csv")
        with st.expander("Complete aggregate summary"):
            aggregate_df = pd.DataFrame([summary | bootstrap])
            display_table(aggregate_df)
            download_csv(aggregate_df, "multi_seed_reproducibility_summary.csv", "Download summary (CSV)")


with preset_tab:
    st.subheader("Preset JSON and semantic glossary")
    with st.expander("Geometry terminology"):
        for name, meaning in GEOMETRY_GLOSSARY.items():
            st.markdown(f"- **{name}** - {meaning}")
    with st.expander("Theory and result status"):
        for name, meaning in THEORY_STATUS.items():
            st.markdown(f"- **{name}** - {meaning}")

    preset_text = st.text_area("Preset JSON", dumps_preset(DEFAULT_PRESET), height=320)
    c1, c2 = st.columns(2)
    if c1.button("Validate preset", use_container_width=True):
        try:
            validated = loads_preset(preset_text)
            st.success("Preset schema is valid.")
            st.json(validated)
        except Exception as exc:
            st.error(str(exc))
    c2.download_button(
        "Download preset",
        preset_text,
        "random_walk_monte_carlo_preset_v4.json",
        "application/json",
        use_container_width=True,
    )

st.divider()
st.caption(
    "Version 4.0.0: English-only interface, persistent results, visualization-first reporting, "
    "standardized numeric display, and complete-data exports."
)
