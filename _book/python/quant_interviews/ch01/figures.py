"""Reproducible publication figures for Chapter 1."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from scipy.stats import beta as beta_distribution

from quant_interviews.ch01.utility import chapter_utility_examples


INTERVIEW = "#203A5F"
GRAPHITE = "#4A4F55"
MUTED = "#9AA0A6"
GRID = "#D7D9DC"


def _publication_style() -> None:
    font_path = (
        Path.home()
        / "Library"
        / "TinyTeX"
        / "texmf-dist"
        / "fonts"
        / "opentype"
        / "public"
        / "libertine"
        / "LinLibertine_R.otf"
    )
    publication_font = "DejaVu Serif"
    if font_path.is_file():
        font_manager.fontManager.addfont(font_path)
        publication_font = font_manager.FontProperties(
            fname=font_path
        ).get_name()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [publication_font],
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "axes.edgecolor": GRAPHITE,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": GRAPHITE,
            "ytick.color": GRAPHITE,
            "text.color": GRAPHITE,
            "axes.labelcolor": GRAPHITE,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _save(figure: plt.Figure, destination: Path) -> None:
    figure.savefig(
        destination,
        dpi=225,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def build_base_rate(destination: Path) -> None:
    prevalence = np.logspace(-4, -0.3, 400)
    sensitivity = 0.90
    specificity = 0.90
    posterior = (
        sensitivity * prevalence
        / (
            sensitivity * prevalence
            + (1.0 - specificity) * (1.0 - prevalence)
        )
    )
    reference_prevalence = 0.005
    reference_posterior = (
        sensitivity * reference_prevalence
        / (
            sensitivity * reference_prevalence
            + (1.0 - specificity) * (1.0 - reference_prevalence)
        )
    )

    figure, axis = plt.subplots(figsize=(7.2, 3.4))
    axis.semilogx(prevalence * 100.0, posterior * 100.0, color=GRAPHITE)
    axis.scatter(
        [reference_prevalence * 100.0],
        [reference_posterior * 100.0],
        color=INTERVIEW,
        s=36,
        zorder=3,
    )
    axis.annotate(
        "0.5% base rate -> 4.33% posterior",
        xy=(reference_prevalence * 100.0, reference_posterior * 100.0),
        xytext=(1.3, 22.0),
        color=INTERVIEW,
        arrowprops={"arrowstyle": "-", "color": INTERVIEW},
    )
    axis.set_xlabel("Event base rate (%)")
    axis.set_ylabel("Posterior after positive signal (%)")
    axis.grid(axis="both", color=GRID, linewidth=0.5, alpha=0.8)
    _save(figure, destination)


def build_sharpe_uncertainty(destination: Path) -> None:
    years = np.linspace(1.0, 10.0, 200)
    correlations = (0.0, 0.10, 0.20)

    figure, axis = plt.subplots(figsize=(7.2, 3.4))
    for correlation in correlations:
        dependence_multiplier = np.sqrt(
            (1.0 + correlation) / (1.0 - correlation)
        )
        half_width = 1.96 * dependence_multiplier / np.sqrt(years)
        color = INTERVIEW if correlation == 0.0 else MUTED
        axis.plot(
            years,
            half_width,
            color=color,
            linewidth=1.8 if correlation == 0.0 else 1.0,
        )
        axis.text(
            years[-1] + 0.1,
            half_width[-1],
            f"rho={correlation:.1f}",
            va="center",
            color=color,
        )
    reference = 1.96 / np.sqrt(3.0)
    axis.scatter([3.0], [reference], color=INTERVIEW, s=36, zorder=3)
    axis.annotate(
        "3 years, rho=0 -> half-width 1.13",
        xy=(3.0, reference),
        xytext=(4.3, 1.55),
        color=INTERVIEW,
        arrowprops={"arrowstyle": "-", "color": INTERVIEW},
    )
    axis.set_xlim(1.0, 11.2)
    axis.set_xlabel("Sample length (years)")
    axis.set_ylabel("Approximate 95% Sharpe half-width")
    axis.grid(axis="y", color=GRID, linewidth=0.5)
    _save(figure, destination)


def build_multiple_testing(destination: Path) -> None:
    tests = 5_000
    nominal_alpha = 0.05
    bonferroni_alpha = nominal_alpha / tests
    expected_false = [tests * nominal_alpha, tests * bonferroni_alpha]
    family_error = [
        1.0 - (1.0 - nominal_alpha) ** tests,
        1.0 - (1.0 - bonferroni_alpha) ** tests,
    ]
    labels = ["Nominal 5%", "Bonferroni 5% FWER"]

    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    colors = [MUTED, INTERVIEW]
    first_bars = axes[0].bar(labels, expected_false, color=colors)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Expected false rejections (log scale)")
    axes[0].bar_label(first_bars, labels=["250", "0.05"], padding=3)
    second_bars = axes[1].bar(labels, np.array(family_error) * 100.0, color=colors)
    axes[1].set_ylabel("P(at least one false rejection), %")
    axes[1].bar_label(
        second_bars,
        labels=["~100%", "4.88%"],
        padding=3,
    )
    for axis in axes:
        axis.tick_params(axis="x", rotation=12)
        axis.grid(axis="y", color=GRID, linewidth=0.5)
    figure.tight_layout(w_pad=2.0)
    _save(figure, destination)


def build_beta_posterior(destination: Path) -> None:
    probability = np.linspace(0.25, 0.80, 500)
    prior = beta_distribution.pdf(probability, 8.0, 8.0)
    posterior = beta_distribution.pdf(probability, 66.0, 50.0)
    probability_of_edge = 1.0 - beta_distribution.cdf(0.5, 66.0, 50.0)

    figure, axis = plt.subplots(figsize=(7.2, 3.4))
    axis.plot(probability, prior, color=MUTED, label="Prior Beta(8, 8)")
    axis.plot(
        probability,
        posterior,
        color=INTERVIEW,
        linewidth=2.0,
        label="Posterior Beta(66, 50)",
    )
    axis.axvline(0.5, color=GRAPHITE, linewidth=0.8, linestyle="--")
    axis.text(
        0.505,
        axis.get_ylim()[1] * 0.82,
        "Break-even probability",
        rotation=90,
        va="top",
    )
    axis.text(
        0.64,
        axis.get_ylim()[1] * 0.83,
        f"P(p > 0.5 | D) = {probability_of_edge:.1%}",
        color=INTERVIEW,
        ha="center",
    )
    axis.set_xlabel("Win probability")
    axis.set_ylabel("Density")
    axis.legend(frameon=False, loc="upper left")
    axis.grid(axis="y", color=GRID, linewidth=0.5)
    _save(figure, destination)


def build_utility_waterfall(destination: Path) -> None:
    signals = chapter_utility_examples()
    labels = [
        "Gross alpha",
        "Trading cost",
        "Variance penalty",
        "Tail + model",
        "Certainty equivalent",
    ]
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), sharey=True)
    for axis, (signal, result) in zip(axes, signals.items(), strict=True):
        components = [
            result["gross_alpha"],
            -result["trading_cost"],
            -result["variance_penalty"],
            -(result["tail_penalty"] + result["model_penalty"]),
        ]
        running = 0.0
        starts: list[float] = []
        heights: list[float] = []
        for component in components:
            starts.append(running if component >= 0.0 else running + component)
            heights.append(abs(component))
            running += component
        starts.append(0.0)
        heights.append(running)
        colors = [INTERVIEW, MUTED, MUTED, MUTED, GRAPHITE]
        axis.bar(
            np.arange(len(labels)),
            heights,
            bottom=starts,
            color=colors,
            width=0.68,
        )
        axis.axhline(0.0, color=GRAPHITE, linewidth=0.7)
        axis.set_title(signal)
        axis.set_xticks(np.arange(len(labels)), labels, rotation=25, ha="right")
        axis.text(
            len(labels) - 1,
            running + 0.003,
            f"{result['certainty_equivalent']:.2%}",
            ha="center",
            color=GRAPHITE,
        )
        axis.grid(axis="y", color=GRID, linewidth=0.5)
    axes[0].set_ylabel("Annual contribution")
    figure.tight_layout(w_pad=1.2)
    _save(figure, destination)


def build_all(output_directory: Path) -> None:
    """Build all revised v0.8 chapter figures."""

    output_directory.mkdir(parents=True, exist_ok=True)
    _publication_style()
    build_base_rate(output_directory / "fig_base_rate.png")
    build_sharpe_uncertainty(
        output_directory / "fig_sharpe_uncertainty.png"
    )
    build_multiple_testing(
        output_directory / "fig_multiple_testing.png"
    )
    build_beta_posterior(
        output_directory / "fig_beta_posterior.png"
    )
    build_utility_waterfall(
        output_directory / "fig_utility_comparison.png"
    )


if __name__ == "__main__":
    build_all(Path(__file__).resolve().parents[3] / "figures" / "ch01")
