from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from core import BeliefState
from helpers import Config, write_to_log
from .formatting import _format_source_array, _log_location
from .physics import _signal_grid, _top_source_rmse
from .types import LocationFindingEnv, LocationObservation, LocationStrategyLibrary, _LocationTrialState


def _plot_location_trial(
    env: LocationFindingEnv,
    observations: list[LocationObservation],
    belief_state: BeliefState,
    trial_idx: int,
    final_rmse: float,
    final_top_probability: float,
    output_path: Path,
) -> None:
    try:
        import matplotlib
    except ImportError:
        _plot_location_trial_pillow(
            env,
            observations,
            belief_state,
            trial_idx,
            final_rmse,
            final_top_probability,
            output_path,
        )
        return

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    extent = (-3.0, 3.0, -3.0, 3.0)
    _x_values, _y_values, signal = _signal_grid(env, extent=extent)
    signal_vmax = float(np.nanpercentile(signal, 99.0))
    if not math.isfinite(signal_vmax) or signal_vmax <= 0.0:
        signal_vmax = float(np.nanmax(signal))

    fig, ax = plt.subplots(figsize=(7.2, 6.0), constrained_layout=True)
    image = ax.imshow(
        signal,
        origin="lower",
        extent=extent,
        cmap="Purples",
        alpha=0.72,
        vmin=0.0,
        vmax=signal_vmax,
        interpolation="bilinear",
    )
    signal_cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    signal_cbar.set_label("Signal intensity")

    true_sources = np.asarray(env.true_theta, dtype=float)
    ax.scatter(
        true_sources[:, 0],
        true_sources[:, 1],
        marker="*",
        s=180,
        c="black",
        edgecolors="white",
        linewidths=0.8,
        label="True sources",
        zorder=4,
    )

    if belief_state.hypotheses:
        top_sources = np.asarray(belief_state.hypotheses[0], dtype=float)
        ax.scatter(
            top_sources[:, 0],
            top_sources[:, 1],
            marker="x",
            s=90,
            c="#1f77b4",
            linewidths=2.0,
            label="Top belief",
            zorder=4,
        )

    if observations:
        queries = np.asarray([observation.query for observation in observations], dtype=float)
        order = np.arange(1, len(observations) + 1)
        query_scatter = ax.scatter(
            queries[:, 0],
            queries[:, 1],
            c=order,
            cmap="YlOrRd",
            s=70,
            edgecolors="#333333",
            linewidths=0.8,
            label="Queries",
            zorder=5,
        )
        order_cbar = fig.colorbar(query_scatter, ax=ax, fraction=0.046, pad=0.10)
        order_cbar.set_label("Experiment order")

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.grid(color="#d9e0e3", linewidth=0.8, alpha=0.7)
    ax.set_title(
        f"Location Finding trial {trial_idx + 1}: "
        f"RMSE={final_rmse:.3f}, top p={final_top_probability:.3f}"
    )
    ax.legend(loc="upper right", frameon=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _extent_to_pixel(
    x_value: float,
    y_value: float,
    extent: tuple[float, float, float, float],
    plot_size: int,
) -> tuple[int, int]:
    x_min, x_max, y_min, y_max = extent
    x_pixel = int(round((x_value - x_min) / (x_max - x_min) * (plot_size - 1)))
    y_pixel = int(round((y_max - y_value) / (y_max - y_min) * (plot_size - 1)))
    return x_pixel, y_pixel


def _order_color(index: int, total: int) -> tuple[int, int, int]:
    fraction = 0.0 if total <= 1 else index / (total - 1)
    start = np.asarray([230, 61, 38], dtype=float)
    end = np.asarray([255, 245, 140], dtype=float)
    color = start * (1.0 - fraction) + end * fraction
    return tuple(int(round(value)) for value in color)


def _plot_location_trial_pillow(
    env: LocationFindingEnv,
    observations: list[LocationObservation],
    belief_state: BeliefState,
    trial_idx: int,
    final_rmse: float,
    final_top_probability: float,
    output_path: Path,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    extent = (-3.0, 3.0, -3.0, 3.0)
    plot_size = 720
    right_margin = 190
    bottom_margin = 115
    top_margin = 45
    left_margin = 60
    _x_values, _y_values, signal = _signal_grid(env, extent=extent, resolution=plot_size)
    signal_vmax = float(np.nanpercentile(signal, 99.0))
    if not math.isfinite(signal_vmax) or signal_vmax <= 0.0:
        signal_vmax = float(np.nanmax(signal))
    normalized = np.clip(signal / max(signal_vmax, 1e-12), 0.0, 1.0)
    low = np.asarray([238, 244, 246], dtype=float)
    high = np.asarray([65, 54, 160], dtype=float)
    rgb = (low[None, None, :] * (1.0 - normalized[:, :, None]) + high[None, None, :] * normalized[:, :, None])
    heatmap = Image.fromarray(np.flipud(rgb.astype(np.uint8)), mode="RGB")

    canvas = Image.new("RGB", (left_margin + plot_size + right_margin, top_margin + plot_size + bottom_margin), "white")
    canvas.paste(heatmap, (left_margin, top_margin))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for tick in range(-3, 4):
        x_pixel, _ = _extent_to_pixel(tick, 0.0, extent, plot_size)
        _, y_pixel = _extent_to_pixel(0.0, tick, extent, plot_size)
        x_abs = left_margin + x_pixel
        y_abs = top_margin + y_pixel
        draw.line((x_abs, top_margin, x_abs, top_margin + plot_size), fill=(218, 226, 230), width=1)
        draw.line((left_margin, y_abs, left_margin + plot_size, y_abs), fill=(218, 226, 230), width=1)
        draw.text((x_abs - 5, top_margin + plot_size + 8), str(tick), fill=(60, 60, 60), font=font)
        draw.text((left_margin - 28, y_abs - 6), str(tick), fill=(60, 60, 60), font=font)

    draw.rectangle(
        (left_margin, top_margin, left_margin + plot_size, top_margin + plot_size),
        outline=(45, 45, 45),
        width=2,
    )

    true_sources = np.asarray(env.true_theta, dtype=float)
    for x_value, y_value in true_sources:
        x_pixel, y_pixel = _extent_to_pixel(float(x_value), float(y_value), extent, plot_size)
        x_abs = left_margin + x_pixel
        y_abs = top_margin + y_pixel
        draw.line((x_abs - 9, y_abs, x_abs + 9, y_abs), fill="black", width=3)
        draw.line((x_abs, y_abs - 9, x_abs, y_abs + 9), fill="black", width=3)
        draw.line((x_abs - 6, y_abs - 6, x_abs + 6, y_abs + 6), fill="black", width=2)
        draw.line((x_abs - 6, y_abs + 6, x_abs + 6, y_abs - 6), fill="black", width=2)

    if belief_state.hypotheses:
        for x_value, y_value in np.asarray(belief_state.hypotheses[0], dtype=float):
            x_pixel, y_pixel = _extent_to_pixel(float(x_value), float(y_value), extent, plot_size)
            x_abs = left_margin + x_pixel
            y_abs = top_margin + y_pixel
            draw.line((x_abs - 9, y_abs - 9, x_abs + 9, y_abs + 9), fill=(31, 119, 180), width=3)
            draw.line((x_abs - 9, y_abs + 9, x_abs + 9, y_abs - 9), fill=(31, 119, 180), width=3)

    for observation_idx, observation in enumerate(observations):
        x_pixel, y_pixel = _extent_to_pixel(observation.query[0], observation.query[1], extent, plot_size)
        x_abs = left_margin + x_pixel
        y_abs = top_margin + y_pixel
        color = _order_color(observation_idx, len(observations))
        radius = 8
        draw.ellipse((x_abs - radius, y_abs - radius, x_abs + radius, y_abs + radius), fill=color, outline=(40, 40, 40), width=2)

    title = f"Location Finding trial {trial_idx + 1}: RMSE={final_rmse:.3f}, top p={final_top_probability:.3f}"
    draw.text((left_margin, 15), title, fill=(20, 20, 20), font=font)
    legend_x = left_margin + plot_size + 20
    legend_y = top_margin + 20
    draw.text((legend_x, legend_y), "true sources: black", fill=(20, 20, 20), font=font)
    draw.text((legend_x, legend_y + 18), "top belief: blue x", fill=(31, 119, 180), font=font)
    draw.text((legend_x, legend_y + 36), "queries: order color", fill=(20, 20, 20), font=font)

    bar_x = left_margin + plot_size + 35
    bar_y = top_margin + 95
    bar_width = 28
    bar_height = 180
    for offset in range(bar_height):
        frac = 1.0 - offset / max(bar_height - 1, 1)
        color = tuple(int(round(value)) for value in (low * (1.0 - frac) + high * frac))
        draw.line((bar_x, bar_y + offset, bar_x + bar_width, bar_y + offset), fill=color)
    draw.rectangle((bar_x, bar_y, bar_x + bar_width, bar_y + bar_height), outline=(80, 80, 80))
    draw.text((bar_x - 5, bar_y + bar_height + 8), "Signal", fill=(20, 20, 20), font=font)

    order_y = top_margin + plot_size + 45
    order_x = left_margin + 125
    order_width = 260
    order_height = 22
    for offset in range(order_width):
        color = _order_color(offset, order_width)
        draw.line((order_x + offset, order_y, order_x + offset, order_y + order_height), fill=color)
    draw.rectangle((order_x, order_y, order_x + order_width, order_y + order_height), outline=(80, 80, 80))
    draw.text((order_x - 95, order_y + 3), "Experiment order", fill=(20, 20, 20), font=font)
    if observations:
        draw.text((order_x - 5, order_y + order_height + 6), "1", fill=(20, 20, 20), font=font)
        draw.text((order_x + order_width - 14, order_y + order_height + 6), str(len(observations)), fill=(20, 20, 20), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _write_to_log_if_configured(message: str, config: Config) -> None:
    if config.log_path is not None:
        write_to_log(message, config)


def _location_trial_rng(
    config: Config,
    trial_idx: int,
    fallback_rng: np.random.Generator,
) -> np.random.Generator:
    if config.location_seed is None:
        return fallback_rng
    seed_sequence = np.random.SeedSequence([config.location_seed, trial_idx])
    return np.random.default_rng(seed_sequence)


def _location_trial_planning_rng(
    config: Config,
    trial_idx: int,
    fallback_rng: np.random.Generator,
) -> np.random.Generator:
    if config.location_seed is None:
        return np.random.default_rng(fallback_rng.integers(0, np.iinfo(np.uint32).max))
    seed_sequence = np.random.SeedSequence([config.location_seed, trial_idx, 1])
    return np.random.default_rng(seed_sequence)


def _make_location_trial_state(
    config: Config,
    trial_idx: int,
    fallback_rng: np.random.Generator,
    method_name: str,
) -> _LocationTrialState:
    env_rng = _location_trial_rng(config, trial_idx, fallback_rng)
    planning_rng = _location_trial_planning_rng(config, trial_idx, fallback_rng)
    env = LocationFindingEnv(
        num_sources=config.location_num_sources,
        dim=config.location_dim,
        noise_sd=config.location_noise_sd,
        rng=env_rng,
    )
    return _LocationTrialState(
        trial_idx=trial_idx,
        env=env,
        observations=[],
        rng=planning_rng,
        strategy_library=LocationStrategyLibrary() if method_name in {"StrategyEIG", "StrategyEIG+root"} else None,
    )


def _plot_location_trial_state(
    state: _LocationTrialState,
    belief_state: BeliefState,
    final_rmse: float,
    final_top_probability: float,
    config: Config,
    output_dir: Path | None,
) -> None:
    if not config.location_plot_trials:
        return
    if output_dir is None:
        _log_location("plotting requested but no output directory was provided; skipping trial plot", config)
        return
    plot_path = output_dir / f"location_trial_{state.trial_idx + 1:03d}.png"
    _plot_location_trial(
        state.env,
        state.observations,
        belief_state,
        state.trial_idx,
        final_rmse,
        final_top_probability,
        plot_path,
    )
    _log_location(f"saved trial plot to {plot_path}", config)
