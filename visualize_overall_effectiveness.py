#!/usr/bin/env python3
"""
Generate maps comparing KGE scores before and after FGSM perturbations
for both the LSTM and HBV models.

Data sources:
    - Germany boundary: downloaded from the `datasets/geo-countries` repository
      (countries.geojson) via `GERMANY_GEOJSON_URL` and cached at
      `data/external/germany_boundary.geojson`.
      https://github.com/datasets/geo-countries
    - Catchment/station locations: `data/gauging_stations/CAMELS_DE_gauging_stations.shp`,
      provided by the CAMELS-DE dataset.

Outputs:
    data/figures/kge_maps_combined.pdf
    data/figures/kge_delta_map.pdf
    data/figures/kge_point_ecdf.pdf
"""
from __future__ import annotations

import json
import string
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import shapefile
from matplotlib.patches import Patch
from pyproj import Transformer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = Path("data")
FIGURE_DIR = DATA_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

LSTM_METRICS_CSV = DATA_DIR / "lstm_fgsm_metrics_eps_0.20.csv"
HBV_METRICS_CSV = DATA_DIR / "hbv_fgsm_metrics_eps_0.20.csv"
# CAMELS-DE dataset shapefile with gauging station locations.
STATION_SHP = DATA_DIR / "gauging_stations" / "CAMELS_DE_gauging_stations.shp"
# Cached copy of the Germany boundary geometry (downloaded if missing).
GERMANY_GEOJSON = DATA_DIR / "external" / "germany_boundary.geojson"
GERMANY_GEOJSON.parent.mkdir(parents=True, exist_ok=True)

# Source for the Germany boundary GeoJSON (countries.geojson).
GERMANY_GEOJSON_URL = (
    "https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson"
)

# Lambert Azimuthal Equal Area (EPSG:3035) -> WGS84 (EPSG:4326)
TRANSFORMER = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)

KGE_BINS = [-np.inf, 0.0, 0.4, 0.6, 0.8, np.inf]
KGE_LABELS = [
    "<0",
    "0 – 0.4",
    "0.4 – 0.6",
    "0.6 – 0.8",
    "≥ 0.8",
]
KGE_COLORS = [
    "#67001f",  # KGE < 0 (dark red)
    "#f4a582",  # 0 – 0.4 (salmon)
    "#fee08b",  # 0.4 – 0.6 (yellow)
    "#c7e9c0",  # 0.6 – 0.8 (light green)
    "#006d2c",  # ≥ 0.8 (dark green)
]
KGE_COLOR_MAP = {label: color for label, color in zip(KGE_LABELS, KGE_COLORS)}

DELTA_BINS = [-np.inf, -0.5, -0.2, -0.05, 0.05, 0.2, 0.5, np.inf]
DELTA_LABELS = [
    "Δ < -0.5",
    "-0.5 ≤ Δ < -0.2",
    "-0.2 ≤ Δ < -0.05",
    "-0.05 ≤ Δ < 0.05",
    "0.05 ≤ Δ < 0.2",
    "0.2 ≤ Δ < 0.5",
    "Δ ≥ 0.5",
]
DELTA_COLORS = ["#67001f", "#b2182b", "#f4a582", "#f7f7f7", "#92c5de", "#4393c3", "#2166ac"]
DELTA_COLOR_MAP = {label: color for label, color in zip(DELTA_LABELS, DELTA_COLORS)}


def save_figure(fig: plt.Figure, stem: Path) -> None:
    """Save a matplotlib figure as PDF."""
    pdf_path = stem.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")


def map_categories_to_colors(series: pd.Series, mapping: dict[str, str], default: str = "#bdbdbd") -> pd.Series:
    """Map categorical values to colors with a fallback for missing entries."""
    mapped = series.map(mapping)
    return mapped.where(mapped.notna(), default)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_germany_boundary() -> dict:
    """
    Ensure the Germany boundary geojson is available locally and return its geometry.
    """
    if not GERMANY_GEOJSON.exists():
        response = requests.get(GERMANY_GEOJSON_URL, timeout=30)
        response.raise_for_status()
        payload = response.json()

        with GERMANY_GEOJSON.open("w", encoding="utf-8") as fout:
            json.dump(payload, fout)
    else:
        with GERMANY_GEOJSON.open(encoding="utf-8") as fin:
            payload = json.load(fin)

    for feature in payload["features"]:
        props = feature.get("properties", {})
        name = props.get("ADMIN") or props.get("name")
        iso3 = props.get("ISO_A3") or props.get("ISO3166-1-Alpha-3")
        if (name and name.lower() == "germany") or (iso3 and iso3.upper() == "DEU"):
            return feature["geometry"]

    raise RuntimeError("Germany boundary was not found in the downloaded dataset.")


def iter_polygons(geometry: dict) -> Iterable[Iterable[Iterable[Tuple[float, float]]]]:
    """
    Yield polygon coordinate sequences (outer + holes) from a GeoJSON geometry.
    """
    gtype = geometry.get("type")
    coords = geometry.get("coordinates", [])

    if gtype == "Polygon":
        yield coords
    elif gtype == "MultiPolygon":
        for polygon in coords:
            yield polygon
    else:
        raise ValueError(f"Unsupported geometry type {gtype!r}")


def load_station_coordinates() -> pd.DataFrame:
    """
    Read the CAMELS-DE gauging station shapefile and convert coordinates to WGS84.
    """
    reader = shapefile.Reader(str(STATION_SHP), encoding="latin1")
    shapes = reader.shapes()
    records = reader.records()

    rows = []
    for rec, shape in zip(records, shapes):
        attributes = rec.as_dict()
        x, y = shape.points[0]
        lon, lat = TRANSFORMER.transform(x, y)
        rows.append(
            {
                "catchment": attributes["gauge_id"],
                "gauge_name": attributes.get("gauge_name"),
                "lon": lon,
                "lat": lat,
            }
        )

    return pd.DataFrame(rows)


def load_metrics() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the LSTM and HBV FGSM result tables with harmonised column names.
    """
    if not LSTM_METRICS_CSV.exists():
        raise FileNotFoundError(f"LSTM metrics file not found at {LSTM_METRICS_CSV}")
    if not HBV_METRICS_CSV.exists():
        raise FileNotFoundError(f"HBV metrics file not found at {HBV_METRICS_CSV}")

    lstm = pd.read_csv(LSTM_METRICS_CSV)
    if "catchment" not in lstm.columns and "catchment_name" in lstm.columns:
        lstm = lstm.rename(columns={"catchment_name": "catchment"})
    missing_lstm_cols = sorted({"catchment", "kge_before", "kge_after"} - set(lstm.columns))
    if missing_lstm_cols:
        raise ValueError(f"LSTM metrics file is missing columns: {missing_lstm_cols}")

    hbv = pd.read_csv(HBV_METRICS_CSV)
    if "catchment" not in hbv.columns and "catchment_name" in hbv.columns:
        hbv = hbv.rename(columns={"catchment_name": "catchment"})
    if "kge_before" not in hbv.columns and "kge_routed_before" in hbv.columns:
        hbv = hbv.rename(columns={"kge_routed_before": "kge_before"})
    if "kge_after" not in hbv.columns and "kge_routed_after" in hbv.columns:
        hbv = hbv.rename(columns={"kge_routed_after": "kge_after"})
    missing_hbv_cols = sorted({"catchment", "kge_before", "kge_after"} - set(hbv.columns))
    if missing_hbv_cols:
        raise ValueError(f"HBV metrics file is missing columns: {missing_hbv_cols}")

    return lstm, hbv


def plot_germany(ax: plt.Axes, geometry: dict, **kwargs) -> None:
    """
    Draw Germany's boundary onto the provided axis.
    """
    facecolor = kwargs.pop("facecolor", "#f0f0f0")
    edgecolor = kwargs.pop("edgecolor", "#666666")
    linewidth = kwargs.pop("linewidth", 0.5)

    for polygon in iter_polygons(geometry):
        outer = polygon[0]
        xs, ys = zip(*outer)
        ax.fill(xs, ys, facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth, zorder=0)
        for hole in polygon[1:]:
            hx, hy = zip(*hole)
            ax.fill(hx, hy, color="white", zorder=0)


def configure_map_axis(ax: plt.Axes, latitudes: Iterable[float], longitudes: Iterable[float]) -> None:
    """
    Apply consistent styling to a map axis.
    """
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")

    lats = list(latitudes)
    lons = list(longitudes)
    if lats and lons:
        lat_min, lat_max = min(lats), max(lats)
        lon_min, lon_max = min(lons), max(lons)
        lat_margin = max(0.25, (lat_max - lat_min) * 0.05)
        lon_margin = max(0.25, (lon_max - lon_min) * 0.05)
        ax.set_xlim(lon_min - lon_margin, lon_max + lon_margin)
        ax.set_ylim(lat_min - lat_margin, lat_max + lat_margin)

        lat_span = (lat_max - lat_min) + 2 * lat_margin
        lon_span = (lon_max - lon_min) + 2 * lon_margin
        mean_lat = 0.5 * (lat_min + lat_max)
        cos_lat = np.cos(np.deg2rad(mean_lat))
        if lon_span > 0 and cos_lat > 0:
            ax.set_aspect(1.0 / cos_lat)
        else:
            ax.set_aspect("auto")
    else:
        ax.set_aspect("auto")

    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.4, zorder=-1)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def main() -> None:
    germany_geom = load_germany_boundary()
    stations = load_station_coordinates()
    lstm, hbv = load_metrics()

    lstm = lstm.merge(stations, on="catchment", how="left", validate="one_to_one")
    hbv = hbv.merge(stations, on="catchment", how="left", validate="one_to_one")

    missing_lstm = lstm[lstm["lon"].isna()]
    missing_hbv = hbv[hbv["lon"].isna()]
    if not missing_lstm.empty or not missing_hbv.empty:
        missing_ids = sorted(
            set(missing_lstm["catchment"].dropna().tolist())
            | set(missing_hbv["catchment"].dropna().tolist())
        )
        raise RuntimeError(
            "Some catchments could not be matched to station coordinates: "
            + ", ".join(missing_ids)
        )

    lstm["kge_delta"] = lstm["kge_after"] - lstm["kge_before"]
    hbv["kge_delta"] = hbv["kge_after"] - hbv["kge_before"]

    # classify KGEs into bins for colour mapping
    for frame in (lstm, hbv):
        for phase in ("kge_before", "kge_after"):
            frame[f"{phase}_class"] = pd.cut(
                frame[phase],
                bins=KGE_BINS,
                labels=KGE_LABELS,
                right=False,
                include_lowest=True,
            )
        frame["kge_delta_class"] = pd.cut(
            frame["kge_delta"],
            bins=DELTA_BINS,
            labels=DELTA_LABELS,
            right=False,
            include_lowest=True,
        )

    # --- Combined map panels (before, after, delta) -----------------------
    model_frames = [
        ("LSTM", lstm),
        ("HBV", hbv),
    ]
    phase_info = [
        ("kge_before", "KGE before perturbation"),
        ("kge_after", "KGE after perturbation"),
    ]

    fig_map, axes_map = plt.subplots(2, 2, figsize=(9, 9))
    label_iter_map = iter(string.ascii_lowercase)

    for row_idx, (model_name, frame) in enumerate(model_frames):
        for col_idx, (metric_key, title_text) in enumerate(phase_info):
            ax = axes_map[row_idx, col_idx]
            plot_germany(ax, germany_geom)

            colors = map_categories_to_colors(frame[f"{metric_key}_class"], KGE_COLOR_MAP)

            ax.scatter(
                frame["lon"],
                frame["lat"],
                c=colors,
                s=18,
                edgecolor="k",
                linewidth=0.25,
                zorder=2,
            )
            configure_map_axis(ax, frame["lat"], frame["lon"])
            ax.set_title(f"{model_name} – {title_text}")

            label = next(label_iter_map)
            ax.text(
                0.02,
                0.98,
                f"({label})",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=12,
                fontweight="bold",
            )

    kge_patches = [
        Patch(facecolor=KGE_COLOR_MAP[label], edgecolor="k", linewidth=0.3, label=label)
        for label in KGE_LABELS
    ]
    fig_map.subplots_adjust(left=0.05, right=0.88, top=0.95, bottom=0.06, wspace=0.04, hspace=0.3)

    fig_map.legend(
        handles=kge_patches,
        loc="center left",
        bbox_to_anchor=(0.9, 0.5),
        ncol=1,
        frameon=False,
        title="KGE range",
        fontsize=9,
        title_fontsize=10,
        columnspacing=1.4,
        handlelength=1.2,
    )

    save_figure(fig_map, FIGURE_DIR / "kge_maps_combined")

    # --- Delta map figure -------------------------------------------------
    fig_delta, axes_delta = plt.subplots(1, 2, figsize=(10, 5))
    fig_delta.subplots_adjust(left=0.06, right=0.9, top=0.92, bottom=0.12, wspace=0.08)
    label_iter_delta = iter(string.ascii_lowercase)
    for ax, (model_name, frame) in zip(axes_delta, model_frames):
        plot_germany(ax, germany_geom)
        colors = map_categories_to_colors(frame["kge_delta_class"], DELTA_COLOR_MAP)
        ax.scatter(
            frame["lon"],
            frame["lat"],
            c=colors,
            s=18,
            edgecolor="k",
            linewidth=0.25,
            zorder=2,
        )
        configure_map_axis(ax, frame["lat"], frame["lon"])
        ax.set_title(f"{model_name} ΔKGE (after − before)")
        label = next(label_iter_delta)
        ax.text(
            0.02,
            0.98,
            f"({label})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=12,
            fontweight="bold",
        )

    delta_patches = [
        Patch(facecolor=DELTA_COLOR_MAP[label], edgecolor="k", linewidth=0.3, label=label)
        for label in DELTA_LABELS
    ]
    fig_delta.legend(
        handles=delta_patches,
        loc="center left",
        bbox_to_anchor=(0.92, 0.5),
        ncol=1,
        frameon=False,
        title="ΔKGE range",
        fontsize=9,
        title_fontsize=10,
        columnspacing=1.4,
        handlelength=1.2,
    )
    save_figure(fig_delta, FIGURE_DIR / "kge_delta_map")

    # --- Combined dot + ECDF figure --------------------------------------
    fig_combo, (ax_dot, ax_ecdf) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"wspace": 0.18})
    label_iter_combo = iter(string.ascii_lowercase)

    rng = np.random.default_rng(seed=42)
    jitter_scale = 0.04
    dot_specs = [
        ("LSTM", lstm, 0, "#1f78b4", "#a6cee3"),
        ("HBV", hbv, 2, "#33a02c", "#b2df8a"),
    ]

    for model_name, frame, base_x, before_color, after_color in dot_specs:
        ordered = frame.sort_values("kge_before").reset_index(drop=True)
        before_x = base_x + rng.normal(0, jitter_scale, len(ordered))
        after_x = base_x + 1 + rng.normal(0, jitter_scale, len(ordered))

        for xb, xa, yb, ya in zip(before_x, after_x, ordered["kge_before"], ordered["kge_after"]):
            ax_dot.plot([xb, xa], [yb, ya], color="gray", alpha=0.2, linewidth=0.5, zorder=1)

        ax_dot.scatter(
            before_x,
            ordered["kge_before"],
            color=before_color,
            s=14,
            alpha=0.85,
            edgecolor="k",
            linewidth=0.5,
            zorder=2,
        )
        ax_dot.scatter(
            after_x,
            ordered["kge_after"],
            color=after_color,
            s=14,
            alpha=0.9,
            edgecolor="k",
            linewidth=0.5,
            zorder=3,
        )

    ax_dot.set_xticks([0, 1, 2, 3])
    ax_dot.set_xticklabels(["LSTM\nbefore", "LSTM\nafter", "HBV\nbefore", "HBV\nafter"], fontsize=11)
    ax_dot.set_xlim(-0.6, 3.6)
    ax_dot.set_ylabel("KGE")
    ax_dot.set_ylim(-1.0, 1.0)
    ax_dot.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

    # legend not required; x-axis labels convey conditions

    def ecdf(values: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        arr = np.sort(values.to_numpy())
        n = arr.size
        y = np.arange(1, n + 1) / n
        return arr, y

    ecdf_specs = [
        ("LSTM", "Before perturbation", lstm["kge_before"], "#1f78b4", "-"),
        ("LSTM", "After perturbation", lstm["kge_after"], "#1f78b4", "--"),
        ("HBV", "Before perturbation", hbv["kge_before"], "#33a02c", "-"),
        ("HBV", "After perturbation", hbv["kge_after"], "#33a02c", "--"),
    ]

    for model_name, phase, series, color, style in ecdf_specs:
        x, y = ecdf(series.dropna())
        ax_ecdf.step(x, y, where="post", color=color, linestyle=style, label=f"{model_name} – {phase}")

    ax_ecdf.set_xlabel("KGE")
    ax_ecdf.set_ylabel("Cumulative probability")
    ax_ecdf.set_xlim(-1.0, 1.0)
    ax_ecdf.set_ylim(0, 1)
    ax_ecdf.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax_ecdf.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax_ecdf.legend(loc="upper left", frameon=False, title="Model & stage")

    # panel labels continue sequence
    for ax in (ax_dot, ax_ecdf):
        label = next(label_iter_combo)
        ax.text(
            0.02,
            0.98,
            f"({label})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=12,
            fontweight="bold",
        )

    save_figure(fig_combo, FIGURE_DIR / "kge_point_ecdf")

    print("Saved figures to:")
    print(f"  {(FIGURE_DIR / 'kge_maps_combined').with_suffix('.pdf')}")
    print(f"  {(FIGURE_DIR / 'kge_delta_map').with_suffix('.pdf')}")
    print(f"  {(FIGURE_DIR / 'kge_point_ecdf').with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
