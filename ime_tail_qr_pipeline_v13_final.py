#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IME Tail-QR Zoning — reproducible empirical pipeline
=====================================================

Purpose
-------
This script implements a reviewer-auditable computational pipeline aligned with
Sections 2–4 of the manuscript:

  * fixed commune support and adjacency graph;
  * predecision risk representation Rhat_t = (mhat_t, qhat_t, Deltahat_t);
  * finite admissible action approximation A_{t,h}(S_t);
  * explicit Keep / Split / Merge / Shift geographic edits;
  * admissible pricing perturbations around indicated zonal relativities;
  * policy-conditional distributional learning with quantile Huber loss;
  * four empirical policies: Static Zoning, Mean DQN, QR Mean, Tail-QR Zoning;
  * train / selection / test separation at trajectory level;
  * 10 independent seeds by default;
  * CUDA support through PyTorch;
  * signature figures/tables plus CSV data used to regenerate them;
  * detailed logs and a machine-readable manifest.

Important scientific limitation
--------------------------------
The open-data baseline does NOT silently identify physical hazard observations
with observed insured claims. The public-data baseline constructs a CALIBRATED ACTUARIAL LOSS ENVIRONMENT
from hazard/exposure information rather than identifying hazard records with
insured claims. The optional claims_file field is reserved for a separately
schema-audited claims extension and is not silently treated as active. Every
generated table and manifest records the baseline status explicitly.

Production usage
----------------
    python ime_tail_qr_pipeline_v13_final.py --run-all --use-cuda

Fast local validation without internet:
    python ime_tail_qr_pipeline_v13_final.py --smoke-test

Download only:
    python ime_tail_qr_pipeline_v13_final.py --download-data

Rebuild figures/tables from saved CSV outputs:
    python ime_tail_qr_pipeline_v13_final.py --rebuild-outputs

Dependencies
------------
Python >= 3.10 recommended.

    pip install numpy pandas scipy scikit-learn requests matplotlib \
        geopandas shapely pyarrow networkx torch

Optional but useful:
    pip install tqdm

The script intentionally avoids seaborn so that every published figure is
fully controlled by matplotlib/geopandas.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import io
import json
import logging
import math
import os
import random
import re
import shutil
import sys
import time
import traceback
import warnings
import zipfile
import tempfile

# Windows/Conda safeguard: NumPy/SciPy/PyTorch wheels may load distinct Intel
# OpenMP runtimes in the same process. Set this before importing NumPy/SciPy/Torch.
# The single-thread limits also avoid CPU oversubscription while CUDA is active.
if sys.platform.startswith("win"):
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from collections import Counter, defaultdict, deque, OrderedDict
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import requests
except Exception as exc:  # pragma: no cover
    raise RuntimeError("requests is required. Install with: pip install requests") from exc

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise RuntimeError("matplotlib is required. Install with: pip install matplotlib") from exc

try:
    import geopandas as gpd
    from shapely.geometry import Polygon, box
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "geopandas and shapely are required. Install with: pip install geopandas shapely"
    ) from exc

try:
    import networkx as nx
except Exception as exc:  # pragma: no cover
    raise RuntimeError("networkx is required. Install with: pip install networkx") from exc

try:
    from scipy.stats import rankdata, norm, t as student_t
except Exception as exc:  # pragma: no cover
    raise RuntimeError("scipy is required. Install with: pip install scipy") from exc

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "PyTorch is required. Install the CUDA-enabled build from https://pytorch.org/"
    ) from exc


# -----------------------------------------------------------------------------
# 0. Configuration
# -----------------------------------------------------------------------------

SCRIPT_VERSION = "2026.09.14-q1pp-v13-final-analysis"
DATA_GOUV_API = "https://www.data.gouv.fr/api/1"
CONTOURS_URL = (
    "https://etalab-datasets.geo.data.gouv.fr/contours-administratifs/"
    "latest/geojson/communes-100m.geojson.gz"
)
API_GEO_COMMUNES = (
    "https://geo.api.gouv.fr/communes?fields=nom,code,population,departement,region&format=json"
)

DEFAULT_SEEDS = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]


@dataclass
class Config:
    # Reproducibility / hardware
    seeds: List[int] = field(default_factory=lambda: DEFAULT_SEEDS.copy())
    use_cuda: bool = True
    deterministic_torch: bool = False
    torch_float32_matmul_precision: str = "high"
    num_threads: int = 4

    # Directories
    data_dir: str = "Data_IME"
    results_dir: str = "Results_IME"

    # Spatial design
    geography: str = "commune"
    metropolitan_only: bool = True
    commune_geometry_resolution_m: int = 100
    adjacency_predicate: str = "touches"
    initial_partition: str = "department"
    k_max: int = 110
    e_min: float = 2500.0
    # V4 topology safeguard: pure polygon touching can fragment administrative
    # departments because of islands or generalized geometries. The adopted
    # portfolio graph therefore augments contiguity with the minimum number of
    # deterministic within-department bridge edges needed to make each
    # department connected. All bridge edges are audited and exported.
    repair_department_connectivity: bool = True

    # Time design
    start_year: int = 2000
    end_year: int = 2025
    # Strictly disjoint baseline split: 2000--2015 / 2016--2020 / 2021--2025.
    # Each selection/test block contains one complete 5-year horizon.
    train_end_year: int = 2015
    selection_end_year: int = 2020
    test_start_year: int = 2021
    horizon: int = 5
    gamma: float = 0.97

    # Actuarial environment
    claims_file: Optional[str] = None
    loss_environment_name: str = "calibrated_actuarial_loss_environment"
    loading_rate: float = 0.08
    loss_scale: float = 185.0
    tail_df: float = 3.5
    spatial_common_factor: float = 0.35
    idiosyncratic_sigma: float = 0.45
    exposure_elasticity: float = 0.12
    controlled_response_baseline: bool = True

    # Conditional risk representation
    loss_quantiles: List[float] = field(default_factory=lambda: [0.50, 0.75, 0.90, 0.95, 0.99])
    rolling_window_years: int = 8
    dependence_neighbour_weight: float = 0.35

    # Economic payoff coefficients
    c_psi: float = 1_000_000.0
    c_c: float = 25_000.0
    c_p: float = 50_000.0
    c_z: float = 150_000.0

    # Pricing feasibility / finite action approximation
    delta_r: float = 0.25
    h: float = 0.08
    m_split_per_zone: int = 2
    # V8: only the highest-dispersion zones are considered for expensive split construction.
    m_split_target_zones: int = 10
    m_merge_total: int = 12
    m_shift_total: int = 24
    m_geographic_total: int = 20
    m_pricing_per_partition: int = 4
    max_action_set: int = 80
    boundary_patch_shift_max: int = 1

    # Candidate screening
    screen_weight_dpsi: float = 1.0
    screen_weight_uncertainty: float = 0.25
    screen_weight_boundary_contrast: float = 0.50

    # Distributional RL
    n_quantiles: int = 201
    alpha: float = 0.95
    lambda_tail: float = 0.75
    huber_kappa: float = 1.0
    hidden_dim: int = 192
    learning_rate: float = 2e-4
    weight_decay: float = 1e-6
    batch_size: int = 256
    replay_capacity: int = 120_000
    warmup_steps: int = 512
    train_steps_per_stage: int = 1_600
    target_refresh: int = 400
    gradient_clip: float = 5.0
    epsilon_start: float = 0.30
    epsilon_end: float = 0.03
    epsilon_decay_steps: int = 16_000
    evaluation_stages: int = 4

    # Monte Carlo selection / evaluation
    n_selection_trajectories: int = 1_000
    n_test_trajectories: int = 1000
    common_random_numbers: bool = True
    # V11: selection/test Monte Carlo counts are TOTAL across seeds; stress tests use a smaller total.
    n_stress_trajectories: int = 250
    # Scientific/runtime gates.
    runtime_gate_hours: float = 24.0
    enforce_runtime_gate: bool = True
    candidate_keep_per_stage: int = 1
    # Fast evaluation is exact under the baseline controlled kernel because realized losses
    # do not enter the next state; a smoke-test parity check guards this assumption.
    fast_policy_evaluation: bool = True

    # Stress tests
    stress_mean_shift: float = 0.20
    stress_tail_multiplier: float = 1.45
    stress_spatial_concentration: float = 0.30

    # V13 post-training preference / stress sensitivity.
    # These do NOT alter the baseline training protocol. They re-rank the frozen
    # distributional candidate family on the independent selection block and
    # evaluate the selected policies on the untouched test block.
    v13_lambda_grid: List[float] = field(default_factory=lambda: [0.25, 0.50, 0.75, 1.00, 1.50])
    v13_alpha_grid: List[float] = field(default_factory=lambda: [0.90, 0.95, 0.975])
    v13_tail_multiplier_grid: List[float] = field(default_factory=lambda: [1.00, 1.45, 1.80])
    v13_sensitivity_mc_total: int = 600
    v13_granularity_mc_total: int = 600

    # Output / runtime controls
    figure_dpi: int = 220
    save_svg: bool = True
    force_download: bool = False
    force_compute: bool = False
    max_download_mb: int = 900
    log_every: int = 250
    # V10 performance controls: scientific action space unchanged; implementation only.
    profile_runtime: bool = True
    cache_environment_contexts: bool = True
    cache_initial_action_set: bool = True
    # V10 action-engine controls. These change implementation, not admissibility.
    structure_cache_size: int = 8
    action_metric_cache_size: int = 128
    profile_action_engine: bool = True
    # V9: defer full Partition materialization until the action is actually executed.
    defer_partition_materialization: bool = True
    lazy_action_execution: bool = True
    # V10: SPLIT candidates are also descriptor-only until execution.
    lazy_split_execution: bool = True
    # V8: cache exact zone sufficient statistics and previous/current transition summaries.
    cache_zone_moments: bool = True
    transition_cache_size: int = 256

    # Optional auxiliary risks; kept out of climate baseline by default
    download_ssmsi: bool = False
    download_baac: bool = False
    download_meteo: bool = False
    max_meteo_departments: int = 96

    # Smoke test settings
    smoke_nx: int = 8
    smoke_ny: int = 6
    smoke_years: int = 8
    smoke_seeds: int = 2
    smoke_train_steps: int = 300


# -----------------------------------------------------------------------------
# 1. Logging, paths, reproducibility
# -----------------------------------------------------------------------------


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def finite_array(x: Any, *, fill: float = 0.0, clip: Optional[float] = None) -> np.ndarray:
    """Return a finite float64 ndarray; optionally clip extreme values symmetrically."""
    a = np.asarray(x, dtype=float)
    a = np.nan_to_num(a, nan=fill, posinf=fill, neginf=fill)
    if clip is not None:
        a = np.clip(a, -abs(float(clip)), abs(float(clip)))
    return a


def finite_float(x: Any, default: float = 0.0) -> float:
    try:
        y = float(x)
    except Exception:
        return float(default)
    return y if math.isfinite(y) else float(default)


def assert_finite_array(name: str, x: Any) -> None:
    a = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(a)):
        bad = int(np.size(a) - np.isfinite(a).sum())
        raise RuntimeError(f"Non-finite values in {name}: {bad}/{a.size}")


def setup_paths(cfg: Config) -> Dict[str, Path]:
    data = ensure_dir(Path(cfg.data_dir))
    results = ensure_dir(Path(cfg.results_dir))
    paths = {
        "data": data,
        "raw": ensure_dir(data / "raw"),
        "processed": ensure_dir(data / "processed"),
        "cache": ensure_dir(data / "cache"),
        "metadata": ensure_dir(data / "metadata"),
        "results": results,
        "figures": ensure_dir(results / "figures"),
        "tables": ensure_dir(results / "tables"),
        "csv_figures": ensure_dir(results / "csv_figures"),
        "csv_tables": ensure_dir(results / "csv_tables"),
        "models": ensure_dir(results / "models"),
        "logs": ensure_dir(results / "logs"),
        "manifests": ensure_dir(results / "manifests"),
    }
    return paths


def setup_logger(log_dir: Path) -> logging.Logger:
    logger = logging.getLogger("IME")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(log_dir / "ime_pipeline.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def set_seed(seed: int, deterministic_torch: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def select_device(cfg: Config, logger: logging.Logger) -> torch.device:
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(cfg.torch_float32_matmul_precision)
    torch.set_num_threads(max(1, int(cfg.num_threads)))
    if cfg.use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        props = torch.cuda.get_device_properties(0)
        logger.info(
            "CUDA enabled | device=%s | VRAM=%.2f GB | torch=%s",
            props.name,
            props.total_memory / 1024**3,
            torch.__version__,
        )
        return device
    logger.info("Using CPU | torch=%s", torch.__version__)
    return torch.device("cpu")


def save_config(cfg: Config, paths: Mapping[str, Path]) -> Path:
    out = paths["results"] / "ime_parameters.json"
    payload = asdict(cfg)
    payload["script_version"] = SCRIPT_VERSION
    payload["created_utc"] = utc_now()
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return out


def sha256_file(path: Path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(block)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# -----------------------------------------------------------------------------
# 2. Open-data acquisition
# -----------------------------------------------------------------------------

class DataGouvClient:
    def __init__(self, session: requests.Session, logger: logging.Logger):
        self.session = session
        self.logger = logger

    def search(self, query: str, page_size: int = 20) -> List[Dict[str, Any]]:
        url = f"{DATA_GOUV_API}/datasets/"
        r = self.session.get(url, params={"q": query, "page_size": page_size}, timeout=60)
        r.raise_for_status()
        return r.json().get("data", [])

    @staticmethod
    def _org_name(dataset: Mapping[str, Any]) -> str:
        org = dataset.get("organization") or {}
        return str(org.get("name") or "")

    def resolve_dataset(
        self,
        query: str,
        preferred_org_tokens: Sequence[str] = (),
        title_tokens: Sequence[str] = (),
    ) -> Dict[str, Any]:
        items = self.search(query)
        if not items:
            raise RuntimeError(f"No data.gouv.fr dataset found for query: {query}")

        def score(ds: Mapping[str, Any]) -> float:
            title = str(ds.get("title") or "").lower()
            org = self._org_name(ds).lower()
            s = 0.0
            for tok in preferred_org_tokens:
                if tok.lower() in org:
                    s += 5.0
            for tok in title_tokens:
                if tok.lower() in title:
                    s += 2.0
            # small preference for recent metadata
            s += min(len(ds.get("resources") or []), 20) / 100.0
            return s

        items = sorted(items, key=score, reverse=True)
        chosen = items[0]
        self.logger.info(
            "Resolved data.gouv dataset | query=%r | title=%r | org=%r",
            query,
            chosen.get("title"),
            self._org_name(chosen),
        )
        return chosen

    def choose_resource(
        self,
        dataset: Mapping[str, Any],
        formats: Sequence[str],
        name_tokens: Sequence[str] = (),
        reject_tokens: Sequence[str] = ("documentation", "dictionnaire", "schema"),
    ) -> Dict[str, Any]:
        formats_l = {x.lower().lstrip(".") for x in formats}
        candidates = []
        for res in dataset.get("resources") or []:
            fmt = str(res.get("format") or "").lower().lstrip(".")
            title = str(res.get("title") or res.get("description") or "").lower()
            url = str(res.get("url") or "")
            if formats_l and fmt not in formats_l:
                # permit format inferred from URL
                if not any(url.lower().split("?")[0].endswith("." + f) for f in formats_l):
                    continue
            if any(tok.lower() in title for tok in reject_tokens):
                continue
            score = 0.0
            for tok in name_tokens:
                if tok.lower() in title or tok.lower() in url.lower():
                    score += 2.0
            if res.get("latest"):
                score += 1.0
            if res.get("type") == "main":
                score += 1.0
            candidates.append((score, res))
        if not candidates:
            raise RuntimeError(
                f"No matching resource for dataset={dataset.get('title')} formats={formats}"
            )
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]


def download_file(
    session: requests.Session,
    url: str,
    dest: Path,
    logger: logging.Logger,
    force: bool = False,
    max_mb: Optional[int] = None,
) -> Path:
    if dest.exists() and dest.stat().st_size > 0 and not force:
        logger.info("Data present, skip download | %s", dest)
        return dest
    ensure_dir(dest.parent)
    logger.info("Downloading | %s", url)
    with session.get(url, stream=True, timeout=(30, 300)) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length") or 0)
        if max_mb is not None and total and total > max_mb * 1024**2:
            raise RuntimeError(
                f"Refusing download > {max_mb} MB ({total / 1024**2:.1f} MB): {url}"
            )
        tmp = dest.with_suffix(dest.suffix + ".part")
        n = 0
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if not chunk:
                    continue
                f.write(chunk)
                n += len(chunk)
                if max_mb is not None and n > max_mb * 1024**2:
                    f.close()
                    tmp.unlink(missing_ok=True)
                    raise RuntimeError(f"Download exceeded max_download_mb={max_mb}: {url}")
        tmp.replace(dest)
    logger.info("Downloaded %.2f MB -> %s", dest.stat().st_size / 1024**2, dest)
    return dest


def resource_filename(res: Mapping[str, Any], fallback: str) -> str:
    url = str(res.get("url") or "").split("?")[0]
    name = Path(url).name
    if not name or "." not in name:
        fmt = str(res.get("format") or "").lower().lstrip(".")
        return fallback + (f".{fmt}" if fmt else "")
    return name


def download_public_data(cfg: Config, paths: Mapping[str, Path], logger: logging.Logger) -> pd.DataFrame:
    session = requests.Session()
    session.headers.update({"User-Agent": f"IME-TailQR/{SCRIPT_VERSION} research-pipeline"})
    dg = DataGouvClient(session, logger)
    records: List[Dict[str, Any]] = []

    def record(name: str, source: str, file_path: Optional[Path], status: str, role: str, notes: str = ""):
        records.append(
            {
                "source_name": name,
                "source_url": source,
                "local_path": str(file_path) if file_path else "",
                "status": status,
                "model_role": role,
                "observed_or_constructed": "observed public input",
                "downloaded_utc": utc_now(),
                "sha256": sha256_file(file_path) if file_path and file_path.exists() and file_path.is_file() else "",
                "notes": notes,
            }
        )

    # A. Communal geometry — stable official Etalab endpoint.
    try:
        p = download_file(
            session,
            CONTOURS_URL,
            paths["raw"] / "communes-100m.geojson.gz",
            logger,
            force=cfg.force_download,
            max_mb=cfg.max_download_mb,
        )
        record(
            "Contours administratifs — communes",
            CONTOURS_URL,
            p,
            "ok",
            "fixed geographic support I and adjacency graph G",
            "Etalab/data.gouv latest 100m generalized communal contours",
        )
    except Exception as exc:
        logger.exception("Communal contours download failed")
        record("Contours administratifs — communes", CONTOURS_URL, None, "failed", "spatial support", str(exc))

    # B. Commune metadata/population — API Géo.
    try:
        p = download_file(
            session,
            API_GEO_COMMUNES,
            paths["raw"] / "api_geo_communes_population.json",
            logger,
            force=cfg.force_download,
            max_mb=cfg.max_download_mb,
        )
        record(
            "API Découpage administratif — communes",
            API_GEO_COMMUNES,
            p,
            "ok",
            "exposure proxy and administrative identifiers",
            "Population is an exposure proxy, not insured exposure",
        )
    except Exception as exc:
        logger.exception("API Géo population download failed")
        record("API Découpage administratif — communes", API_GEO_COMMUNES, None, "failed", "exposure proxy", str(exc))

    # C. GASPAR / CatNat national CSV.
    try:
        gaspar_errors = []
        ds = None
        for gaspar_query in (
            "GASPAR",
            "Base nationale Gestion Assistée Procédures Administratives Risques GASPAR",
            "Gestion Assistée Procédures Administratives Risques",
        ):
            try:
                ds = dg.resolve_dataset(
                    gaspar_query,
                    preferred_org_tokens=("transition", "écologique", "ecologique", "ministère", "ministere"),
                    title_tokens=("GASPAR", "Gestion"),
                )
                break
            except Exception as gaspar_exc:
                gaspar_errors.append(f"{gaspar_query}: {gaspar_exc}")
        if ds is None:
            raise RuntimeError("; ".join(gaspar_errors))
        res = dg.choose_resource(ds, ("zip", "csv"), name_tokens=("gaspar", "catnat"))
        fname = resource_filename(res, "gaspar")
        p = download_file(session, res["url"], paths["raw"] / fname, logger, cfg.force_download, cfg.max_download_mb)
        if p.suffix.lower() == ".zip":
            try:
                extract_gaspar_archive(paths, logger)
            except Exception as exc_extract:
                logger.warning("GASPAR downloaded but extraction failed: %s", exc_extract)
        record(
            "GASPAR — procédures administratives / CatNat",
            res["url"],
            p,
            "ok",
            "historical hazard-event information",
            "Administrative CatNat recognition is not an insured-claims database",
        )
    except Exception as exc:
        logger.exception("GASPAR resolution/download failed")
        record("GASPAR", "https://www.data.gouv.fr/", None, "failed", "hazard history", str(exc))

    # D. RGA 2026. We resolve dynamically because data.gouv resource URLs evolve.
    try:
        ds = dg.resolve_dataset(
            "Aléas retraits gonflements argiles France métropolitaine",
            preferred_org_tokens=("recherches géologiques", "brgm"),
            title_tokens=("argiles", "France métropolitaine"),
        )
        file_candidates = []
        for rr in ds.get("resources") or []:
            url = str(rr.get("url") or "")
            fmt = str(rr.get("format") or "").lower()
            if url.startswith("http") and not ("/api/" in url.lower() or url.rstrip().endswith("?")):
                if fmt in {"geojson", "json", "shp", "zip"} or url.lower().split("?")[0].endswith((".geojson", ".json", ".zip", ".shp")):
                    file_candidates.append(rr)
        if not file_candidates:
            raise RuntimeError("No downloadable RGA vector file resource found; API endpoint deliberately skipped")
        res = file_candidates[0]
        fname = resource_filename(res, "rga_2026")
        p = download_file(session, res["url"], paths["raw"] / fname, logger, cfg.force_download, cfg.max_download_mb)
        record(
            "BRGM/Géorisques — RGA",
            res["url"],
            p,
            "ok",
            "structural drought/RGA hazard layer",
            "2026 official RGA layer when available in JSON/GeoJSON",
        )
    except Exception as exc:
        logger.warning("RGA vector layer unavailable through resolver: %s", exc)
        record(
            "BRGM/Géorisques — RGA",
            "https://www.georisques.gouv.fr/donnees/bases-de-donnees/retrait-gonflement-des-argiles",
            None,
            "optional-fallback",
            "RGA hazard layer",
            "Fallback uses drought CatNat history as an explicit RGA-risk proxy; never mislabeled as official RGA polygon coverage.",
        )

    # E. Météo-France daily climate. Download only a bounded subset automatically.
    #    Resource names are department/period-specific and evolve, therefore discovery is dynamic.
    if cfg.download_meteo:
        try:
            ds = dg.resolve_dataset(
                "Données climatologiques de base quotidiennes",
                preferred_org_tokens=("météo", "meteo"),
                title_tokens=("quotidiennes",),
            )
            resources = []
            for res in ds.get("resources") or []:
                fmt = str(res.get("format") or "").lower()
                title = str(res.get("title") or "")
                url = str(res.get("url") or "")
                if fmt in {"csv.gz", "csv", "gz"} or url.lower().endswith((".csv.gz", ".csv")):
                    resources.append(res)
            # Prefer recent period files; cap count to prevent accidental very large downloads.
            resources = sorted(resources, key=lambda r: str(r.get("last_modified") or r.get("modified") or ""), reverse=True)
            resources = resources[: max(1, int(cfg.max_meteo_departments))]
            meteo_dir = ensure_dir(paths["raw"] / "meteo_daily")
            ok = 0
            for idx, res in enumerate(resources, start=1):
                try:
                    fname = resource_filename(res, f"meteo_{idx:03d}.csv.gz")
                    download_file(session, res["url"], meteo_dir / fname, logger, cfg.force_download, cfg.max_download_mb)
                    ok += 1
                except Exception as sub_exc:
                    logger.warning("Meteo resource skipped (%s): %s", idx, sub_exc)
            record(
                "Météo-France — climatologie quotidienne",
                str(ds.get("page") or ds.get("uri") or "https://www.data.gouv.fr/"),
                meteo_dir if ok else None,
                "ok" if ok else "failed",
                "time-varying climate hazard dynamics",
                f"Downloaded {ok}/{len(resources)} discovered resources",
            )
        except Exception as exc:
            logger.warning("Météo-France automatic acquisition failed: %s", exc)
            record("Météo-France daily climate", "https://www.data.gouv.fr/", None, "optional-fallback", "climate dynamics", str(exc))

    # F. Optional auxiliary modules — not used in climate baseline unless explicitly requested.
    if cfg.download_ssmsi:
        try:
            ds = dg.resolve_dataset(
                "bases statistiques communale délinquance police gendarmerie",
                preferred_org_tokens=("intérieur", "interieur"),
                title_tokens=("communale", "délinquance"),
            )
            res = dg.choose_resource(ds, ("parquet", "csv"), name_tokens=("communal", "commune"))
            fname = resource_filename(res, "ssmsi_communal.parquet")
            p = download_file(session, res["url"], paths["raw"] / fname, logger, cfg.force_download, cfg.max_download_mb)
            record("SSMSI — délinquance communale", res["url"], p, "ok", "optional MRH theft/vandalism covariate")
        except Exception as exc:
            logger.warning("SSMSI optional acquisition failed: %s", exc)

    if cfg.download_baac:
        try:
            ds = dg.resolve_dataset(
                "bases annuelles accidents corporels circulation routière 2005 2024",
                preferred_org_tokens=("intérieur", "sécurité routière", "securite routiere"),
                title_tokens=("accidents", "corporels"),
            )
            meta_path = paths["metadata"] / "baac_dataset_metadata.json"
            meta_path.write_text(json.dumps(ds, ensure_ascii=False, indent=2), encoding="utf-8")
            record("ONISR — BAAC", str(ds.get("page") or "https://www.data.gouv.fr/"), meta_path, "metadata-only", "optional auto-risk extension")
        except Exception as exc:
            logger.warning("BAAC optional discovery failed: %s", exc)

    manifest = pd.DataFrame(records)
    manifest.to_csv(paths["metadata"] / "data_download_manifest.csv", index=False)
    return manifest


# -----------------------------------------------------------------------------
# 3. Robust loaders and spatial graph
# -----------------------------------------------------------------------------


def read_json_maybe_gzip(path: Path) -> Any:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(path.read_text(encoding="utf-8"))


def load_communes(paths: Mapping[str, Path], cfg: Config, logger: logging.Logger) -> gpd.GeoDataFrame:
    p = paths["raw"] / "communes-100m.geojson.gz"
    if not p.exists():
        raise FileNotFoundError(f"Missing communal geometry: {p}")
    logger.info("Loading communal geometry")
    gdf = gpd.read_file(f"/vsigzip/{p.resolve()}") if p.suffix == ".gz" else gpd.read_file(p)
    # normalize code/name columns
    code_col = first_existing(gdf.columns, ["code", "CODE", "INSEE_COM", "code_insee", "id"])
    name_col = first_existing(gdf.columns, ["nom", "NOM", "NOM_COM", "name"])
    if code_col is None:
        raise RuntimeError(f"Cannot identify commune code column. Columns={list(gdf.columns)[:30]}")
    gdf = gdf.rename(columns={code_col: "code_commune"})
    if name_col:
        gdf = gdf.rename(columns={name_col: "commune_name"})
    gdf["code_commune"] = gdf["code_commune"].astype(str).str.zfill(5)
    if cfg.metropolitan_only:
        # Mainland + Corsica: standard five-char commune codes starting 01..95, including 2A/2B.
        dep = gdf["code_commune"].map(department_from_commune_code)
        mask = dep.map(is_metropolitan_department)
        gdf = gdf.loc[mask].copy()
        gdf["department_code"] = dep.loc[mask].values
    else:
        gdf["department_code"] = gdf["code_commune"].map(department_from_commune_code)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    gdf = gdf.to_crs(4326)
    logger.info("Communes loaded | n=%s", len(gdf))
    return gdf


def first_existing(columns: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    cols = list(columns)
    lower = {str(c).lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def department_from_commune_code(code: str) -> str:
    code = str(code).strip().upper().zfill(5)
    if code.startswith("2A") or code.startswith("2B"):
        return code[:2]
    if code.startswith("97") or code.startswith("98"):
        return code[:3]
    return code[:2]


def is_metropolitan_department(dep: str) -> bool:
    dep = str(dep).upper()
    if dep in {"2A", "2B"}:
        return True
    return dep.isdigit() and 1 <= int(dep) <= 95


def load_population(paths: Mapping[str, Path], logger: logging.Logger) -> pd.DataFrame:
    p = paths["raw"] / "api_geo_communes_population.json"
    if not p.exists():
        logger.warning("Population API file absent; exposure proxy will be geometric area")
        return pd.DataFrame(columns=["code_commune", "population"])
    data = json.loads(p.read_text(encoding="utf-8"))
    rows = []
    for x in data:
        rows.append(
            {
                "code_commune": str(x.get("code", "")).zfill(5),
                "population": pd.to_numeric(x.get("population"), errors="coerce"),
                "commune_name_api": x.get("nom"),
            }
        )
    out = pd.DataFrame(rows)
    logger.info("Population metadata loaded | n=%s", len(out))
    return out


def find_raw_file(paths: Mapping[str, Path], tokens: Sequence[str], suffixes: Sequence[str]) -> Optional[Path]:
    files = []
    for p in paths["raw"].rglob("*"):
        if not p.is_file():
            continue
        name = p.name.lower()
        if suffixes and not any(name.endswith(s.lower()) for s in suffixes):
            continue
        score = sum(tok.lower() in name for tok in tokens)
        if score:
            files.append((score, p))
    if not files:
        return None
    files.sort(key=lambda x: (x[0], x[1].stat().st_mtime), reverse=True)
    return files[0][1]


def robust_read_csv(path: Path) -> pd.DataFrame:
    attempts = [
        {"sep": ";", "encoding": "utf-8"},
        {"sep": ",", "encoding": "utf-8"},
        {"sep": ";", "encoding": "latin-1"},
        {"sep": ",", "encoding": "latin-1"},
        {"sep": "\t", "encoding": "utf-8"},
    ]
    last = None
    for kw in attempts:
        try:
            df = pd.read_csv(path, low_memory=False, **kw)
            if df.shape[1] >= 2:
                return df
        except Exception as exc:
            last = exc
    raise RuntimeError(f"Cannot parse CSV {path}: {last}")


def infer_column(df: pd.DataFrame, regexes: Sequence[str]) -> Optional[str]:
    for pat in regexes:
        rgx = re.compile(pat, flags=re.I)
        for c in df.columns:
            if rgx.search(str(c)):
                return c
    return None


def _connected_components(nodes: Sequence[str], adjacency: Mapping[str, Sequence[str]]) -> List[List[str]]:
    remaining = set(nodes)
    comps: List[List[str]] = []
    while remaining:
        start = min(remaining)
        q = deque([start])
        seen = {start}
        remaining.remove(start)
        while q:
            u = q.popleft()
            for v in adjacency.get(u, ()):
                if v in remaining:
                    remaining.remove(v)
                    seen.add(v)
                    q.append(v)
        comps.append(sorted(seen))
    comps.sort(key=lambda c: (-len(c), c[0]))
    return comps


def _repair_department_topology(
    work: gpd.GeoDataFrame,
    adjacency: Dict[str, set],
    paths: Mapping[str, Path],
    logger: logging.Logger,
) -> Dict[str, set]:
    """Add the minimum deterministic intra-department bridges required for connectivity.

    Pure polygon-touch adjacency may split a department because of islands or
    cartographic generalisation.  Section 2 only requires a fixed adopted
    spatial topology.  V4 therefore preserves geometric contiguity wherever
    available and adds auditable intra-department bridge edges only when an
    administrative department is otherwise disconnected.
    """
    cent = work.geometry.centroid
    xy = {c: (float(p.x), float(p.y)) for c, p in zip(work.code_commune, cent)}
    dep_map = dict(zip(work.code_commune, work.department_code.astype(str)))
    bridge_rows: List[Dict[str, Any]] = []

    for dep in sorted(set(dep_map.values())):
        nodes = sorted([c for c, d in dep_map.items() if d == dep])
        if len(nodes) <= 1:
            continue
        comps = _connected_components(nodes, adjacency)
        if len(comps) <= 1:
            continue
        logger.warning(
            "Department topology fragmented | department=%s components=%s; adding deterministic intra-department bridges",
            dep, len(comps),
        )
        anchor = set(comps[0])
        for comp in comps[1:]:
            # Exact nearest centroid pair between the current connected anchor
            # and the disconnected component.  Fragmented components are rare,
            # so this O(|A||C|) calculation is negligible relative to graph build.
            best = None
            for u in sorted(anchor):
                xu, yu = xy[u]
                for v in comp:
                    xv, yv = xy[v]
                    d2 = (xu - xv) ** 2 + (yu - yv) ** 2
                    key = (d2, u, v)
                    if best is None or key < best:
                        best = key
            if best is None:
                raise RuntimeError(f"Cannot repair fragmented department topology for {dep}")
            d2, u, v = best
            adjacency[u].add(v)
            adjacency[v].add(u)
            distance_km = math.sqrt(d2) / 1000.0
            bridge_rows.append({
                "department_code": dep,
                "commune_a": u,
                "commune_b": v,
                "centroid_distance_km": distance_km,
                "reason": "department_connectivity_repair",
            })
            anchor.update(comp)

    audit = pd.DataFrame(bridge_rows, columns=[
        "department_code", "commune_a", "commune_b",
        "centroid_distance_km", "reason"
    ])
    audit.to_csv(paths["metadata"] / "adjacency_bridge_audit_v4.csv", index=False)
    logger.info(
        "Topology repair complete | bridge_edges=%s max_bridge_km=%s",
        len(audit),
        f"{audit.centroid_distance_km.max():.2f}" if len(audit) else "0.00",
    )
    return adjacency


def build_adjacency(gdf: gpd.GeoDataFrame, paths: Mapping[str, Path], cfg: Config, logger: logging.Logger) -> Dict[str, List[str]]:
    cache = paths["processed"] / "commune_adjacency_v4.json"
    if cache.exists() and not cfg.force_compute:
        logger.info("Loading cached V4 commune adjacency")
        out = json.loads(cache.read_text(encoding="utf-8"))
        return {str(k): list(v) for k, v in out.items()}

    logger.info("Building communal adjacency graph (touches/intersects via spatial index)")
    # Work in projected CRS to reduce topological precision issues.
    work = gdf[["code_commune", "department_code", "geometry"]].to_crs(2154).reset_index(drop=True)
    work["department_code"] = work["department_code"].astype(str)
    codes = work["code_commune"].tolist()
    geoms = work.geometry.values
    sindex = work.sindex
    adjacency: Dict[str, set] = {c: set() for c in codes}

    for i, geom in enumerate(geoms):
        cand = list(sindex.query(geom, predicate="intersects"))
        ci = codes[i]
        for j in cand:
            if j <= i:
                continue
            gj = geoms[j]
            if geom.touches(gj) or geom.boundary.intersects(gj.boundary):
                cj = codes[j]
                adjacency[ci].add(cj)
                adjacency[cj].add(ci)
        if (i + 1) % 5000 == 0:
            logger.info("Adjacency progress %s/%s", i + 1, len(work))

    pure_edges = sum(len(v) for v in adjacency.values()) // 2
    if cfg.repair_department_connectivity:
        adjacency = _repair_department_topology(work, adjacency, paths, logger)

    out = {k: sorted(v) for k, v in adjacency.items()}
    cache.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    n_edges = sum(len(v) for v in out.values()) // 2
    logger.info(
        "Adjacency built | nodes=%s pure_edges=%s final_edges=%s bridges=%s",
        len(out), pure_edges, n_edges, n_edges - pure_edges,
    )
    return out


# -----------------------------------------------------------------------------
# 4. Hazard/exposure features and calibrated actuarial loss environment
# -----------------------------------------------------------------------------

DROUGHT_PATTERNS = (
    "sécheresse",
    "secheresse",
    "réhydratation",
    "rehydratation",
    "mouvements de terrain différentiels",
    "retrait gonflement",
)
FLOOD_PATTERNS = ("inondation", "coulée de boue", "coulee de boue", "submersion")
STORM_PATTERNS = ("tempête", "tempete", "cyclone", "vent")


def normalize_text(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().str.normalize("NFKD").str.encode("ascii", errors="ignore").str.decode("ascii")


def extract_gaspar_archive(paths: Mapping[str, Path], logger: logging.Logger) -> Optional[Path]:
    """Extract GASPAR ZIP safely and return the CatNat CSV path if present."""
    raw = paths["raw"]
    archives = sorted(raw.glob("*gaspar*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not archives:
        return None
    archive = archives[0]
    out_dir = ensure_dir(raw / "gaspar_extracted")
    try:
        with zipfile.ZipFile(archive, "r") as zf:
            safe_members = []
            for info in zf.infolist():
                target = (out_dir / info.filename).resolve()
                if out_dir.resolve() not in target.parents and target != out_dir.resolve():
                    raise RuntimeError(f"Unsafe path in GASPAR archive: {info.filename}")
                safe_members.append(info)
            zf.extractall(out_dir, members=safe_members)
        candidates = sorted(out_dir.rglob("*.csv"))
        catnat = [p for p in candidates if "catnat" in p.name.lower()]
        chosen = catnat[0] if catnat else (max(candidates, key=lambda p: p.stat().st_size) if candidates else None)
        if chosen is not None:
            logger.info("GASPAR extracted | archive=%s catnat=%s", archive.name, chosen.name)
        return chosen
    except zipfile.BadZipFile as exc:
        logger.error("Invalid GASPAR ZIP: %s", archive)
        raise RuntimeError(f"Invalid GASPAR ZIP: {archive}") from exc


def load_catnat_events(paths: Mapping[str, Path], logger: logging.Logger) -> pd.DataFrame:
    p = find_raw_file(paths, ("catnat", "gaspar"), (".csv",))
    if p is None:
        p = extract_gaspar_archive(paths, logger)
    if p is None:
        # Last-resort fallback: large top-level CSV, excluding weather files.
        candidates = [x for x in paths["raw"].glob("*.csv") if x.stat().st_size > 1_000_000 and "meteo" not in x.name.lower()]
        p = max(candidates, key=lambda x: x.stat().st_size) if candidates else None
    if p is None:
        logger.warning("No GASPAR CSV found")
        return pd.DataFrame(columns=["code_commune", "year", "hazard_text"])

    logger.info("Parsing GASPAR/CatNat | %s", p)
    df = robust_read_csv(p)
    code_col = infer_column(df, [r"cod.*comm", r"insee", r"code.*insee", r"com.*cod"])
    hazard_col = infer_column(df, [r"lib.*risq", r"lib.*phen", r"lib.*alea", r"risque", r"phenom", r"nature"])
    date_col = infer_column(df, [r"date.*debut", r"date.*arrete", r"date.*pub", r"date", r"debut"])
    if code_col is None or hazard_col is None:
        logger.warning("GASPAR schema not recognized | columns=%s", list(df.columns)[:50])
        return pd.DataFrame(columns=["code_commune", "year", "hazard_text"])
    out = pd.DataFrame()
    out["code_commune"] = df[code_col].astype(str).str.extract(r"([0-9A-Za-z]{5})", expand=False).str.upper()
    out["hazard_text"] = normalize_text(df[hazard_col])
    if date_col:
        # GASPAR currently uses ISO-like timestamps; parse those first to avoid
        # locale ambiguity, then fall back to French day-first dates if needed.
        raw_date = df[date_col]
        d = pd.to_datetime(raw_date, errors="coerce", format="mixed", dayfirst=False)
        if d.isna().mean() > 0.50:
            d = pd.to_datetime(raw_date, errors="coerce", format="mixed", dayfirst=True)
        out["year"] = d.dt.year
    else:
        year_col = infer_column(df, [r"annee", r"year"])
        out["year"] = pd.to_numeric(df[year_col], errors="coerce") if year_col else np.nan
    out = out.dropna(subset=["code_commune", "year"])
    out["year"] = out["year"].astype(int)
    out = out[(out["year"] >= 1982) & (out["year"] <= datetime.now().year)]
    logger.info("CatNat events parsed | rows=%s years=%s-%s", len(out), out.year.min() if len(out) else None, out.year.max() if len(out) else None)
    return out


def classify_catnat(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.assign(rga_event=[], flood_event=[], storm_event=[])
    txt = df["hazard_text"].fillna("")
    def anypat(patterns: Sequence[str]) -> pd.Series:
        return pd.Series(False, index=df.index) if not patterns else np.logical_or.reduce([txt.str.contains(p, regex=False) for p in patterns])
    out = df.copy()
    out["rga_event"] = anypat(tuple(x.encode("ascii", "ignore").decode() for x in DROUGHT_PATTERNS)).astype(int)
    out["flood_event"] = anypat(tuple(x.encode("ascii", "ignore").decode() for x in FLOOD_PATTERNS)).astype(int)
    out["storm_event"] = anypat(tuple(x.encode("ascii", "ignore").decode() for x in STORM_PATTERNS)).astype(int)
    return out


def build_commune_year_panel(
    communes: gpd.GeoDataFrame,
    population: pd.DataFrame,
    catnat: pd.DataFrame,
    cfg: Config,
    paths: Mapping[str, Path],
    logger: logging.Logger,
) -> pd.DataFrame:
    cache = paths["processed"] / "commune_year_risk_panel_v4.parquet"
    if cache.exists() and not cfg.force_compute:
        logger.info("Loading cached risk panel")
        return pd.read_parquet(cache)

    geo = communes[["code_commune", "department_code", "geometry"]].copy()
    geo2154 = geo.to_crs(2154)
    geo["area_km2"] = geo2154.geometry.area.values / 1e6
    base = geo.drop(columns="geometry").merge(population, on="code_commune", how="left")
    if "population" not in base:
        base["population"] = np.nan
    # Explicit proxy: population when available, else area-scaled support.
    fallback = np.maximum(base["area_km2"].to_numpy(), 1.0) * 100.0
    pop = pd.to_numeric(base["population"], errors="coerce").to_numpy(dtype=float)
    exposure = np.where(np.isfinite(pop) & (pop > 0), pop, fallback)
    base["exposure_proxy"] = exposure
    base["log_exposure"] = np.log1p(exposure)

    years = np.arange(cfg.start_year, cfg.end_year + 1)
    panel = base.loc[base.index.repeat(len(years))].reset_index(drop=True)
    panel["year"] = np.tile(years, len(base))

    cat = classify_catnat(catnat)
    if not cat.empty:
        agg = cat.groupby(["code_commune", "year"], as_index=False)[["rga_event", "flood_event", "storm_event"]].sum()
        panel = panel.merge(agg, on=["code_commune", "year"], how="left")
    for c in ["rga_event", "flood_event", "storm_event"]:
        if c not in panel:
            panel[c] = 0
        panel[c] = panel[c].fillna(0).astype(float)

    # Historical, strictly predecision feature construction: rolling sums use years < t.
    panel = panel.sort_values(["code_commune", "year"]).reset_index(drop=True)
    for c in ["rga_event", "flood_event", "storm_event"]:
        hist = (
            panel.groupby("code_commune", sort=False)[c]
            .transform(lambda s: s.shift(1).rolling(cfg.rolling_window_years, min_periods=1).sum())
            .fillna(0.0)
        )
        panel[c.replace("_event", "_hist")] = hist

    # Cross-sectional ranks are stable, transparent hazard indices in [0,1].
    for src, dst in [
        ("rga_hist", "rga_hazard"),
        ("flood_hist", "flood_hazard"),
        ("storm_hist", "storm_hazard"),
    ]:
        panel[dst] = panel.groupby("year")[src].rank(pct=True, method="average").fillna(0.5)

    # Vulnerability proxy: exposure density. It is labeled as proxy in Table 1/manifest.
    density = panel["exposure_proxy"] / panel["area_km2"].clip(lower=0.1)
    panel["vulnerability_proxy"] = panel.groupby("year")[density.name if density.name in panel else "exposure_proxy"].rank(pct=True) if False else 0.0
    # rank explicitly from temporary series
    panel["_density"] = density
    panel["vulnerability_proxy"] = panel.groupby("year")["_density"].rank(pct=True, method="average").fillna(0.5)
    panel.drop(columns=["_density"], inplace=True)

    # Composite predecision hazard. No claim that weights are statistically identified.
    panel["hazard_index"] = (
        0.50 * panel["rga_hazard"]
        + 0.35 * panel["flood_hazard"]
        + 0.15 * panel["storm_hazard"]
    )

    # Conditional mean loss cost proxy. Calibrated monetary scale; NOT observed claims.
    panel["mhat"] = cfg.loss_scale * np.exp(
        1.00 * panel["hazard_index"]
        + 0.35 * panel["vulnerability_proxy"]
        - 0.55
    )

    # Conditional quantile approximation from a heavy-tail multiplicative environment.
    # We intentionally save each qhat_tau to make the risk representation auditable.
    rng = np.random.default_rng(1234567)
    sim_n = 60_000
    eps = np.exp(cfg.idiosyncratic_sigma * rng.standard_normal(sim_n) - 0.5 * cfg.idiosyncratic_sigma**2)
    tail = np.maximum(rng.standard_t(df=cfg.tail_df, size=sim_n), 0.0)
    mult = eps * (1.0 + 0.20 * tail)
    for tau in cfg.loss_quantiles:
        qmult = float(np.quantile(mult, tau))
        panel[f"qhat_{tau:.2f}"] = panel["mhat"] * qmult

    panel["tail_mean_ratio"] = panel[f"qhat_{max(cfg.loss_quantiles):.2f}"] / panel["mhat"].clip(lower=1e-9)
    panel.to_parquet(cache, index=False)
    logger.info("Risk panel built | rows=%s communes=%s years=%s", len(panel), panel.code_commune.nunique(), panel.year.nunique())
    return panel


def add_dependence_summary(panel: pd.DataFrame, adjacency: Mapping[str, Sequence[str]], cfg: Config, paths: Mapping[str, Path], logger: logging.Logger) -> pd.DataFrame:
    cache = paths["processed"] / "commune_year_risk_panel_with_dependence_v3.parquet"
    if cache.exists() and not cfg.force_compute:
        out = pd.read_parquet(cache)
        if "delta_hat" in out and np.isfinite(pd.to_numeric(out["delta_hat"], errors="coerce")).all():
            logger.info("Loading validated cached dependence panel")
            return out
        logger.warning("Cached dependence panel contains non-finite values; rebuilding")
    logger.info("Constructing spatial dependence summary Deltahat_t [finite-safe]")
    out_parts = []
    diagnostics = []
    for year, gy in panel.groupby("year", sort=True):
        z = gy.copy()
        h = pd.to_numeric(z["hazard_index"], errors="coerce").to_numpy(dtype=float)
        h = np.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
        risk = dict(zip(z.code_commune.astype(str), h))
        neigh = []
        for code, own in zip(z.code_commune.astype(str), h):
            x = [finite_float(risk.get(n), default=np.nan) for n in adjacency.get(code, ()) if n in risk]
            x = [v for v in x if math.isfinite(v)]
            neigh.append(float(np.mean(x)) if x else float(own))
        nh = np.asarray(neigh, dtype=float)
        finite = np.isfinite(h) & np.isfinite(nh)
        hf, nf = h[finite], nh[finite]
        sx = float(np.std(hf)) if hf.size else 0.0
        sy = float(np.std(nf)) if nf.size else 0.0
        if hf.size >= 3 and sx > 1e-12 and sy > 1e-12:
            hx = hf - hf.mean(); ny = nf - nf.mean()
            den = math.sqrt(float(np.dot(hx, hx) * np.dot(ny, ny)))
            delta_scalar = float(np.dot(hx, ny) / den) if den > 1e-18 else 0.0
        else:
            delta_scalar = 0.0
        if not math.isfinite(delta_scalar):
            delta_scalar = 0.0
        delta_scalar = float(np.clip(delta_scalar, -1.0, 1.0))
        z["neighbor_hazard"] = np.nan_to_num(nh, nan=0.0, posinf=0.0, neginf=0.0)
        z["delta_hat"] = delta_scalar
        out_parts.append(z)
        diagnostics.append({"year": int(year), "delta_hat": delta_scalar, "hazard_sd": sx, "neighbor_hazard_sd": sy, "n_finite": int(finite.sum())})
    out = pd.concat(out_parts, ignore_index=True)
    for col in ["hazard_index", "neighbor_hazard", "delta_hat", "mhat", "exposure_proxy"]:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    required = [c for c in ["hazard_index", "neighbor_hazard", "delta_hat", "mhat", "exposure_proxy"] if c in out]
    if out[required].isna().any().any():
        bad = out[required].isna().sum().to_dict()
        logger.warning("Non-finite risk-panel values sanitized | %s", bad)
        out[required] = out[required].fillna(0.0)
    pd.DataFrame(diagnostics).to_csv(paths["metadata"] / "dependence_diagnostics_v4.csv", index=False)
    out.to_parquet(cache, index=False)
    logger.info("Dependence summary ready | years=%s delta_range=[%.4f, %.4f]", out.year.nunique(), out.delta_hat.min(), out.delta_hat.max())
    return out


def simulate_loss_vector(
    year_df: pd.DataFrame,
    seed: int,
    cfg: Config,
    stress: str = "historical",
    exposure_multiplier: Optional[np.ndarray] = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(year_df)
    m = year_df["mhat"].to_numpy(dtype=float)
    hazard = year_df["hazard_index"].to_numpy(dtype=float)
    if stress == "mean_shift":
        m = m * (1.0 + cfg.stress_mean_shift)
    elif stress == "tail_amplification":
        pass
    elif stress == "spatial_concentration":
        q = np.quantile(hazard, 0.80)
        m = m * (1.0 + cfg.stress_spatial_concentration * (hazard >= q))
    common = rng.standard_t(df=max(cfg.tail_df, 2.1))
    common_mult = math.exp(cfg.spatial_common_factor * np.clip(common, -3, 4) * 0.20)
    eps = np.exp(cfg.idiosyncratic_sigma * rng.standard_normal(n) - 0.5 * cfg.idiosyncratic_sigma**2)
    tail = np.maximum(rng.standard_t(df=cfg.tail_df, size=n), 0.0)
    tail_scale = cfg.stress_tail_multiplier if stress == "tail_amplification" else 1.0
    mult = common_mult * eps * (1.0 + 0.20 * tail_scale * tail)
    exposure = year_df["exposure_proxy"].to_numpy(dtype=float)
    if exposure_multiplier is not None:
        exposure = exposure * exposure_multiplier
    # m is per exposure unit; use /1000 to keep monetary magnitudes manageable.
    return exposure * m * mult / 1000.0


# -----------------------------------------------------------------------------
# 5. Partitions, pricing, and finite admissible actions
# -----------------------------------------------------------------------------

@dataclass
class Partition:
    zone_of: Dict[str, int]
    _signature_cache: Optional[str] = field(default=None, init=False, repr=False, compare=False)

    def copy(self) -> "Partition":
        return Partition(dict(self.zone_of))

    def zones(self) -> Dict[int, List[str]]:
        z: Dict[int, List[str]] = defaultdict(list)
        for node, k in self.zone_of.items():
            z[int(k)].append(node)
        return dict(z)

    @property
    def k(self) -> int:
        return len(set(self.zone_of.values()))

    def signature(self) -> str:
        """Fast deterministic signature used by V5 metric/action caches.

        Partitions are canonicalized before entering the environment, so the vector
        of zone labels in dictionary insertion order is sufficient to identify the
        computational partition generated by this pipeline.
        """
        if self._signature_cache is None:
            # V9 cache-safety: include commune identities, not only the label sequence.
            # This prevents cache aliasing when equal-length mappings have different
            # insertion orders or when distinct shifted units yield the same raw label vector.
            h = hashlib.blake2b(digest_size=12)
            for k, v in self.zone_of.items():
                h.update(str(k).encode("utf-8"))
                h.update(b"\x00")
                h.update(int(v).to_bytes(2, byteorder="little", signed=True))
            self._signature_cache = h.hexdigest()
        return self._signature_cache

    def canonicalize(self) -> "Partition":
        groups = self.zones()
        ordered = sorted(groups.values(), key=lambda nodes: min(nodes))
        mapping = {}
        for newk, nodes in enumerate(ordered):
            for n in nodes:
                mapping[n] = newk
        return Partition(mapping)


@dataclass
class Action:
    action_id: str
    edit_type: str
    partition: Optional[Partition]
    relativities: Dict[int, float]
    geographic_score: float = 0.0
    features: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    # V9 lazy execution: MERGE/SHIFT actions may carry only an edit descriptor,
    # exact sufficient statistics and precomputed metrics until env.step() executes them.
    lazy_descriptor: Optional[Tuple[Any, ...]] = None
    lazy_zone_stats: Optional[pd.DataFrame] = None
    lazy_metrics: Optional[Dict[str, float]] = None
    lazy_k: Optional[int] = None


def initial_department_partition(communes: pd.DataFrame) -> Partition:
    deps = sorted(communes["department_code"].astype(str).unique())
    dep_to_zone = {d: i for i, d in enumerate(deps)}
    return Partition({r.code_commune: dep_to_zone[str(r.department_code)] for r in communes.itertuples()}).canonicalize()


def validate_initial_partition(
    part: Partition,
    year_df: pd.DataFrame,
    adjacency: Mapping[str, Sequence[str]],
    cfg: Config,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Fail early with an auditable diagnosis if the initial partition is invalid."""
    groups = part.zones()
    exposure = dict(zip(year_df.code_commune.astype(str), year_df.exposure_proxy.astype(float)))
    issues: List[str] = []
    if not (1 <= len(groups) <= cfg.k_max):
        issues.append(f"zone_count={len(groups)} outside [1,{cfg.k_max}]")
    for k, nodes in groups.items():
        e = sum(float(exposure.get(str(n), 0.0)) for n in nodes)
        if e < cfg.e_min:
            issues.append(f"zone={k}: exposure={e:.3f}<E_min={cfg.e_min}")
        if not is_connected_nodes(nodes, adjacency):
            comps = _connected_components(nodes, adjacency)
            issues.append(f"zone={k}: disconnected components={len(comps)}")
    if issues:
        msg = "Initial partition inadmissible after topology construction | " + "; ".join(issues[:20])
        if logger is not None:
            logger.error(msg)
        raise RuntimeError(msg)
    if logger is not None:
        logger.info(
            "Initial partition admissibility PASS | zones=%s E_min=%.2f",
            len(groups), cfg.e_min,
        )


def is_connected_nodes(nodes: Sequence[str], adjacency: Mapping[str, Sequence[str]]) -> bool:
    nodes_set = set(nodes)
    if len(nodes_set) <= 1:
        return True
    start = next(iter(nodes_set))
    seen = {start}
    q = deque([start])
    while q:
        u = q.popleft()
        for v in adjacency.get(u, ()):
            if v in nodes_set and v not in seen:
                seen.add(v)
                q.append(v)
    return len(seen) == len(nodes_set)


# V7 fast caches. They are process-local and keyed by the identity of the annual
# DataFrame plus a deterministic partition signature.  This avoids repeated
# pandas joins/groupbys for the same state while keeping the scientific objects
# unchanged.
_YEAR_ARRAY_CACHE: Dict[int, Dict[str, Any]] = {}
_FAST_CACHE_MAX = 256
_ZONE_STATS_CACHE: "OrderedDict[Tuple[int, str], pd.DataFrame]" = OrderedDict()
_PSI_Z_CACHE: "OrderedDict[Tuple[int, str], float]" = OrderedDict()
_ADMISSIBLE_CACHE: "OrderedDict[Tuple[int, str], bool]" = OrderedDict()
_ASSIGNMENT_CACHE: "OrderedDict[Tuple[int, str], np.ndarray]" = OrderedDict()

# V7 structural cache: only a handful of recent partitions are retained because
# each partition references the full commune support.  The cache stores groups
# and boundary primitives, never full action sets.
_PARTITION_STRUCTURE_CACHE: "OrderedDict[Tuple[int, str], Dict[str, Any]]" = OrderedDict()
_ACTION_METRIC_CACHE: "OrderedDict[Tuple[Any, ...], Dict[str, float]]" = OrderedDict()
# V8 exact sufficient-statistics caches.
_ZONE_MOMENT_CACHE: "OrderedDict[Tuple[int, str], Dict[str, np.ndarray]]" = OrderedDict()
_TRANSITION_SUMMARY_CACHE: "OrderedDict[Tuple[int, str, str], Tuple[np.ndarray, np.ndarray, np.ndarray]]" = OrderedDict()
_TURNOVER_CACHE: "OrderedDict[Tuple[int, str, str], float]" = OrderedDict()
_PARTITION_COMMON_METRIC_CACHE: "OrderedDict[Tuple[int, str, str], Dict[str, float]]" = OrderedDict()


def _articulation_points_subset(nodes: Sequence[str], adjacency: Mapping[str, Sequence[str]]) -> set:
    """Tarjan articulation points on one induced zone graph, O(|V|+|E|).

    Used only for source zones appearing among shortlisted SHIFT candidates.
    Removing a non-articulation boundary commune preserves source connectivity.
    """
    node_set = set(nodes)
    if len(node_set) <= 2:
        # In a 2-node connected graph, removing either node leaves one connected
        # singleton, so neither node is an articulation point.
        return set()
    disc: Dict[str, int] = {}
    low: Dict[str, int] = {}
    parent: Dict[str, Optional[str]] = {}
    arts: set = set()
    tick = 0

    # Iterative DFS avoids recursion-depth failures on large zones.
    for root in sorted(node_set):
        if root in disc:
            continue
        parent[root] = None
        disc[root] = low[root] = tick; tick += 1
        child_count: Dict[str, int] = defaultdict(int)
        stack: List[Tuple[str, Iterator[str]]] = [
            (root, iter(v for v in adjacency.get(root, ()) if v in node_set))
        ]
        while stack:
            u, it = stack[-1]
            try:
                v = next(it)
            except StopIteration:
                stack.pop()
                p = parent.get(u)
                if p is not None:
                    low[p] = min(low[p], low[u])
                    if parent.get(p) is not None and low[u] >= disc[p]:
                        arts.add(p)
                else:
                    if child_count.get(u, 0) > 1:
                        arts.add(u)
                continue
            if v not in disc:
                parent[v] = u
                child_count[u] += 1
                disc[v] = low[v] = tick; tick += 1
                stack.append((v, iter(w for w in adjacency.get(v, ()) if w in node_set)))
            elif v != parent.get(u):
                low[u] = min(low[u], disc[v])
    return arts


def _partition_structure(
    part: "Partition",
    adjacency: Mapping[str, Sequence[str]],
    maxsize: int = 8,
) -> Dict[str, Any]:
    """Compute groups + boundary triples + neighboring-zone pairs in one edge scan."""
    key = (id(adjacency), part.signature())
    cached = _PARTITION_STRUCTURE_CACHE.get(key)
    if cached is not None:
        _PARTITION_STRUCTURE_CACHE.move_to_end(key)
        return cached
    groups = part.zones()
    bset = set()
    pset = set()
    # One traversal of directed adjacency; deterministic sets are sorted once.
    for u, ku in part.zone_of.items():
        for v in adjacency.get(u, ()):
            kv = part.zone_of.get(v)
            if kv is None or kv == ku:
                continue
            bset.add((u, int(ku), int(kv)))
            pset.add(tuple(sorted((int(ku), int(kv)))))
    out = {
        "groups": groups,
        "boundary": sorted(bset),
        "neighbor_pairs": sorted(pset),
    }
    _PARTITION_STRUCTURE_CACHE[key] = out
    _PARTITION_STRUCTURE_CACHE.move_to_end(key)
    while len(_PARTITION_STRUCTURE_CACHE) > max(1, int(maxsize)):
        _PARTITION_STRUCTURE_CACHE.popitem(last=False)
    return out


def _bounded_put(cache: OrderedDict, key: Any, value: Any, maxsize: int = _FAST_CACHE_MAX) -> None:
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > maxsize:
        cache.popitem(last=False)


def _year_arrays(year_df: pd.DataFrame, cfg: Optional[Config] = None) -> Dict[str, Any]:
    key = id(year_df)
    cached = _YEAR_ARRAY_CACHE.get(key)
    if cached is not None:
        return cached
    codes = year_df["code_commune"].astype(str).to_numpy()
    exposure = finite_array(year_df["exposure_proxy"].to_numpy(float), fill=0.0)
    mhat = finite_array(year_df["mhat"].to_numpy(float), fill=0.0)
    total_e = float(exposure.sum())
    avg_m = float(np.dot(exposure, mhat) / max(total_e, 1e-12))
    local_rel = mhat / max(avg_m, 1e-12)
    qcols = [c for c in year_df.columns if str(c).startswith("qhat_")]
    if qcols:
        # Explicitly take the numerically largest quantile, not the last column by chance.
        qcol = max(qcols, key=lambda c: float(str(c).split("_")[-1]))
        q99 = finite_array(year_df[qcol].to_numpy(float), fill=0.0)
    else:
        q99 = mhat.copy()
    out = {
        "codes": codes,
        "exposure": exposure,
        "mhat": mhat,
        "q99": q99,
        "total_e": total_e,
        "avg_m": avg_m,
        "local_rel": local_rel,
        "code_to_index": {c: i for i, c in enumerate(codes)},
    }
    _YEAR_ARRAY_CACHE[key] = out
    return out


def _zone_assignment(part: Partition, year_df: pd.DataFrame) -> np.ndarray:
    key = (id(year_df), part.signature())
    cached = _ASSIGNMENT_CACHE.get(key)
    if cached is not None:
        return cached
    ctx = _year_arrays(year_df)
    z = np.fromiter((int(part.zone_of.get(c, -1)) for c in ctx["codes"]), dtype=np.int16, count=len(ctx["codes"]))
    _bounded_put(_ASSIGNMENT_CACHE, key, z)
    return z


def partition_admissible(
    part: Partition,
    year_df: pd.DataFrame,
    adjacency: Mapping[str, Sequence[str]],
    cfg: Config,
) -> bool:
    key = (id(year_df), part.signature())
    cached = _ADMISSIBLE_CACHE.get(key)
    if cached is not None:
        return cached
    groups = part.zones()
    if not (1 <= len(groups) <= cfg.k_max):
        _bounded_put(_ADMISSIBLE_CACHE, key, False)
        return False
    ctx = _year_arrays(year_df)
    z = _zone_assignment(part, year_df)
    valid = z >= 0
    kmax = int(z[valid].max()) + 1 if valid.any() else 0
    zone_e = np.bincount(z[valid].astype(np.int64), weights=ctx["exposure"][valid], minlength=kmax)
    if zone_e.size and np.any(zone_e[:len(groups)] < cfg.e_min):
        _bounded_put(_ADMISSIBLE_CACHE, key, False)
        return False
    for nodes in groups.values():
        if not is_connected_nodes(nodes, adjacency):
            _ADMISSIBLE_CACHE[key] = False
            return False
    _bounded_put(_ADMISSIBLE_CACHE, key, True)
    return True


def zone_statistics(part: Partition, year_df: pd.DataFrame) -> pd.DataFrame:
    key = (id(year_df), part.signature())
    cached = _ZONE_STATS_CACHE.get(key)
    if cached is not None:
        return cached.copy()
    ctx = _year_arrays(year_df)
    z = _zone_assignment(part, year_df)
    valid = z >= 0
    if not valid.any():
        return pd.DataFrame(columns=["zone","exposure","weight","mhat_zone","risk_sd","n_communes","q99_zone","indicated_relativity"])
    zi = z[valid].astype(np.int64)
    k = int(zi.max()) + 1
    e = ctx["exposure"][valid]
    m = ctx["mhat"][valid]
    q = ctx["q99"][valid]
    e_sum = np.bincount(zi, weights=e, minlength=k)
    em_sum = np.bincount(zi, weights=e*m, minlength=k)
    em2_sum = np.bincount(zi, weights=e*m*m, minlength=k)
    eq_sum = np.bincount(zi, weights=e*q, minlength=k)
    n = np.bincount(zi, minlength=k)
    pooled = em_sum / np.maximum(e_sum, 1e-12)
    var = np.maximum(em2_sum / np.maximum(e_sum, 1e-12) - pooled**2, 0.0)
    total_e = max(float(e_sum.sum()), 1e-12)
    avg = float(em_sum.sum() / total_e)
    out = pd.DataFrame({
        "zone": np.arange(k, dtype=int),
        "exposure": e_sum,
        "weight": e_sum / total_e,
        "mhat_zone": pooled,
        "risk_sd": np.sqrt(var),
        "n_communes": n.astype(int),
        "q99_zone": eq_sum / np.maximum(e_sum, 1e-12),
        "indicated_relativity": pooled / max(avg, 1e-12),
    })
    out = out[out["n_communes"] > 0].reset_index(drop=True)
    _bounded_put(_ZONE_STATS_CACHE, key, out.copy())
    return out


def location_relativities(year_df: pd.DataFrame) -> Dict[str, float]:
    ctx = _year_arrays(year_df)
    return dict(zip(ctx["codes"], ctx["local_rel"]))


def psi_z(part: Partition, year_df: pd.DataFrame) -> float:
    key = (id(year_df), part.signature())
    cached = _PSI_Z_CACHE.get(key)
    if cached is not None:
        return cached
    ctx = _year_arrays(year_df)
    z = _zone_assignment(part, year_df)
    valid = z >= 0
    if not valid.any():
        return 0.0
    zi = z[valid].astype(np.int64)
    e = ctx["exposure"][valid]
    r = ctx["local_rel"][valid]
    k = int(zi.max()) + 1
    e_sum = np.bincount(zi, weights=e, minlength=k)
    er_sum = np.bincount(zi, weights=e*r, minlength=k)
    er2_sum = np.bincount(zi, weights=e*r*r, minlength=k)
    within = er2_sum - np.square(er_sum) / np.maximum(e_sum, 1e-12)
    val = float(np.maximum(within, 0.0).sum() / max(ctx["total_e"], 1e-12))
    _bounded_put(_PSI_Z_CACHE, key, val)
    return val


def indicated_relativities(part: Partition, year_df: pd.DataFrame) -> Dict[int, float]:
    zs = zone_statistics(part, year_df)
    return dict(zip(zs.zone.astype(int), zs.indicated_relativity.astype(float)))


def pricing_normalize(rel: Dict[int, float], weights: Dict[int, float]) -> Dict[int, float]:
    den = sum(weights[k] * rel[k] for k in rel)
    if den <= 0:
        return rel
    return {k: float(rel[k] / den) for k in rel}


def pricing_candidates(part: Partition, year_df: pd.DataFrame, cfg: Config) -> List[Dict[int, float]]:
    zs = zone_statistics(part, year_df)
    indicated = dict(zip(zs.zone.astype(int), zs.indicated_relativity.astype(float)))
    weights = dict(zip(zs.zone.astype(int), zs.weight.astype(float)))
    out = [indicated]
    if len(indicated) <= 1 or cfg.m_pricing_per_partition <= 1:
        return out
    zone_risk = dict(zip(zs.zone.astype(int), zs.q99_zone.astype(float)))
    arr = np.array([zone_risk[k] for k in sorted(zone_risk)])
    med = float(np.median(arr))
    direction = {k: (1.0 if zone_risk[k] >= med else -1.0) for k in indicated}
    levels = [cfg.h, -cfg.h, 2 * cfg.h, -2 * cfg.h]
    for level in levels[: max(0, cfg.m_pricing_per_partition - 1)]:
        cand = {k: indicated[k] * math.exp(level * direction[k]) for k in indicated}
        cand = pricing_normalize(cand, weights)
        ok = all(abs(math.log(max(cand[k], 1e-12) / max(indicated[k], 1e-12))) <= cfg.delta_r + 1e-9 for k in indicated)
        if ok:
            out.append(cand)
    return out[: cfg.m_pricing_per_partition]


def boundary_nodes(part: Partition, adjacency: Mapping[str, Sequence[str]]) -> List[Tuple[str, int, int]]:
    out = []
    for u, ku in part.zone_of.items():
        for v in adjacency.get(u, ()):
            kv = part.zone_of.get(v)
            if kv is not None and kv != ku:
                out.append((u, ku, kv))
    # deterministic unique
    return sorted(set(out))


def split_groups_two_seed(
    zone: int,
    nodes: Sequence[str],
    risk: Mapping[str, float],
    adjacency: Mapping[str, Sequence[str]],
) -> Optional[Tuple[List[str], List[str]]]:
    """V10 exact split geometry without materializing a full Partition.

    This reproduces the V9 two-seed rule exactly and returns the connected
    low/high children only. The national partition is built only if the action
    is actually executed.
    """
    if len(nodes) < 4:
        return None
    ordered = sorted(nodes, key=lambda n: (risk.get(n, 0.0), n))
    seed_lo, seed_hi = ordered[0], ordered[-1]
    if seed_lo == seed_hi:
        return None
    node_set = set(nodes)
    dist_lo = bfs_distances(seed_lo, node_set, adjacency)
    dist_hi = bfs_distances(seed_hi, node_set, adjacency)
    median_risk = float(np.median([risk.get(n, 0.0) for n in nodes]))
    g_lo: List[str] = []
    g_hi: List[str] = []
    for n in nodes:
        dl = dist_lo.get(n, 10**9)
        dh = dist_hi.get(n, 10**9)
        if dl < dh:
            g_lo.append(n)
        elif dh < dl:
            g_hi.append(n)
        else:
            (g_hi if risk.get(n, 0.0) >= median_risk else g_lo).append(n)
    if not g_lo or not g_hi:
        return None
    if not is_connected_nodes(g_lo, adjacency) or not is_connected_nodes(g_hi, adjacency):
        return None
    return g_lo, g_hi


def split_partition_two_seed(
    part: Partition,
    zone: int,
    nodes: Sequence[str],
    risk: Mapping[str, float],
    adjacency: Mapping[str, Sequence[str]],
) -> Optional[Partition]:
    if len(nodes) < 4:
        return None
    ordered = sorted(nodes, key=lambda n: (risk.get(n, 0.0), n))
    seed_lo, seed_hi = ordered[0], ordered[-1]
    if seed_lo == seed_hi:
        return None
    node_set = set(nodes)
    # Multi-source BFS assigning each node to closest low/high seed. Ties by risk relative to median.
    dist_lo = bfs_distances(seed_lo, node_set, adjacency)
    dist_hi = bfs_distances(seed_hi, node_set, adjacency)
    median_risk = float(np.median([risk.get(n, 0.0) for n in nodes]))
    g_lo, g_hi = [], []
    for n in nodes:
        dl = dist_lo.get(n, 10**9)
        dh = dist_hi.get(n, 10**9)
        if dl < dh:
            g_lo.append(n)
        elif dh < dl:
            g_hi.append(n)
        else:
            (g_hi if risk.get(n, 0.0) >= median_risk else g_lo).append(n)
    if not g_lo or not g_hi:
        return None
    if not is_connected_nodes(g_lo, adjacency) or not is_connected_nodes(g_hi, adjacency):
        return None
    new = part.copy()
    next_zone = max(part.zone_of.values()) + 1
    for n in g_hi:
        new.zone_of[n] = next_zone
    return new.canonicalize()


def bfs_distances(seed: str, allowed: set, adjacency: Mapping[str, Sequence[str]]) -> Dict[str, int]:
    dist = {seed: 0}
    q = deque([seed])
    while q:
        u = q.popleft()
        for v in adjacency.get(u, ()):
            if v in allowed and v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def merge_partition(part: Partition, ka: int, kb: int) -> Partition:
    new = part.copy()
    for n, k in list(new.zone_of.items()):
        if k == kb:
            new.zone_of[n] = ka
    return new.canonicalize()


def shift_partition(part: Partition, node: str, from_zone: int, to_zone: int) -> Partition:
    new = part.copy()
    if new.zone_of.get(node) == from_zone:
        new.zone_of[node] = to_zone
    return new.canonicalize()


def zone_neighbor_pairs(part: Partition, adjacency: Mapping[str, Sequence[str]]) -> List[Tuple[int, int]]:
    pairs = set()
    for u, ku in part.zone_of.items():
        for v in adjacency.get(u, ()):
            kv = part.zone_of.get(v)
            if kv is not None and kv != ku:
                pairs.add(tuple(sorted((ku, kv))))
    return sorted(pairs)



def _zone_moments_for_screening(part: Partition, year_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Exact zonal sufficient statistics used by V8 geographic screening.

    The quantities below are sufficient to update Psi^Z exactly after a MERGE or
    one-commune SHIFT, without materializing a 34k-entry Partition or rescanning
    the whole portfolio.
    """
    key = (id(year_df), part.signature())
    cached = _ZONE_MOMENT_CACHE.get(key)
    if cached is not None:
        _ZONE_MOMENT_CACHE.move_to_end(key)
        return cached
    ctx = _year_arrays(year_df)
    z = _zone_assignment(part, year_df).astype(np.int64)
    valid = z >= 0
    if not valid.any():
        out = {
            "e": np.zeros(0, dtype=float),
            "er": np.zeros(0, dtype=float),
            "er2": np.zeros(0, dtype=float),
            "within": np.zeros(0, dtype=float),
        }
    else:
        zi = z[valid]
        e = ctx["exposure"][valid]
        r = ctx["local_rel"][valid]
        k = int(zi.max()) + 1
        es = np.bincount(zi, weights=e, minlength=k)
        ers = np.bincount(zi, weights=e*r, minlength=k)
        er2s = np.bincount(zi, weights=e*r*r, minlength=k)
        within = er2s - np.square(ers) / np.maximum(es, 1e-12)
        out = {"e": es, "er": ers, "er2": er2s, "within": np.maximum(within, 0.0)}
    _bounded_put(_ZONE_MOMENT_CACHE, key, out)
    return out


def _merge_delta_psi_exact(
    ka: int,
    kb: int,
    moments: Mapping[str, np.ndarray],
    total_e: float,
) -> float:
    """Return base Psi^Z - merged Psi^Z exactly from sufficient statistics."""
    e = moments["e"]; er = moments["er"]; er2 = moments["er2"]; within = moments["within"]
    if ka >= len(e) or kb >= len(e):
        return float("-inf")
    em = float(e[ka] + e[kb])
    if em <= 0:
        return float("-inf")
    merged_within = float(er2[ka] + er2[kb] - (er[ka] + er[kb])**2 / max(em, 1e-12))
    old_within = float(within[ka] + within[kb])
    return float((old_within - max(merged_within, 0.0)) / max(total_e, 1e-12))


def _shift_delta_psi_exact(
    node_i: int,
    k_from: int,
    k_to: int,
    moments: Mapping[str, np.ndarray],
    ctx: Mapping[str, Any],
) -> float:
    """Return base Psi^Z - shifted Psi^Z exactly for a one-unit SHIFT."""
    e = moments["e"]; er = moments["er"]; er2 = moments["er2"]; within = moments["within"]
    if k_from >= len(e) or k_to >= len(e):
        return float("-inf")
    en = float(ctx["exposure"][node_i])
    rn = float(ctx["local_rel"][node_i])
    ern = en * rn
    er2n = en * rn * rn
    ef = float(e[k_from] - en)
    et = float(e[k_to] + en)
    if ef <= 0 or et <= 0:
        return float("-inf")
    wf = float((er2[k_from] - er2n) - (er[k_from] - ern)**2 / max(ef, 1e-12))
    wt = float((er2[k_to] + er2n) - (er[k_to] + ern)**2 / max(et, 1e-12))
    old = float(within[k_from] + within[k_to])
    new = max(wf, 0.0) + max(wt, 0.0)
    return float((old - new) / max(float(ctx["total_e"]), 1e-12))


def _materialize_deferred_edit(
    part: Partition,
    descriptor: Tuple[Any, ...],
) -> Partition:
    kind = descriptor[0]
    if kind == "MERGE":
        return merge_partition(part, int(descriptor[1]), int(descriptor[2]))
    if kind == "SHIFT":
        return shift_partition(part, str(descriptor[1]), int(descriptor[2]), int(descriptor[3]))
    if kind == "SPLIT":
        zone = int(descriptor[1])
        g_hi = tuple(str(x) for x in descriptor[2])
        new = part.copy()
        next_zone = max(part.zone_of.values()) + 1
        for n in g_hi:
            if new.zone_of.get(n) != zone:
                raise RuntimeError(f"Invalid lazy SPLIT descriptor for unit={n}: {descriptor[:2]}")
            new.zone_of[n] = next_zone
        return new.canonicalize()
    raise ValueError(f"Unknown deferred edit descriptor: {descriptor}")



def _pricing_candidates_from_stats(zs: pd.DataFrame, cfg: Config) -> List[Dict[int, float]]:
    """Exact V9 pricing candidates from zonal sufficient statistics."""
    if zs.empty:
        return []
    indicated = dict(zip(zs.zone.astype(int), zs.indicated_relativity.astype(float)))
    weights = dict(zip(zs.zone.astype(int), zs.weight.astype(float)))
    out = [indicated]
    if len(indicated) <= 1 or cfg.m_pricing_per_partition <= 1:
        return out
    zone_risk = dict(zip(zs.zone.astype(int), zs.q99_zone.astype(float)))
    arr = np.asarray([zone_risk[k] for k in sorted(zone_risk)], dtype=float)
    med = float(np.median(arr))
    direction = {k: (1.0 if zone_risk[k] >= med else -1.0) for k in indicated}
    levels = [cfg.h, -cfg.h, 2 * cfg.h, -2 * cfg.h]
    for level in levels[: max(0, cfg.m_pricing_per_partition - 1)]:
        cand = {k: indicated[k] * math.exp(level * direction[k]) for k in indicated}
        cand = pricing_normalize(cand, weights)
        ok = all(
            abs(math.log(max(cand[k], 1e-12) / max(indicated[k], 1e-12)))
            <= cfg.delta_r + 1e-9
            for k in indicated
        )
        if ok:
            out.append(cand)
    return out[: cfg.m_pricing_per_partition]


def _lazy_canonical_mapping(
    part: Partition,
    descriptor: Tuple[Any, ...],
    groups: Mapping[int, Sequence[str]],
) -> Dict[int, int]:
    """Map old zone identities to canonical new labels without materializing the partition."""
    kind = descriptor[0]
    mins: Dict[int, str] = {int(k): min(map(str, v)) for k, v in groups.items() if v}
    identities = set(mins)
    if kind == "MERGE":
        ka, kb = int(descriptor[1]), int(descriptor[2])
        if ka not in mins or kb not in mins:
            raise RuntimeError(f"Invalid lazy MERGE descriptor: {descriptor}")
        mins[ka] = min(mins[ka], mins[kb])
        mins.pop(kb, None)
        identities.discard(kb)
    elif kind == "SHIFT":
        node, kf, kt = str(descriptor[1]), int(descriptor[2]), int(descriptor[3])
        src = [str(n) for n in groups.get(kf, ()) if str(n) != node]
        dst = [str(n) for n in groups.get(kt, ())]
        if not src:
            raise RuntimeError(f"Invalid lazy SHIFT leaves empty source: {descriptor}")
        mins[kf] = min(src)
        mins[kt] = min(dst + [node])
    elif kind == "SPLIT":
        kf = int(descriptor[1])
        g_hi = {str(n) for n in descriptor[2]}
        src = [str(n) for n in groups.get(kf, ()) if str(n) not in g_hi]
        dst = sorted(g_hi)
        if not src or not dst:
            raise RuntimeError(f"Invalid lazy SPLIT descriptor: {descriptor[:2]}")
        new_identity = (max(identities) + 1) if identities else 0
        identities.add(new_identity)
        mins[kf] = min(src)
        mins[new_identity] = min(dst)
    else:
        raise ValueError(f"Unsupported lazy descriptor: {descriptor}")
    ordered = sorted(identities, key=lambda k: mins[int(k)])
    return {int(old): int(new) for new, old in enumerate(ordered)}


def _lazy_zone_stats_from_descriptor(
    part: Partition,
    year_df: pd.DataFrame,
    descriptor: Tuple[Any, ...],
    groups: Mapping[int, Sequence[str]],
) -> Tuple[pd.DataFrame, Dict[int, int]]:
    """Exact post-edit zonal statistics without a 34k-entry Partition."""
    base = zone_statistics(part, year_df).set_index("zone", drop=False)
    ctx = _year_arrays(year_df)
    mapping = _lazy_canonical_mapping(part, descriptor, groups)

    # sufficient statistics indexed by old zone identity
    stats: Dict[int, Dict[str, float]] = {}
    for k, row in base.iterrows():
        e = float(row["exposure"])
        mu = float(row["mhat_zone"])
        sd = float(row["risk_sd"])
        stats[int(k)] = {
            "e": e,
            "em": e * mu,
            "em2": e * (sd * sd + mu * mu),
            "eq": e * float(row["q99_zone"]),
            "n": float(row["n_communes"]),
        }

    kind = descriptor[0]
    if kind == "MERGE":
        ka, kb = int(descriptor[1]), int(descriptor[2])
        for key in ("e", "em", "em2", "eq", "n"):
            stats[ka][key] += stats[kb][key]
        stats.pop(kb, None)
    elif kind == "SHIFT":
        node, kf, kt = str(descriptor[1]), int(descriptor[2]), int(descriptor[3])
        ii = ctx["code_to_index"].get(node)
        if ii is None:
            raise RuntimeError(f"Lazy SHIFT node missing from year context: {node}")
        en = float(ctx["exposure"][ii])
        mn = float(ctx["mhat"][ii])
        qn = float(ctx["q99"][ii])
        delta = {"e": en, "em": en * mn, "em2": en * mn * mn, "eq": en * qn, "n": 1.0}
        for key, val in delta.items():
            stats[kf][key] -= val
            stats[kt][key] += val
    elif kind == "SPLIT":
        kf = int(descriptor[1])
        g_hi = tuple(str(n) for n in descriptor[2])
        new_identity = max(mapping.keys())
        child = {"e": 0.0, "em": 0.0, "em2": 0.0, "eq": 0.0, "n": 0.0}
        for node in g_hi:
            ii = ctx["code_to_index"].get(node)
            if ii is None:
                continue
            en = float(ctx["exposure"][ii]); mn = float(ctx["mhat"][ii]); qn = float(ctx["q99"][ii])
            child["e"] += en; child["em"] += en * mn; child["em2"] += en * mn * mn; child["eq"] += en * qn; child["n"] += 1.0
        stats[new_identity] = dict(child)
        for key, val in child.items():
            stats[kf][key] -= val
    else:
        raise ValueError(f"Unsupported lazy descriptor: {descriptor}")

    total_e = max(sum(v["e"] for v in stats.values()), 1e-12)
    total_em = sum(v["em"] for v in stats.values())
    avg = total_em / total_e
    rows = []
    for oldk, v in stats.items():
        e = max(float(v["e"]), 1e-12)
        pooled = float(v["em"]) / e
        var = max(float(v["em2"]) / e - pooled * pooled, 0.0)
        rows.append({
            "zone": mapping[int(oldk)],
            "exposure": float(v["e"]),
            "weight": float(v["e"]) / total_e,
            "mhat_zone": pooled,
            "risk_sd": math.sqrt(var),
            "n_communes": int(round(v["n"])),
            "q99_zone": float(v["eq"]) / e,
            "indicated_relativity": pooled / max(avg, 1e-12),
        })
    out = pd.DataFrame(rows).sort_values("zone").reset_index(drop=True)
    return out, mapping


def _uncertainty_from_stats(zs: pd.DataFrame) -> float:
    if zs.empty:
        return 0.0
    kappa = 1.0 / np.sqrt(np.maximum(zs["exposure"].to_numpy(float), 1.0))
    mx = float(np.max(kappa)) if kappa.size else 0.0
    if mx > 0:
        kappa = kappa / mx
    return float(np.dot(zs["weight"].to_numpy(float), kappa))


def _psi_p_from_stats(zs: pd.DataFrame, rel: Mapping[int, float]) -> float:
    if zs.empty:
        return 0.0
    zones = zs["zone"].to_numpy(int)
    indicated = zs["indicated_relativity"].to_numpy(float)
    weights = zs["weight"].to_numpy(float)
    rr = np.asarray([float(rel.get(int(k), indicated[j])) for j, k in enumerate(zones)], dtype=float)
    return float(np.dot(weights, np.square(rr - indicated)))


def _lazy_turnover_exact(
    descriptor: Tuple[Any, ...],
    part: Partition,
    year_df: pd.DataFrame,
    zone_exposure: Mapping[int, float],
) -> float:
    ctx = _year_arrays(year_df)
    evec = ctx["exposure"]
    W = 0.5 * (float(evec.sum()) ** 2 - float(np.dot(evec, evec)))
    if W <= 0:
        return 0.0
    kind = descriptor[0]
    if kind == "MERGE":
        ka, kb = int(descriptor[1]), int(descriptor[2])
        disagree = float(zone_exposure.get(ka, 0.0)) * float(zone_exposure.get(kb, 0.0))
    elif kind == "SHIFT":
        node, kf, kt = str(descriptor[1]), int(descriptor[2]), int(descriptor[3])
        ii = ctx["code_to_index"].get(node)
        if ii is None:
            return 0.0
        en = float(evec[ii])
        disagree = en * (
            max(float(zone_exposure.get(kf, 0.0)) - en, 0.0)
            + float(zone_exposure.get(kt, 0.0))
        )
    elif kind == "SPLIT":
        kf = int(descriptor[1])
        e_hi = 0.0
        for node in descriptor[2]:
            ii = ctx["code_to_index"].get(str(node))
            if ii is not None:
                e_hi += float(evec[ii])
        e_lo = max(float(zone_exposure.get(kf, 0.0)) - e_hi, 0.0)
        disagree = e_lo * e_hi
    else:
        return 0.0
    return float(np.clip(disagree / W, 0.0, 1.0))


def _lazy_premium_adjustment_exact(
    descriptor: Tuple[Any, ...],
    rel: Mapping[int, float],
    prev_rel: Mapping[int, float],
    mapping: Mapping[int, int],
    year_df: pd.DataFrame,
    zone_exposure: Mapping[int, float],
) -> float:
    ctx = _year_arrays(year_df)
    total_e = max(float(ctx["total_e"]), 1e-12)
    kind = descriptor[0]
    num = 0.0

    def term(oldk: int, newk: int, mass: float) -> float:
        if mass <= 0:
            return 0.0
        ro = max(float(prev_rel.get(int(oldk), 1.0)), 1e-12)
        rn = max(float(rel.get(int(newk), 1.0)), 1e-12)
        return float(mass) * abs(math.log(rn / ro))

    if kind == "MERGE":
        ka, kb = int(descriptor[1]), int(descriptor[2])
        for oldk, mass in zone_exposure.items():
            identity = ka if int(oldk) == kb else int(oldk)
            num += term(int(oldk), int(mapping[identity]), float(mass))
    elif kind == "SHIFT":
        node, kf, kt = str(descriptor[1]), int(descriptor[2]), int(descriptor[3])
        ii = ctx["code_to_index"].get(node)
        if ii is None:
            return 0.0
        en = float(ctx["exposure"][ii])
        for oldk, mass in zone_exposure.items():
            oldk = int(oldk)
            if oldk == kf:
                num += term(kf, int(mapping[kf]), max(float(mass) - en, 0.0))
                num += term(kf, int(mapping[kt]), en)
            else:
                num += term(oldk, int(mapping[oldk]), float(mass))
    elif kind == "SPLIT":
        kf = int(descriptor[1])
        new_identity = max(mapping.keys())
        e_hi = 0.0
        for node in descriptor[2]:
            ii = ctx["code_to_index"].get(str(node))
            if ii is not None:
                e_hi += float(ctx["exposure"][ii])
        for oldk, mass in zone_exposure.items():
            oldk = int(oldk)
            if oldk == kf:
                num += term(kf, int(mapping[kf]), max(float(mass) - e_hi, 0.0))
                num += term(kf, int(mapping[new_identity]), e_hi)
            else:
                num += term(oldk, int(mapping[oldk]), float(mass))
    return float(num / total_e)


def _lazy_action_metrics(
    descriptor: Tuple[Any, ...],
    zs_new: pd.DataFrame,
    mapping: Mapping[int, int],
    rel: Mapping[int, float],
    prev_rel: Mapping[int, float],
    part: Partition,
    year_df: pd.DataFrame,
    zone_exposure: Mapping[int, float],
    psi_z_new: float,
) -> Dict[str, float]:
    return {
        "psi_z": float(psi_z_new),
        "psi_p": _psi_p_from_stats(zs_new, rel),
        "c_zonal": _uncertainty_from_stats(zs_new),
        "gamma_p": _lazy_premium_adjustment_exact(
            descriptor, rel, prev_rel, mapping, year_df, zone_exposure
        ),
        "z_delta": _lazy_turnover_exact(descriptor, part, year_df, zone_exposure),
    }


def _materialize_action_if_needed(action: Action, current_part: Partition) -> Partition:
    if action.partition is not None:
        return action.partition
    if action.lazy_descriptor is None:
        raise RuntimeError(f"Lazy action {action.action_id} has neither partition nor descriptor")
    part = _materialize_deferred_edit(current_part, action.lazy_descriptor)
    action.partition = part
    return part


def generate_geographic_candidates(
    part: Partition,
    year_df: pd.DataFrame,
    adjacency: Mapping[str, Sequence[str]],
    cfg: Config,
) -> List[Tuple[str, Optional[Partition], Dict[str, Any]]]:
    """V9 geographic candidate engine.

    Candidate definitions and admissibility are unchanged relative to V8.  V9
    changes *when* expensive objects are built: MERGE and SHIFT candidates are
    screened with exact sufficient-statistics formulas and are materialized only
    after the top-M geographic shortlist has been selected.
    """
    t_all = time.perf_counter()
    prof: Dict[str, float] = {}
    ctx = _year_arrays(year_df, cfg)
    codes = ctx["codes"]
    risk_arr = ctx["mhat"]
    q99_arr = ctx["q99"]
    exp_arr = ctx["exposure"]
    idxmap = ctx["code_to_index"]
    risk = dict(zip(codes, risk_arr))
    q99_map = dict(zip(codes, q99_arr))
    base_psi = psi_z(part, year_df)

    # Each entry: (score, action_id, materialized_partition_or_None, meta, descriptor_or_None)
    candidates: List[Tuple[float, str, Optional[Partition], Dict[str, Any], Optional[Tuple[Any, ...]]]] = []

    t0 = time.perf_counter()
    st = _partition_structure(part, adjacency, getattr(cfg, "structure_cache_size", 8))
    groups: Dict[int, List[str]] = st["groups"]
    boundary: List[Tuple[str, int, int]] = st["boundary"]
    neighbor_pairs: List[Tuple[int, int]] = st["neighbor_pairs"]
    prof["topology"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    zs = zone_statistics(part, year_df)
    zstats = zs.set_index("zone")
    zone_exposure = dict(zip(zs.zone.astype(int), zs.exposure.astype(float)))
    moments = _zone_moments_for_screening(part, year_df)
    prof["moments"] = time.perf_counter() - t0

    if partition_admissible(part, year_df, adjacency, cfg):
        candidates.append((0.0, "KEEP", part, {}, None))

    # SPLIT V10: exact graph cut + sufficient-statistics scoring, no national Partition copy.
    t0 = time.perf_counter()
    if part.k < cfg.k_max:
        split_targets = (
            zs.sort_values("risk_sd", ascending=False).zone.astype(int).tolist()
            [: max(1, int(cfg.m_split_target_zones))]
        )
        local_rel_arr = ctx["local_rel"]
        total_e = max(float(ctx["total_e"]), 1e-12)
        for zone in split_targets:
            nodes = groups.get(zone, [])
            if len(nodes) < 4:
                continue
            parent_idx = np.asarray([idxmap[n] for n in nodes if n in idxmap], dtype=np.int64)
            if parent_idx.size < 4:
                continue
            pe = exp_arr[parent_idx]; pr = local_rel_arr[parent_idx]
            p_er = float(np.dot(pe, pr)); p_er2 = float(np.dot(pe, pr * pr)); p_e = float(pe.sum())
            old_within = max(p_er2 - p_er * p_er / max(p_e, 1e-12), 0.0)
            for variant in range(cfg.m_split_per_zone):
                rmap = risk if variant == 0 else q99_map
                split = split_groups_two_seed(zone, nodes, rmap, adjacency)
                if split is None:
                    continue
                g_lo, g_hi = split
                hi_idx = np.asarray([idxmap[n] for n in g_hi if n in idxmap], dtype=np.int64)
                if hi_idx.size == 0 or hi_idx.size >= parent_idx.size:
                    continue
                he = exp_arr[hi_idx]; hr = local_rel_arr[hi_idx]
                e_hi = float(he.sum()); er_hi = float(np.dot(he, hr)); er2_hi = float(np.dot(he, hr * hr))
                e_lo = p_e - e_hi
                if min(e_lo, e_hi) < cfg.e_min:
                    continue
                er_lo = p_er - er_hi; er2_lo = p_er2 - er2_hi
                w_hi = max(er2_hi - er_hi * er_hi / max(e_hi, 1e-12), 0.0)
                w_lo = max(er2_lo - er_lo * er_lo / max(e_lo, 1e-12), 0.0)
                dpsi = float((old_within - w_hi - w_lo) / total_e)
                aid = f"SPLIT|zone={zone}|variant={variant}"
                candidates.append((
                    dpsi, aid, None,
                    {"zone": zone, "variant": variant, "delta_psi_z": dpsi},
                    ("SPLIT", int(zone), tuple(g_hi)),
                ))
    prof["split"] = time.perf_counter() - t0

    # MERGE: exact Delta Psi from zonal moments; materialize only if selected.
    t0 = time.perf_counter()
    merges = []
    for ka, kb in neighbor_pairs:
        if ka in zstats.index and kb in zstats.index:
            contrast = abs(
                float(zstats.loc[ka, "indicated_relativity"])
                - float(zstats.loc[kb, "indicated_relativity"])
            )
        else:
            contrast = 999.0
        merges.append((contrast, ka, kb))
    for contrast, ka, kb in sorted(merges)[: cfg.m_merge_total]:
        dpsi = _merge_delta_psi_exact(ka, kb, moments, float(ctx["total_e"]))
        aid = f"MERGE|zone_a={ka}|zone_b={kb}"
        candidates.append((
            dpsi - cfg.screen_weight_boundary_contrast * contrast,
            aid, None,
            {"zone_a": ka, "zone_b": kb, "contrast": contrast, "delta_psi_z": dpsi},
            ("MERGE", ka, kb),
        ))
    prof["merge"] = time.perf_counter() - t0

    # SHIFT: exact Delta Psi + articulation-point feasibility, no Partition copy yet.
    t0 = time.perf_counter()
    indicated = dict(zip(zs.zone.astype(int), zs.indicated_relativity.astype(float)))
    local_rel = ctx["local_rel"]
    shifts: List[Tuple[float, str, int, int, int]] = []
    for node, k_from, k_to in boundary:
        ii = idxmap.get(node)
        if ii is None:
            continue
        rr = float(local_rel[ii])
        before = (rr - indicated.get(k_from, 1.0)) ** 2
        after = (rr - indicated.get(k_to, 1.0)) ** 2
        shifts.append((before - after, node, k_from, k_to, ii))

    shortlist = sorted(shifts, reverse=True)[: max(int(cfg.m_shift_total) * 2, int(cfg.m_shift_total))]
    source_zones = sorted({k_from for _, _, k_from, _, _ in shortlist})
    articulation: Dict[int, set] = {
        k: _articulation_points_subset(groups.get(k, []), adjacency) for k in source_zones
    }

    accepted = 0
    for gain, node, k_from, k_to, ii in shortlist:
        if accepted >= cfg.m_shift_total:
            break
        src_nodes = groups.get(k_from, [])
        if len(src_nodes) <= 1 or node in articulation.get(k_from, set()):
            continue
        node_e = float(exp_arr[ii])
        src_e = float(zone_exposure.get(k_from, 0.0)) - node_e
        dst_e = float(zone_exposure.get(k_to, 0.0)) + node_e
        if src_e < cfg.e_min or dst_e < cfg.e_min:
            continue
        dpsi = _shift_delta_psi_exact(ii, k_from, k_to, moments, ctx)
        aid = f"SHIFT|unit={node}|from={k_from}|to={k_to}"
        candidates.append((
            dpsi + gain, aid, None,
            {"unit": node, "from": k_from, "to": k_to,
             "local_gain": gain, "delta_psi_z": dpsi},
            ("SHIFT", node, k_from, k_to),
        ))
        accepted += 1
    prof["shift"] = time.perf_counter() - t0

    # Select before full materialization. This is the main V8 complexity reduction.
    t0 = time.perf_counter()
    keep = [x for x in candidates if x[1] == "KEEP"]
    rest = [x for x in candidates if x[1] != "KEEP"]
    rest.sort(key=lambda x: (x[0], x[1]), reverse=True)
    selected = keep[:1] + rest[: max(0, cfg.m_geographic_total - len(keep[:1]))]

    out: List[Tuple[str, Optional[Partition], Dict[str, Any]]] = []
    for _, aid, pmat, meta, descriptor in selected:
        mm = dict(meta)
        if pmat is None:
            if descriptor is None:
                raise RuntimeError(f"Internal V9 candidate error: {aid}")
            mm["_lazy_descriptor"] = descriptor
        out.append((aid, pmat, mm))
    # V9 deliberately performs no full-partition materialization here.
    prof["materialize"] = time.perf_counter() - t0
    prof["total"] = time.perf_counter() - t_all

    # Keep only lightweight aggregate profiling; no per-step log spam.
    if getattr(cfg, "profile_action_engine", False):
        global _ACTION_ENGINE_PROFILE, _ACTION_ENGINE_PROFILE_N
        try:
            _ACTION_ENGINE_PROFILE
        except NameError:
            _ACTION_ENGINE_PROFILE = defaultdict(float)
            _ACTION_ENGINE_PROFILE_N = defaultdict(int)
        for k, v in prof.items():
            _ACTION_ENGINE_PROFILE[k] += float(v)
            _ACTION_ENGINE_PROFILE_N[k] += 1

    return out


def psi_p(part: Partition, rel: Mapping[int, float], year_df: pd.DataFrame) -> float:
    zs = zone_statistics(part, year_df)
    if zs.empty:
        return 0.0
    indicated = zs["indicated_relativity"].to_numpy(float)
    weights = zs["weight"].to_numpy(float)
    zones = zs["zone"].to_numpy(int)
    r = np.asarray([float(rel.get(int(k), indicated[j])) for j, k in enumerate(zones)], dtype=float)
    return float(np.dot(weights, np.square(r - indicated)))


def uncertainty_c(part: Partition, year_df: pd.DataFrame) -> float:
    # transparent finite-sample proxy: inverse sqrt exposure, exposure-weighted.
    zs = zone_statistics(part, year_df)
    if zs.empty:
        return 0.0
    kappa = 1.0 / np.sqrt(np.maximum(zs["exposure"].to_numpy(float), 1.0))
    mx = float(np.max(kappa)) if kappa.size else 0.0
    if mx > 0:
        kappa = kappa / mx
    return float(np.dot(zs["weight"].to_numpy(float), kappa))


def _partition_transition_summary(
    prev: Partition,
    curr: Partition,
    year_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exposure aggregated by (previous zone, current zone), cached.

    This reduces premium-adjustment evaluation from O(n_communes) per pricing
    action to O(number of non-empty zone-pairs), typically two orders smaller.
    """
    key = (id(year_df), prev.signature(), curr.signature())
    cached = _TRANSITION_SUMMARY_CACHE.get(key)
    if cached is not None:
        _TRANSITION_SUMMARY_CACHE.move_to_end(key)
        return cached
    ctx = _year_arrays(year_df)
    old = _zone_assignment(prev, year_df).astype(np.int64)
    new = _zone_assignment(curr, year_df).astype(np.int64)
    valid = (old >= 0) & (new >= 0)
    if not valid.any():
        out = (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=float))
    else:
        oldv = old[valid]; newv = new[valid]; e = ctx["exposure"][valid]
        nk = int(newv.max()) + 1
        pair = oldv * max(nk, 1) + newv
        weights = np.bincount(pair, weights=e)
        nz = np.flatnonzero(weights > 0)
        out = (nz // max(nk, 1), nz % max(nk, 1), weights[nz])
    _bounded_put(_TRANSITION_SUMMARY_CACHE, key, out)
    return out


def zone_turnover(prev: Partition, curr: Partition, year_df: pd.DataFrame) -> float:
    key = (id(year_df), prev.signature(), curr.signature())
    cached = _TURNOVER_CACHE.get(key)
    if cached is not None:
        _TURNOVER_CACHE.move_to_end(key)
        return cached

    ctx = _year_arrays(year_df)
    old = _zone_assignment(prev, year_df).astype(np.int64)
    new = _zone_assignment(curr, year_df).astype(np.int64)
    valid = (old >= 0) & (new >= 0)
    if valid.sum() < 2:
        _bounded_put(_TURNOVER_CACHE, key, 0.0)
        return 0.0
    e = ctx["exposure"][valid]
    old = old[valid]; new = new[valid]
    W = 0.5 * (float(e.sum()) ** 2 - float(np.dot(e, e)))
    if W <= 0:
        _bounded_put(_TURNOVER_CACHE, key, 0.0)
        return 0.0

    def mass(labels: np.ndarray) -> float:
        es = np.bincount(labels, weights=e)
        e2 = np.bincount(labels, weights=e*e)
        return float(0.5 * np.sum(es*es - e2))

    new_k = int(new.max()) + 1
    same_old = mass(old)
    same_new = mass(new)
    pair_labels = old * max(new_k, 1) + new
    same_both = mass(pair_labels)
    val = float(np.clip((same_old + same_new - 2.0 * same_both) / W, 0.0, 1.0))
    _bounded_put(_TURNOVER_CACHE, key, val)
    return val


def premium_adjustment(
    part: Partition,
    rel: Mapping[int, float],
    prev_part: Partition,
    prev_rel: Mapping[int, float],
    year_df: pd.DataFrame,
) -> float:
    old_z, new_z, w = _partition_transition_summary(prev_part, part, year_df)
    if w.size == 0:
        return 0.0
    old_max = int(old_z.max()) + 1
    new_max = int(new_z.max()) + 1
    old_r = np.ones(old_max, dtype=float)
    new_r = np.ones(new_max, dtype=float)
    for k, v in prev_rel.items():
        if 0 <= int(k) < old_max:
            old_r[int(k)] = max(float(v), 1e-12)
    for k, v in rel.items():
        if 0 <= int(k) < new_max:
            new_r[int(k)] = max(float(v), 1e-12)
    numer = float(np.dot(w, np.abs(np.log(new_r[new_z] / old_r[old_z]))))
    return numer / max(float(w.sum()), 1e-12)


def _action_metric_bundle(
    action: Action,
    prev_part: Partition,
    prev_rel: Mapping[int, float],
    year_df: pd.DataFrame,
) -> Dict[str, float]:
    """Compute exact action metrics; V9 lazy actions carry them precomputed."""
    if action.lazy_metrics is not None:
        return action.lazy_metrics
    if action.partition is None:
        raise RuntimeError(f"Action metrics requested before lazy metrics were attached: {action.action_id}")
    rel_sig = hashlib.blake2b(
        repr(tuple(sorted((int(k), round(float(v), 10)) for k, v in action.relativities.items()))).encode(),
        digest_size=8,
    ).hexdigest()
    prev_rel_sig = hashlib.blake2b(
        repr(tuple(sorted((int(k), round(float(v), 10)) for k, v in prev_rel.items()))).encode(),
        digest_size=8,
    ).hexdigest()
    psig = action.partition.signature()
    prev_psig = prev_part.signature()
    key = (id(year_df), psig, rel_sig, prev_psig, prev_rel_sig)
    cached = _ACTION_METRIC_CACHE.get(key)
    if cached is not None:
        _ACTION_METRIC_CACHE.move_to_end(key)
        return cached

    # These three terms are identical for every pricing variant of one geographic edit.
    ckey = (id(year_df), prev_psig, psig)
    common = _PARTITION_COMMON_METRIC_CACHE.get(ckey)
    if common is None:
        common = {
            "psi_z": psi_z(action.partition, year_df),
            "c_zonal": uncertainty_c(action.partition, year_df),
            "z_delta": zone_turnover(prev_part, action.partition, year_df),
        }
        _bounded_put(_PARTITION_COMMON_METRIC_CACHE, ckey, common)
    else:
        _PARTITION_COMMON_METRIC_CACHE.move_to_end(ckey)

    out = {
        **common,
        "psi_p": psi_p(action.partition, action.relativities, year_df),
        "gamma_p": premium_adjustment(action.partition, action.relativities, prev_part, prev_rel, year_df),
    }
    _bounded_put(_ACTION_METRIC_CACHE, key, out, maxsize=128)
    return out


def action_features(
    action: Action,
    prev_part: Partition,
    prev_rel: Mapping[int, float],
    year_df: pd.DataFrame,
    cfg: Config,
) -> np.ndarray:
    m = _action_metric_bundle(action, prev_part, prev_rel, year_df)
    rel_vals = np.fromiter((float(v) for v in action.relativities.values()), dtype=float)
    edit_onehot = [float(action.edit_type == t) for t in ["KEEP", "SPLIT", "MERGE", "SHIFT"]]
    arr = np.asarray(
        [
            (action.lazy_k if action.lazy_k is not None else action.partition.k) / max(cfg.k_max, 1),
            m["psi_z"], m["psi_p"], m["c_zonal"], m["gamma_p"], m["z_delta"],
            float(rel_vals.mean()) if rel_vals.size else 1.0,
            float(rel_vals.std()) if rel_vals.size else 0.0,
            *edit_onehot,
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)


def generate_action_set(
    part: Partition,
    prev_rel: Mapping[int, float],
    year_df: pd.DataFrame,
    adjacency: Mapping[str, Sequence[str]],
    cfg: Config,
) -> List[Action]:
    """V9 lazy-action generator.

    V10 keeps SPLIT, MERGE and SHIFT descriptor-only through network scoring.
    Graph cutting still determines the exact connected SPLIT children, but the full
    34k-commune Partition is materialized once, only if the action is executed.
    """
    geo = generate_geographic_candidates(part, year_df, adjacency, cfg)
    actions: List[Action] = []
    seen = set()

    st = _partition_structure(part, adjacency, getattr(cfg, "structure_cache_size", 8))
    groups: Dict[int, List[str]] = st["groups"]
    base_zs = zone_statistics(part, year_df)
    zone_exposure = dict(zip(base_zs.zone.astype(int), base_zs.exposure.astype(float)))
    base_psi = psi_z(part, year_df)

    for geo_id, pnew, meta in geo:
        etype = geo_id.split("|", 1)[0]
        descriptor = meta.get("_lazy_descriptor")

        if pnew is not None:
            # Materialized action (normally KEEP).
            rel_candidates = pricing_candidates(pnew, year_df, cfg)
            for j, rel in enumerate(rel_candidates):
                aid = f"{geo_id}|PRICE={j}"
                if aid in seen:
                    continue
                a = Action(aid, etype, pnew, rel, metadata={k:v for k,v in meta.items() if k != "_lazy_descriptor"})
                a.features = action_features(a, part, prev_rel, year_df, cfg)
                if np.all(np.isfinite(a.features)):
                    actions.append(a)
                    seen.add(aid)
            continue

        if descriptor is None:
            raise RuntimeError(f"V9 lazy candidate missing descriptor: {geo_id}")

        # Exact post-edit zonal statistics without constructing a Partition.
        zs_new, mapping = _lazy_zone_stats_from_descriptor(part, year_df, descriptor, groups)
        psi_z_new = float(base_psi - float(meta.get("delta_psi_z", 0.0)))
        rel_candidates = _pricing_candidates_from_stats(zs_new, cfg)

        for j, rel in enumerate(rel_candidates):
            aid = f"{geo_id}|PRICE={j}"
            if aid in seen:
                continue
            metrics = _lazy_action_metrics(
                descriptor=descriptor,
                zs_new=zs_new,
                mapping=mapping,
                rel=rel,
                prev_rel=prev_rel,
                part=part,
                year_df=year_df,
                zone_exposure=zone_exposure,
                psi_z_new=psi_z_new,
            )
            a = Action(
                action_id=aid,
                edit_type=etype,
                partition=None,
                relativities=dict(rel),
                metadata={k:v for k,v in meta.items() if k != "_lazy_descriptor"},
                lazy_descriptor=descriptor,
                lazy_zone_stats=zs_new,
                lazy_metrics=metrics,
                lazy_k=int(len(zs_new)),
            )
            a.features = action_features(a, part, prev_rel, year_df, cfg)
            if np.all(np.isfinite(a.features)):
                actions.append(a)
                seen.add(aid)

    if not actions:
        if not partition_admissible(part, year_df, adjacency, cfg):
            raise RuntimeError(
                "Finite-action availability violated: current partition is not admissible and no feasible edit was generated"
            )
        rel = indicated_relativities(part, year_df)
        a = Action("KEEP|PRICE=0", "KEEP", part, rel)
        a.features = action_features(a, part, prev_rel, year_df, cfg)
        if not np.all(np.isfinite(a.features)):
            raise RuntimeError("Finite-action availability violated: KEEP fallback has non-finite features")
        actions = [a]

    return actions[: max(1, int(cfg.max_action_set))]


# -----------------------------------------------------------------------------
# 6. State encoding and controlled environment
# -----------------------------------------------------------------------------

STATE_FEATURES = [
    "risk_mean", "risk_sd", "risk_q90", "tail_ratio_mean", "rga_mean", "flood_mean",
    "storm_mean", "delta_hat", "exposure_log_mean", "k_scaled", "psi_z", "uncertainty",
]


def state_vector(part: Partition, year_df: pd.DataFrame, cfg: Config) -> np.ndarray:
    risk = year_df.mhat.to_numpy(float)
    tail = year_df.tail_mean_ratio.to_numpy(float)
    vals = np.asarray(
        [
            np.mean(risk),
            np.std(risk),
            np.quantile(risk, 0.90),
            np.mean(tail),
            np.mean(year_df.rga_hazard),
            np.mean(year_df.flood_hazard),
            np.mean(year_df.storm_hazard),
            np.mean(year_df.delta_hat),
            np.mean(np.log1p(year_df.exposure_proxy)),
            part.k / max(cfg.k_max, 1),
            psi_z(part, year_df),
            uncertainty_c(part, year_df),
        ], dtype=np.float32
    )
    # deterministic scale compression; avoids leakage from full-sample standardization.
    vals[:3] = np.log1p(np.maximum(vals[:3], 0.0))
    vals[8] /= 10.0
    vals = np.nan_to_num(vals, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
    return vals


def decision_cost(action: Action, prev_part: Partition, prev_rel: Mapping[int, float], year_df: pd.DataFrame, cfg: Config) -> Dict[str, float]:
    m = _action_metric_bundle(action, prev_part, prev_rel, year_df)
    total = (
        cfg.c_psi * (m["psi_z"] + m["psi_p"])
        + cfg.c_c * m["c_zonal"]
        + cfg.c_p * m["gamma_p"]
        + cfg.c_z * m["z_delta"]
    )
    return {**m, "decision_cost": float(total)}


def location_rel_from_action(action: Action, codes: Sequence[str]) -> np.ndarray:
    z = np.fromiter((int(action.partition.zone_of.get(str(c), -1)) for c in codes), dtype=np.int64, count=len(codes))
    valid = z >= 0
    out = np.ones(len(codes), dtype=float)
    if valid.any():
        rv = np.ones(int(z[valid].max()) + 1, dtype=float)
        for k, v in action.relativities.items():
            if 0 <= int(k) < len(rv):
                rv[int(k)] = float(v)
        out[valid] = rv[z[valid]]
    return out


# V7 shared environment caches.  These objects depend only on the immutable panel,
# horizon and finite-action configuration; they are independent of trajectory seed.
# Sharing them eliminates repeated pandas groupby, initial partition construction and
# first-state action generation across episodes, seeds and Monte Carlo trajectories.
_ENV_BY_YEAR_CACHE: "OrderedDict[Tuple[Any, ...], Dict[int, pd.DataFrame]]" = OrderedDict()
_ENV_INITIAL_CACHE: "OrderedDict[Tuple[Any, ...], Tuple[Partition, Dict[int, float], np.ndarray, List[Action]]]" = OrderedDict()
_ENV_CACHE_MAX = 12


def _environment_cache_key(
    panel: pd.DataFrame,
    adjacency: Mapping[str, Sequence[str]],
    years: Sequence[int],
    cfg: Config,
) -> Tuple[Any, ...]:
    return (
        id(panel), id(adjacency), tuple(map(int, years)),
        int(cfg.k_max), round(float(cfg.e_min), 8),
        int(cfg.m_split_per_zone), int(cfg.m_split_target_zones),
        int(cfg.m_merge_total), int(cfg.m_shift_total),
        int(cfg.m_geographic_total), int(cfg.m_pricing_per_partition),
        int(cfg.max_action_set), round(float(cfg.h), 8), round(float(cfg.delta_r), 8),
    )


def _cache_put_env(cache: OrderedDict, key: Tuple[Any, ...], value: Any) -> None:
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > _ENV_CACHE_MAX:
        cache.popitem(last=False)


class ZoningEnvironment:
    def __init__(
        self,
        panel: pd.DataFrame,
        adjacency: Mapping[str, Sequence[str]],
        cfg: Config,
        years: Sequence[int],
        stress: str = "historical",
        logger: Optional[logging.Logger] = None,
    ):
        self.panel = panel
        self.adjacency = adjacency
        self.cfg = cfg
        self.years = list(map(int, years))
        self.stress = stress
        self.logger = logger
        self._env_key = _environment_cache_key(panel, adjacency, self.years, cfg)

        # V7: build annual DataFrame views once per panel/horizon and share them
        # across all environments used in training, selection and test simulation.
        cached_by_year = _ENV_BY_YEAR_CACHE.get(self._env_key) if cfg.cache_environment_contexts else None
        if cached_by_year is None:
            tctx = time.perf_counter()
            sub = panel.loc[panel.year.isin(self.years)]
            by_year = {int(y): g.reset_index(drop=True) for y, g in sub.groupby("year", sort=True)}
            missing_years = [y for y in self.years if y not in by_year]
            if missing_years:
                raise RuntimeError(f"Environment missing requested years: {missing_years}")
            self.by_year = by_year
            # Force annual NumPy contexts now, outside the learning loop.
            for y in self.years:
                _year_arrays(self.by_year[y], cfg)
            if cfg.cache_environment_contexts:
                _cache_put_env(_ENV_BY_YEAR_CACHE, self._env_key, self.by_year)
            if self.logger is not None:
                self.logger.info("ENVCTX | years=%s..%s build=%.3fs", self.years[0], self.years[-1], time.perf_counter()-tctx)
        else:
            self.by_year = cached_by_year

        # V7: construct, validate and price the initial partition exactly once for
        # this immutable panel/horizon/action design.  The cached Action objects are
        # treated as immutable.
        initial_cached = _ENV_INITIAL_CACHE.get(self._env_key) if cfg.cache_initial_action_set else None
        if initial_cached is None:
            ti = time.perf_counter()
            first = self.by_year[self.years[0]]
            initial_part = initial_department_partition(first)
            validate_initial_partition(initial_part, first, self.adjacency, self.cfg, self.logger)
            initial_rel = indicated_relativities(initial_part, first)
            initial_state = state_vector(initial_part, first, self.cfg)
            initial_actions = generate_action_set(initial_part, initial_rel, first, self.adjacency, self.cfg)
            if not initial_actions:
                raise RuntimeError("Initial finite-action set is empty after successful admissibility validation")
            if cfg.cache_initial_action_set:
                _cache_put_env(
                    _ENV_INITIAL_CACHE, self._env_key,
                    (initial_part, dict(initial_rel), initial_state.copy(), initial_actions),
                )
            self.initial_part = initial_part
            self.initial_rel = dict(initial_rel)
            self.initial_state = initial_state.copy()
            self.initial_actions = initial_actions
            if self.logger is not None:
                self.logger.info(
                    "ENVINIT | years=%s..%s actions=%s build=%.3fs",
                    self.years[0], self.years[-1], len(initial_actions), time.perf_counter()-ti,
                )
        else:
            self.initial_part, rel0, s0, self.initial_actions = initial_cached
            self.initial_rel = dict(rel0)
            self.initial_state = s0.copy()

        self._state_cache: "OrderedDict[Tuple[int, str], np.ndarray]" = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        self._reset_count = 0
        self._profile = defaultdict(float)
        self._profile_n = defaultdict(int)
        first = self.by_year[self.years[0]]
        self._ones_exposure = np.ones(len(first), dtype=float)

    def reset(self, seed: int) -> Tuple[np.ndarray, List[Action]]:
        t0 = time.perf_counter()
        self.seed = int(seed)
        self.t = 0
        # Partitions/actions are immutable in the pipeline: transformations create
        # new Partition objects, so references to the prevalidated initial objects
        # are safe and avoid copying a 34k-entry dictionary at every episode.
        self.part = self.initial_part
        self.rel = self.initial_rel
        self.exposure_mult = self._ones_exposure.copy()
        self._reset_count += 1
        self._cache_hits += 1
        dt = time.perf_counter() - t0
        self._profile["reset"] += dt
        self._profile_n["reset"] += 1
        return self.initial_state.copy(), self.initial_actions

    def _year_df(self) -> pd.DataFrame:
        return self.by_year[self.years[self.t]]

    def _obs_actions(self) -> Tuple[np.ndarray, List[Action]]:
        ydf = self._year_df()
        year = int(self.years[self.t])
        psig = self.part.signature()
        skey = (year, psig)
        s = self._state_cache.get(skey)
        if s is None:
            s = state_vector(self.part, ydf, self.cfg)
            self._state_cache[skey] = s
            if len(self._state_cache) > 64:
                self._state_cache.popitem(last=False)
        else:
            self._state_cache.move_to_end(skey)

        t0 = time.perf_counter()
        acts = generate_action_set(self.part, self.rel, ydf, self.adjacency, self.cfg)
        build_s = time.perf_counter() - t0
        self._profile["actionset"] += build_s
        self._profile_n["actionset"] += 1
        self._cache_misses += 1
        if self.logger is not None and (self._cache_misses <= 5 or build_s > 5.0):
            self.logger.info(
                "ACTIONSET | year=%s actions=%s build=%.3fs reset_hits=%s dynamic_builds=%s",
                year, len(acts), build_s, self._cache_hits, self._cache_misses,
            )
        if not acts:
            raise RuntimeError(f"Finite-action availability failure at year={year}")
        assert_finite_array("state_vector", s)
        for i, a in enumerate(acts):
            assert_finite_array(f"action_features[{i}]", a.features)
        return s.copy(), acts

    def step(self, action: Action) -> Tuple[Optional[np.ndarray], float, bool, Dict[str, Any], List[Action]]:
        t_step = time.perf_counter()
        ydf = self._year_df()
        ctx = _year_arrays(ydf, self.cfg)

        t_cost = time.perf_counter()
        dc = decision_cost(action, self.part, self.rel, ydf, self.cfg)
        self._profile["decision_cost"] += time.perf_counter() - t_cost
        self._profile_n["decision_cost"] += 1

        # aggregate current premium is action invariant by model design
        mu_hat_total = float(np.dot(ctx["exposure"], ctx["mhat"]) / 1000.0)
        loading = self.cfg.loading_rate * mu_hat_total

        t_loss = time.perf_counter()
        loss = simulate_loss_vector(
            ydf,
            seed=self.seed * 100_003 + self.t * 101,
            cfg=self.cfg,
            stress=self.stress,
            exposure_multiplier=self.exposure_mult,
        )
        self._profile["loss_sim"] += time.perf_counter() - t_loss
        self._profile_n["loss_sim"] += 1
        loss_total = float(np.sum(loss))
        u = mu_hat_total + loading - loss_total
        reward = u - dc["decision_cost"]

        # V9: exactly one full partition is materialized per executed lazy action,
        # never for candidates that the policy does not select.
        t_mat = time.perf_counter()
        exec_part = _materialize_action_if_needed(action, self.part)
        self._profile["lazy_materialize"] += time.perf_counter() - t_mat
        self._profile_n["lazy_materialize"] += 1

        info = {
            "year": self.years[self.t],
            "action_id": action.action_id,
            "edit_type": action.edit_type,
            "K": exec_part.k,
            "underwriting_result": u,
            "loss_total": loss_total,
            "mu_hat_total": mu_hat_total,
            **dc,
        }

        if self.cfg.controlled_response_baseline:
            # Reuse cached commune order/assignment instead of rebuilding a list.
            z = _zone_assignment(exec_part, ydf).astype(np.int64)
            valid = z >= 0
            rloc = np.ones(len(z), dtype=float)
            if valid.any():
                rv = np.ones(int(z[valid].max()) + 1, dtype=float)
                for k, v in action.relativities.items():
                    if 0 <= int(k) < len(rv):
                        rv[int(k)] = float(v)
                rloc[valid] = rv[z[valid]]
            self.exposure_mult = np.clip(
                self.exposure_mult * np.exp(-self.cfg.exposure_elasticity * (rloc - 1.0)),
                0.80, 1.20,
            )

        # No copy: candidate partitions and relativity dictionaries are immutable.
        self.part = exec_part
        self.rel = action.relativities
        self.t += 1
        done = self.t >= len(self.years)
        if done:
            self._profile["step_total"] += time.perf_counter() - t_step
            self._profile_n["step_total"] += 1
            return None, float(reward), True, info, []

        s2, a2 = self._obs_actions()
        self._profile["step_total"] += time.perf_counter() - t_step
        self._profile_n["step_total"] += 1
        return s2, float(reward), False, info, a2

    def profile_summary(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for k, total in self._profile.items():
            n = max(int(self._profile_n.get(k, 0)), 1)
            out[f"{k}_mean_s"] = float(total / n)
            out[f"{k}_total_s"] = float(total)
        out["reset_count"] = float(self._reset_count)
        out["dynamic_actionset_builds"] = float(self._cache_misses)
        return out


# -----------------------------------------------------------------------------
# 7. PyTorch networks and replay
# -----------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.data: List[Any] = []
        self.pos = 0

    def add(self, item: Any) -> None:
        if len(self.data) < self.capacity:
            self.data.append(item)
        else:
            self.data[self.pos] = item
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, n: int, rng: np.random.Generator) -> List[Any]:
        idx = rng.choice(len(self.data), size=min(n, len(self.data)), replace=False)
        return [self.data[int(i)] for i in idx]

    def __len__(self) -> int:
        return len(self.data)


class QNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, n_out: int):
        super().__init__()
        d = state_dim + action_dim
        self.net = nn.Sequential(
            nn.Linear(d, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(),
            nn.Linear(hidden_dim // 2, n_out),
        )

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([s, a], dim=-1))


def quantile_huber_loss(pred: torch.Tensor, target: torch.Tensor, taus: torch.Tensor, kappa: float) -> torch.Tensor:
    # pred [B,Nq], target [B,Nq]
    u = target[:, None, :] - pred[:, :, None]  # [B,Nq,Nq]
    abs_u = torch.abs(u)
    huber = torch.where(abs_u <= kappa, 0.5 * u.pow(2), kappa * (abs_u - 0.5 * kappa)) / kappa
    indicator = (u.detach() < 0).float()
    weight = torch.abs(taus[None, :, None] - indicator)
    return (weight * huber).mean()


def empirical_tvar_from_atoms(atoms: torch.Tensor, alpha: float) -> torch.Tensor:
    # atoms are cumulative returns; downside loss atoms are -atoms.
    losses = -atoms
    sorted_loss, _ = torch.sort(losses, dim=-1)
    n = sorted_loss.shape[-1]
    start = min(n - 1, max(0, int(math.floor(alpha * n))))
    return sorted_loss[..., start:].mean(dim=-1)


def score_actions(
    net: QNet,
    state: np.ndarray,
    actions: Sequence[Action],
    mode: str,
    cfg: Config,
    device: torch.device,
) -> np.ndarray:
    if not actions:
        return np.empty(0, dtype=float)
    state = np.nan_to_num(np.asarray(state, dtype=np.float32), nan=0.0, posinf=1e6, neginf=-1e6)
    af = np.stack([np.nan_to_num(x.features, nan=0.0, posinf=1e6, neginf=-1e6) for x in actions]).astype(np.float32)
    s = torch.from_numpy(np.repeat(state[None, :], len(actions), axis=0)).to(device)
    a = torch.from_numpy(af).to(device)
    with torch.no_grad():
        out = net(s, a)
        out = torch.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)
        if out.shape[1] == 1:
            score = out[:, 0]
        else:
            mean = out.mean(dim=1)
            if mode == "tail":
                tvar = empirical_tvar_from_atoms(out, cfg.alpha)
                score = mean - cfg.lambda_tail * tvar
            else:
                score = mean
    return np.nan_to_num(score.detach().cpu().numpy(), nan=-np.inf, posinf=1e12, neginf=-1e12)


def choose_action(
    net: QNet,
    state: np.ndarray,
    actions: Sequence[Action],
    mode: str,
    cfg: Config,
    device: torch.device,
    epsilon: float,
    rng: np.random.Generator,
) -> int:
    n = len(actions)
    if n == 0:
        raise RuntimeError("choose_action called with an empty feasible action set")
    if n == 1:
        return 0
    if rng.random() < float(np.clip(epsilon, 0.0, 1.0)):
        return int(rng.integers(0, n))
    scores = score_actions(net, state, actions, mode, cfg, device)
    if scores.size != n:
        raise RuntimeError(f"Action-score size mismatch: {scores.size} scores for {n} actions")
    finite = np.isfinite(scores)
    if not finite.any():
        # Deterministic defensive fallback. This event is separately detectable in
        # saved diagnostics through finite network checks; never index an empty argmax.
        return 0
    safe = np.where(finite, scores, -np.inf)
    max_score = np.max(safe)
    winners = np.flatnonzero(safe == max_score)
    return int(winners[0]) if winners.size else int(np.flatnonzero(finite)[0])


def epsilon_schedule(step: int, cfg: Config) -> float:
    frac = min(1.0, step / max(cfg.epsilon_decay_steps, 1))
    return cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)



def _batched_next_action_values(
    target: QNet,
    batch: Sequence[Any],
    n_out: int,
    method: str,
    cfg: Config,
    device: torch.device,
) -> torch.Tensor:
    """Evaluate all nonterminal next-action sets with one target-network forward.

    V5 launched one CUDA forward per replay item (up to 256 launches/update).
    V6 concatenates variable action sets, performs one forward, then reduces each
    segment independently.  This is mathematically identical to the former code.
    """
    batch_size = len(batch)
    result = torch.zeros((batch_size, n_out), dtype=torch.float32, device=device)
    state_chunks: List[np.ndarray] = []
    action_chunks: List[np.ndarray] = []
    owners: List[int] = []
    lengths: List[int] = []

    for i, b in enumerate(batch):
        if b[4] or b[5] is None or len(b[5]) == 0:
            continue
        af = np.asarray(b[5], dtype=np.float32)
        if af.ndim == 1:
            af = af[None, :]
        m = int(af.shape[0])
        sf = np.repeat(np.asarray(b[3], dtype=np.float32)[None, :], m, axis=0)
        state_chunks.append(sf)
        action_chunks.append(af)
        owners.append(i)
        lengths.append(m)

    if not action_chunks:
        return result

    sf_all = torch.from_numpy(np.concatenate(state_chunks, axis=0)).to(device, non_blocking=True)
    af_all = torch.from_numpy(np.concatenate(action_chunks, axis=0)).to(device, non_blocking=True)
    qo_all = torch.nan_to_num(target(sf_all, af_all), nan=0.0, posinf=1e6, neginf=-1e6)

    offset = 0
    for owner, m in zip(owners, lengths):
        qo = qo_all[offset:offset + m]
        offset += m
        if n_out == 1:
            scores = qo[:, 0]
        else:
            meanq = qo.mean(dim=1)
            if method == "Tail-QR Zoning":
                scores = meanq - cfg.lambda_tail * empirical_tvar_from_atoms(qo, cfg.alpha)
            else:
                scores = meanq
        j = torch.argmax(scores)
        result[owner] = qo[j]
    return result


def train_agent(
    method: str,
    panel: pd.DataFrame,
    adjacency: Mapping[str, Sequence[str]],
    train_years: Sequence[int],
    seed: int,
    cfg: Config,
    device: torch.device,
    paths: Mapping[str, Path],
    logger: logging.Logger,
) -> Tuple[QNet, pd.DataFrame]:
    set_seed(seed, cfg.deterministic_torch)
    rng = np.random.default_rng(seed)
    env = ZoningEnvironment(panel, adjacency, cfg, train_years[: cfg.horizon], stress="historical", logger=logger)
    # V7 reset is O(n) only for exposure_multiplier allocation; initial action set is cached.
    state, actions = env.reset(seed)
    state_dim = len(state)
    action_dim = len(actions[0].features)
    n_out = 1 if method == "Mean DQN" else cfg.n_quantiles
    mode = "tail" if method == "Tail-QR Zoning" else "mean"
    net = QNet(state_dim, action_dim, cfg.hidden_dim, n_out).to(device)
    target = copy.deepcopy(net).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    replay = ReplayBuffer(cfg.replay_capacity)
    taus = ((torch.arange(cfg.n_quantiles, device=device).float() + 0.5) / cfg.n_quantiles)
    logs = []
    episode_seed = seed

    total_steps = cfg.train_steps_per_stage * cfg.evaluation_stages
    t0 = time.time()
    last_log_time = t0
    update_time_acc = 0.0
    env_time_acc = 0.0
    choose_time_acc = 0.0
    reset_time_acc = 0.0

    for step in range(total_steps):
        eps = epsilon_schedule(step, cfg)

        tc = time.perf_counter()
        idx = choose_action(net, state, actions, mode, cfg, device, eps, rng)
        choose_time_acc += time.perf_counter() - tc
        act = actions[idx]

        te = time.perf_counter()
        next_state, reward, done, info, next_actions = env.step(act)
        env_time_acc += time.perf_counter() - te

        # V7 replay stores one contiguous float32 array instead of a Python list
        # of up to 60 tiny arrays.
        if done or not next_actions:
            naf = np.empty((0, action_dim), dtype=np.float32)
        else:
            naf = np.stack([x.features for x in next_actions]).astype(np.float32, copy=False)
        replay.add((
            state.astype(np.float32, copy=True),
            act.features.astype(np.float32, copy=True),
            float(reward),
            None if done else next_state.astype(np.float32, copy=True),
            bool(done),
            naf,
        ))

        if done:
            episode_seed += 1
            tr = time.perf_counter()
            state, actions = env.reset(episode_seed)
            reset_time_acc += time.perf_counter() - tr
        else:
            state, actions = next_state, next_actions

        loss_value = np.nan
        threshold = max(cfg.batch_size, min(cfg.warmup_steps, cfg.batch_size * 2))
        if len(replay) >= threshold:
            tu = time.perf_counter()
            batch = replay.sample(cfg.batch_size, rng)
            s_np = np.stack([b[0] for b in batch]).astype(np.float32, copy=False)
            a_np = np.stack([b[1] for b in batch]).astype(np.float32, copy=False)
            r_np = np.asarray([b[2] for b in batch], dtype=np.float32)
            d_np = np.asarray([b[4] for b in batch], dtype=np.float32)
            s = torch.from_numpy(s_np).to(device, non_blocking=True)
            a = torch.from_numpy(a_np).to(device, non_blocking=True)
            r = torch.from_numpy(r_np).to(device, non_blocking=True)
            d = torch.from_numpy(d_np).to(device, non_blocking=True)
            pred = torch.nan_to_num(net(s, a), nan=0.0, posinf=1e6, neginf=-1e6)

            with torch.no_grad():
                nv = _batched_next_action_values(target, batch, n_out, method, cfg, device)
                target_atoms = r[:, None] + cfg.gamma * (1.0 - d[:, None]) * nv

            if n_out == 1:
                loss = F.smooth_l1_loss(pred[:, 0], target_atoms[:, 0])
            else:
                loss = quantile_huber_loss(pred, target_atoms, taus, cfg.huber_kappa)

            if torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), cfg.gradient_clip)
                grads_finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
                if grads_finite:
                    opt.step()
                    loss_value = float(loss.detach().cpu())
                else:
                    opt.zero_grad(set_to_none=True)
            else:
                opt.zero_grad(set_to_none=True)
            update_time_acc += time.perf_counter() - tu

        if (step + 1) % cfg.target_refresh == 0:
            target.load_state_dict(net.state_dict())

        if (step + 1) % cfg.log_every == 0 or step == 0:
            now = time.time()
            elapsed = now - t0
            interval = max(now - last_log_time, 1e-9)
            last_log_time = now
            steps_done = step + 1
            sps = steps_done / max(elapsed, 1e-9)
            prof = env.profile_summary()
            logger.info(
                "TRAIN | method=%s seed=%s step=%s/%s eps=%.3f replay=%s loss=%s "
                "elapsed=%.1fs rate=%.2f step/s reset_avg=%.4fs actionset_avg=%.3fs update_total=%.1fs",
                method, seed, steps_done, total_steps, eps, len(replay),
                f"{loss_value:.5g}" if np.isfinite(loss_value) else "NA",
                elapsed, sps,
                prof.get("reset_mean_s", 0.0),
                prof.get("actionset_mean_s", 0.0),
                update_time_acc,
            )
            logs.append({
                "method": method, "seed": seed, "step": steps_done,
                "epsilon": eps, "loss": loss_value, "replay": len(replay),
                "elapsed_s": elapsed, "steps_per_s": sps,
                "choose_total_s": choose_time_acc,
                "env_total_s": env_time_acc,
                "reset_total_s": reset_time_acc,
                "update_total_s": update_time_acc,
                **prof,
            })
            if getattr(cfg, "profile_action_engine", False) and "_ACTION_ENGINE_PROFILE" in globals():
                p_action = globals().get("_ACTION_ENGINE_PROFILE", {})
                n_action = globals().get("_ACTION_ENGINE_PROFILE_N", {})
                def _aavg(name: str) -> float:
                    return float(p_action.get(name, 0.0)) / max(int(n_action.get(name, 0)), 1)
                logger.info(
                    "ACTIONPROFILE | split=%.4fs merge=%.4fs shift=%.4fs "
                    "materialize=%.4fs topology=%.4fs moments=%.4fs total=%.4fs",
                    _aavg("split"), _aavg("merge"), _aavg("shift"),
                    _aavg("materialize"), _aavg("topology"),
                    _aavg("moments"), _aavg("total"),
                )

    model_path = paths["models"] / f"{method.lower().replace(' ', '_').replace('-', '_')}_seed{seed}.pt"
    torch.save({
        "state_dict": net.state_dict(), "state_dim": state_dim, "action_dim": action_dim,
        "n_out": n_out, "method": method, "seed": seed,
    }, model_path)
    return net, pd.DataFrame(logs)


# -----------------------------------------------------------------------------
# 8. Policies, trajectory evaluation, selection
# -----------------------------------------------------------------------------

METHODS = ["Static Zoning", "Mean DQN", "QR Mean", "Tail-QR Zoning"]


def static_action(actions: Sequence[Action]) -> int:
    keep = [i for i, a in enumerate(actions) if a.edit_type == "KEEP"]
    if keep:
        # among KEEP variants pick lowest pricing departure via feature psi_p (index 2)
        return min(keep, key=lambda i: float(actions[i].features[2]))
    return 0


def run_trajectory(
    method: str,
    net: Optional[QNet],
    panel: pd.DataFrame,
    adjacency: Mapping[str, Sequence[str]],
    years: Sequence[int],
    seed: int,
    cfg: Config,
    device: torch.device,
    stress: str,
) -> Tuple[float, List[Dict[str, Any]], Partition, Dict[int, float]]:
    env = ZoningEnvironment(panel, adjacency, cfg, years[: cfg.horizon], stress=stress)
    state, actions = env.reset(seed)
    total = 0.0
    infos = []
    disc = 1.0
    while True:
        if method == "Static Zoning":
            idx = static_action(actions)
        else:
            assert net is not None
            mode = "tail" if method == "Tail-QR Zoning" else "mean"
            scores = score_actions(net, state, actions, mode, cfg, device)
            idx = int(np.flatnonzero(scores == np.max(scores))[0])
        action = actions[idx]
        next_state, reward, done, info, next_actions = env.step(action)
        info.update({"method": method, "seed": seed, "stress": stress, "discount": disc, "reward": reward})
        infos.append(info)
        total += disc * reward
        disc *= cfg.gamma
        if done:
            break
        state, actions = next_state, next_actions
    return float(total), infos, env.part.copy(), dict(env.rel)


def empirical_tvar_np(losses: np.ndarray, alpha: float) -> float:
    x = np.sort(np.asarray(losses, dtype=float))
    if len(x) == 0:
        return np.nan
    # integrated quantile empirical TVaR; discrete approximation with fractional threshold weight.
    q = np.quantile(x, alpha, method="higher")
    tail = x[x >= q]
    return float(np.mean(tail)) if len(tail) else float(q)


def summarize_returns(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows = []
    for (method, stress), g in df.groupby(["method", "stress"]):
        ret = g["G0"].to_numpy(float)
        losses = -ret
        mean = float(np.mean(ret))
        tvar = empirical_tvar_np(losses, cfg.alpha)
        var = float(np.quantile(losses, cfg.alpha))
        j = mean - cfg.lambda_tail * tvar
        rows.append(
            {
                "method": method,
                "stress": stress,
                "mean_G0": mean,
                "median_G0": float(np.median(ret)),
                "VaR_alpha_adverse": var,
                "TVaR_alpha_adverse": tvar,
                "J_alpha_lambda": j,
                "sd_G0": float(np.std(ret, ddof=1)) if len(ret) > 1 else 0.0,
                "n": len(ret),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_paired_ci(values: np.ndarray, seed: int = 991, b: int = 2000) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(b, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


# -----------------------------------------------------------------------------
# 9. Signature tables
# -----------------------------------------------------------------------------


def write_signature_tables(
    cfg: Config,
    paths: Mapping[str, Path],
    data_manifest: pd.DataFrame,
    panel: pd.DataFrame,
    returns: pd.DataFrame,
    steps: pd.DataFrame,
    train_logs: pd.DataFrame,
    runtime_rows: pd.DataFrame,
    logger: logging.Logger,
) -> None:
    ctab = paths["csv_tables"]

    # Table 1: exact source provenance and actuarial role.
    t1 = data_manifest.copy()
    if not t1.empty:
        t1["spatial_resolution"] = np.where(t1.source_name.str.contains("commune|GASPAR|RGA", case=False, na=False), "commune / polygon", "station or source-specific")
        t1["temporal_role"] = np.where(t1.source_name.str.contains("GASPAR|Météo", case=False, na=False), "time-varying", "structural/current")
        t1["predecision_availability"] = "yes, lagged/dated inputs only"
    t1.to_csv(ctab / "table1_data_sources.csv", index=False)

    # Table 2: hyperparameters / economic / computational specification.
    rows = []
    for k, v in asdict(cfg).items():
        if isinstance(v, list):
            vv = json.dumps(v)
        else:
            vv = v
        if k in {"k_max", "e_min", "delta_r", "c_psi", "c_c", "c_p", "c_z", "loading_rate"}:
            grp = "economic"
        elif k in {"h", "m_split_per_zone", "m_merge_total", "m_shift_total", "m_geographic_total", "m_pricing_per_partition", "max_action_set"}:
            grp = "finite-action"
        elif k in {"n_quantiles", "alpha", "lambda_tail", "huber_kappa", "gamma"}:
            grp = "distributional"
        elif k in {"hidden_dim", "learning_rate", "batch_size", "replay_capacity", "target_refresh", "gradient_clip", "evaluation_stages"}:
            grp = "learning"
        elif k in {"seeds", "n_selection_trajectories", "n_test_trajectories", "common_random_numbers"}:
            grp = "evaluation"
        else:
            continue
        rows.append({"component": grp, "parameter": k, "baseline": vv})
    pd.DataFrame(rows).to_csv(ctab / "table2_hyperparameters.csv", index=False)

    # Table 3: design / policy contrasts.
    t3 = pd.DataFrame([
        {"policy": "Static Zoning", "dynamic_partition": False, "distributional_learner": False, "tail_sensitive_selection": False, "pricing_adaptive": True, "identifying_role": "adaptation benchmark"},
        {"policy": "Mean DQN", "dynamic_partition": True, "distributional_learner": False, "tail_sensitive_selection": False, "pricing_adaptive": True, "identifying_role": "scalar expected-return benchmark"},
        {"policy": "QR Mean", "dynamic_partition": True, "distributional_learner": True, "tail_sensitive_selection": False, "pricing_adaptive": True, "identifying_role": "distributional representation with mean ranking"},
        {"policy": "Tail-QR Zoning", "dynamic_partition": True, "distributional_learner": True, "tail_sensitive_selection": True, "pricing_adaptive": True, "identifying_role": "full proposed procedure"},
    ])
    t3.to_csv(ctab / "table3_experimental_design.csv", index=False)

    # Table 4: conditional risk calibration diagnostics.
    # In calibrated environment, q coverage is checked against simulated pseudo-losses only.
    rows4 = []
    rng_seeds = [17, 29, 43]
    sample_years = sorted(panel.year.unique())[-min(5, panel.year.nunique()):]
    for y in sample_years:
        g = panel[panel.year == y].reset_index(drop=True)
        sims = np.stack([simulate_loss_vector(g, s + int(y), cfg) for s in rng_seeds])
        # location pseudo-loss cost per exposure; three replications only for an inexpensive diagnostic.
        exp = g.exposure_proxy.to_numpy(float)
        cost = sims * 1000.0 / np.maximum(exp[None, :], 1e-9)
        mean_obs = cost.mean(axis=0)
        mae = float(np.mean(np.abs(mean_obs - g.mhat.to_numpy(float))))
        r = {"year": y, "mean_abs_calibration_error": mae, "delta_hat_mean": float(g.delta_hat.mean()), "n_communes": len(g), "environment": cfg.loss_environment_name}
        for tau in [x for x in cfg.loss_quantiles if x >= 0.90]:
            q = g[f"qhat_{tau:.2f}"].to_numpy(float)
            r[f"coverage_q{tau:.2f}"] = float(np.mean(cost <= q[None, :]))
        rows4.append(r)
    pd.DataFrame(rows4).to_csv(ctab / "table4_risk_calibration.csv", index=False)

    # Table 5: spatial zoning/action decomposition.
    if not steps.empty:
        t5a = steps.groupby(["method", "stress"]).agg(
            K_mean=("K", "mean"), K_median=("K", "median"),
            psi_z=("psi_z", "mean"), psi_p=("psi_p", "mean"),
            C_z=("c_zonal", "mean"), Gamma_P=("gamma_p", "mean"), Z_delta=("z_delta", "mean")
        ).reset_index()
        action_share = pd.crosstab([steps.method, steps.stress], steps.edit_type, normalize="index").reset_index()
        t5 = t5a.merge(action_share, on=["method", "stress"], how="left")
    else:
        t5 = pd.DataFrame()
    t5.to_csv(ctab / "table5_zoning_action_decomposition.csv", index=False)

    # Table 6: policy performance + paired CI vs Static.
    t6 = summarize_returns(returns, cfg)
    if not returns.empty and "Static Zoning" in returns.method.unique():
        extra = []
        for stress in returns.stress.unique():
            base = returns[(returns.method == "Static Zoning") & (returns.stress == stress)].set_index("trajectory_id")["G0"]
            for method in returns.method.unique():
                cur = returns[(returns.method == method) & (returns.stress == stress)].set_index("trajectory_id")["G0"]
                both = pd.concat([cur.rename("cur"), base.rename("base")], axis=1).dropna()
                d = both.cur - both.base
                lo, hi = bootstrap_paired_ci(d.to_numpy(), seed=123)
                extra.append({"method": method, "stress": stress, "delta_mean_G0_vs_static": float(d.mean()) if len(d) else np.nan, "paired_ci_low": lo, "paired_ci_high": hi})
        t6 = t6.merge(pd.DataFrame(extra), on=["method", "stress"], how="left")
    t6.to_csv(ctab / "table6_policy_performance.csv", index=False)

    # Table 7: robustness + computation.
    if not runtime_rows.empty:
        runtime_rows.to_csv(ctab / "table7_computational_diagnostics.csv", index=False)
    else:
        pd.DataFrame().to_csv(ctab / "table7_computational_diagnostics.csv", index=False)

    logger.info("Signature table CSVs written -> %s", ctab)


# -----------------------------------------------------------------------------
# 10. Signature figures + reproducible CSVs
# -----------------------------------------------------------------------------


def savefig(fig: plt.Figure, base: Path, cfg: Config) -> None:
    fig.savefig(base.with_suffix(".png"), dpi=cfg.figure_dpi, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    if cfg.save_svg:
        fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def write_tikz_figure1(paths: Mapping[str, Path]) -> Path:
    tex = r"""% Figure 1 — Information chronology and dynamic actuarial zoning pipeline
% Requires: \usepackage{tikz}; \usetikzlibrary{arrows.meta,positioning,fit}
\begin{tikzpicture}[
  >=Latex,
  node distance=7mm and 8mm,
  box/.style={draw, rounded corners, align=center, minimum height=8mm, inner sep=4pt},
  small/.style={draw, rounded corners, align=center, minimum height=7mm, inner sep=3pt},
  every edge/.style={draw,->,thick}
]
\node[box] (data) {Public spatial data\\available at $t$};
\node[box, right=of data] (x) {$H_{it},E_{it},B_i,C_{it}$};
\node[box, right=of x] (rhat) {$\widehat m_t,\widehat q_t,\widehat\Delta_t$};
\node[box, right=of rhat] (state) {$S_t$};
\node[box, right=of state] (aset) {$\mathcal A_{t,h}(S_t)$};
\node[box, right=of aset] (decision) {$D_t=(\mathcal Z_t,r_t)$};
\node[box, right=of decision] (outcome) {$L_{t+1},R_{t+1}$};
\node[box, right=of outcome] (next) {$S_{t+1}$};
\node[small, below=9mm of state] (inherit) {$\mathcal Z_{t-1},p_{t-1}$};
\draw[->,thick] (data)--(x);
\draw[->,thick] (x)--(rhat);
\draw[->,thick] (rhat)--(state);
\draw[->,thick] (state)--(aset);
\draw[->,thick] (aset)--(decision);
\draw[->,thick] (decision)--(outcome);
\draw[->,thick] (outcome)--(next);
\draw[->,thick] (inherit)--(state);
\end{tikzpicture}
"""
    p = paths["figures"] / "figure1_information_flow_tikz.tex"
    p.write_text(tex, encoding="utf-8")
    return p


def plot_maps_base(gdf: gpd.GeoDataFrame, data: pd.DataFrame, columns: Sequence[Tuple[str, str]], outbase: Path, cfg: Config, csv_path: Path) -> None:
    merge = gdf[["code_commune", "geometry"]].merge(data, on="code_commune", how="left")
    export_cols = ["code_commune"] + [c for c, _ in columns]
    merge.drop(columns="geometry")[export_cols].to_csv(csv_path, index=False)
    n = len(columns)
    ncols = 3
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.8 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (col, title) in zip(axes, columns):
        merge.plot(column=col, ax=ax, legend=True, linewidth=0.0, missing_kwds={"color": "lightgrey"})
        ax.set_title(title)
        ax.set_axis_off()
    for ax in axes[n:]:
        ax.set_axis_off()
    fig.tight_layout()
    savefig(fig, outbase, cfg)


def figure2_risk_atlas(gdf: gpd.GeoDataFrame, panel: pd.DataFrame, cfg: Config, paths: Mapping[str, Path]) -> None:
    year = int(panel.year.max())
    g = panel[panel.year == year].copy()
    qcol = f"qhat_{max(cfg.loss_quantiles):.2f}"
    columns = [
        ("rga_hazard", "(a) RGA / drought hazard"),
        ("flood_hazard", "(b) Flood hazard"),
        ("exposure_proxy", "(c) Exposure proxy"),
        ("mhat", "(d) Estimated mean loss cost"),
        (qcol, f"(e) Upper conditional quantile q={max(cfg.loss_quantiles):.2f}"),
        ("tail_mean_ratio", "(f) Tail-to-mean ratio"),
    ]
    plot_maps_base(gdf, g, columns, paths["figures"] / "figure2_climate_risk_atlas", cfg, paths["csv_figures"] / "figure2_climate_risk_atlas.csv")


def partition_frame(gdf: gpd.GeoDataFrame, part: Partition, rel: Mapping[int, float], label: str) -> pd.DataFrame:
    d = pd.DataFrame({"code_commune": list(part.zone_of), f"zone_{label}": list(part.zone_of.values())})
    d[f"rel_{label}"] = d[f"zone_{label}"].map(rel).fillna(1.0)
    return d


def figure3_zoning_maps(gdf: gpd.GeoDataFrame, panel: pd.DataFrame, final_parts: Dict[str, Tuple[Partition, Dict[int, float]]], cfg: Config, paths: Mapping[str, Path]) -> None:
    year = int(panel.year.max())
    g = panel[panel.year == year][["code_commune", "mhat"]].copy()
    avg = np.average(g.mhat, weights=panel[panel.year == year].exposure_proxy)
    g["loc_rel"] = g.mhat / max(avg, 1e-12)
    merged = g.copy()
    methods = [m for m in METHODS if m in final_parts]
    for m in methods:
        p, r = final_parts[m]
        safe = re.sub(r"[^a-z0-9]+", "_", m.lower()).strip("_")
        merged = merged.merge(partition_frame(gdf, p, r, safe), on="code_commune", how="left")
    merged.to_csv(paths["csv_figures"] / "figure3_implemented_zones.csv", index=False)

    nplots = 2 + len(methods)
    ncols = 3
    nrows = int(math.ceil(nplots / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.8*nrows))
    axes = np.atleast_1d(axes).ravel()
    geo = gdf[["code_commune", "geometry"]].merge(merged, on="code_commune", how="left")
    geo.plot(column="loc_rel", ax=axes[0], legend=True, linewidth=0)
    axes[0].set_title("(a) Estimated location-level relativity")
    axes[0].set_axis_off()
    pi = 1
    for m in methods:
        safe = re.sub(r"[^a-z0-9]+", "_", m.lower()).strip("_")
        geo.plot(column=f"zone_{safe}", ax=axes[pi], categorical=True, legend=False, linewidth=0.02)
        axes[pi].set_title(f"({chr(97+pi)}) {m} zones")
        axes[pi].set_axis_off()
        pi += 1
    if "Tail-QR Zoning" in methods and pi < len(axes):
        safe = "tail_qr_zoning"
        geo.plot(column=f"rel_{safe}", ax=axes[pi], legend=True, linewidth=0)
        axes[pi].set_title(f"({chr(97+pi)}) Tail-QR implemented relativity")
        axes[pi].set_axis_off()
        pi += 1
    for ax in axes[pi:]:
        ax.set_axis_off()
    fig.tight_layout()
    savefig(fig, paths["figures"] / "figure3_implemented_zones", cfg)


def figure4_mean_tail_frontier(returns: pd.DataFrame, cfg: Config, paths: Mapping[str, Path]) -> None:
    summ = summarize_returns(returns[returns.stress == "historical"], cfg)
    summ.to_csv(paths["csv_figures"] / "figure4_mean_tail_frontier.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for _, r in summ.iterrows():
        ax.scatter(r.mean_G0, r.TVaR_alpha_adverse, s=80)
        ax.annotate(r.method, (r.mean_G0, r.TVaR_alpha_adverse), xytext=(5, 4), textcoords="offset points")
    ax.set_xlabel("Mean cumulative actuarial return")
    ax.set_ylabel(f"TVaR$_{{{cfg.alpha:.2f}}}$ of adverse cumulative return")
    ax.set_title("Mean-return versus downside-tail policy frontier")
    ax.grid(alpha=0.2)
    savefig(fig, paths["figures"] / "figure4_mean_tail_frontier", cfg)


def figure5_return_distribution(returns: pd.DataFrame, cfg: Config, paths: Mapping[str, Path]) -> None:
    d = returns[returns.stress == "historical"].copy()
    d["adverse_return"] = -d.G0
    d.to_csv(paths["csv_figures"] / "figure5_adverse_return_ecdf.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 6))
    for method, g in d.groupby("method"):
        x = np.sort(g.adverse_return.to_numpy(float))
        y = np.arange(1, len(x)+1) / max(len(x), 1)
        ax.plot(x, y, label=method)
    ax.axhline(cfg.alpha, linestyle="--", linewidth=1)
    ax.set_xlabel("Adverse cumulative return $-G_0$")
    ax.set_ylabel("Empirical CDF")
    ax.set_title("Distribution of cumulative actuarial returns and adverse tail")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend()
    ax.grid(alpha=0.2)
    savefig(fig, paths["figures"] / "figure5_adverse_return_distribution", cfg)


def figure6_granularity_frontier(steps: pd.DataFrame, returns: pd.DataFrame, cfg: Config, paths: Mapping[str, Path]) -> None:
    # Aggregate policy-run granularity vs adverse-tail performance.
    s = steps.groupby(["method", "stress", "seed"]).agg(K=("K", "mean"), adjustment=("gamma_p", "mean"), turnover=("z_delta", "mean")).reset_index()
    r = returns.groupby(["method", "stress", "seed"], as_index=False).G0.mean()
    m = s.merge(r, on=["method", "stress", "seed"], how="inner")
    m["burden"] = cfg.c_p * m.adjustment + cfg.c_z * m.turnover
    m["adverse"] = -m.G0
    m.to_csv(paths["csv_figures"] / "figure6_granularity_frontier.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 6))
    for method, g in m[m.stress == "historical"].groupby("method"):
        ax.scatter(g.burden, g.adverse, s=20 + 5*g.K, alpha=0.65, label=method)
    ax.set_xlabel("Adjustment / reclassification burden")
    ax.set_ylabel("Adverse cumulative return")
    ax.set_title("Actuarial granularity frontier")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend()
    ax.grid(alpha=0.2)
    savefig(fig, paths["figures"] / "figure6_granularity_frontier", cfg)


def figure7_dynamic_adaptation(steps: pd.DataFrame, cfg: Config, paths: Mapping[str, Path]) -> None:
    d = steps.copy()
    d.to_csv(paths["csv_figures"] / "figure7_dynamic_adaptation.csv", index=False)
    fig = plt.figure(figsize=(10, 12))
    ax1 = fig.add_axes([0.10, 0.69, 0.84, 0.23])
    ax2 = fig.add_axes([0.10, 0.39, 0.84, 0.23])
    ax3 = fig.add_axes([0.10, 0.09, 0.84, 0.23])
    hist = d[d.stress == "historical"]
    for method, g in hist.groupby("method"):
        agg = g.groupby("year", as_index=False).agg(K=("K","mean"), gamma_p=("gamma_p","mean"), z_delta=("z_delta","mean"), reward=("reward","mean"))
        ax1.plot(agg.year, agg.K, marker="o", label=method)
        ax2.plot(agg.year, agg.gamma_p + agg.z_delta, marker="o", label=method)
        ax3.plot(agg.year, agg.reward, marker="o", label=method)
    ax1.set_ylabel("Mean number of zones")
    ax2.set_ylabel("Premium adjustment + zoning turnover")
    ax3.set_ylabel("One-period actuarial payoff")
    ax3.set_xlabel("Decision year")
    ax1.set_title("Dynamic zoning adaptation under climate-risk evolution")
    ax1.legend(ncol=2)
    for ax in [ax1, ax2, ax3]:
        ax.grid(alpha=0.2)
    savefig(fig, paths["figures"] / "figure7_dynamic_adaptation", cfg)


def build_all_figures(
    gdf: gpd.GeoDataFrame,
    panel: pd.DataFrame,
    returns: pd.DataFrame,
    steps: pd.DataFrame,
    final_parts: Dict[str, Tuple[Partition, Dict[int, float]]],
    cfg: Config,
    paths: Mapping[str, Path],
    logger: logging.Logger,
) -> None:
    write_tikz_figure1(paths)
    figure2_risk_atlas(gdf, panel, cfg, paths)
    if final_parts:
        figure3_zoning_maps(gdf, panel, final_parts, cfg, paths)
    figure4_mean_tail_frontier(returns, cfg, paths)
    figure5_return_distribution(returns, cfg, paths)
    figure6_granularity_frontier(steps, returns, cfg, paths)
    figure7_dynamic_adaptation(steps, cfg, paths)
    logger.info("Signature figures written -> %s", paths["figures"])


# -----------------------------------------------------------------------------
# 11. Synthetic smoke-test data
# -----------------------------------------------------------------------------


def make_smoke_data(cfg: Config, paths: Mapping[str, Path], logger: logging.Logger) -> Tuple[gpd.GeoDataFrame, pd.DataFrame, Dict[str, List[str]]]:
    nxg, nyg = cfg.smoke_nx, cfg.smoke_ny
    rows = []
    for iy in range(nyg):
        for ix in range(nxg):
            idx = iy * nxg + ix
            code = f"{idx+1:05d}"
            rows.append({"code_commune": code, "commune_name": f"C{idx}", "department_code": f"{1 + ix//4:02d}", "geometry": box(ix, iy, ix+1, iy+1)})
    gdf = gpd.GeoDataFrame(rows, crs=4326)
    adjacency = {}
    for r in rows:
        c = r["code_commune"]
        idx = int(c)-1
        x, y = idx % nxg, idx // nxg
        nbs=[]
        for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            xx,yy=x+dx,y+dy
            if 0<=xx<nxg and 0<=yy<nyg:
                nbs.append(f"{yy*nxg+xx+1:05d}")
        adjacency[c]=nbs
    years = list(range(2015, 2015+cfg.smoke_years))
    rng=np.random.default_rng(123)
    panel_rows=[]
    for year in years:
        drift=(year-years[0])/max(len(years)-1,1)
        for idx,r in enumerate(rows):
            x=idx%nxg; y=idx//nxg
            rga=np.clip((x/(nxg-1))*0.7+0.2*drift+0.1*rng.random(),0,1)
            flood=np.clip((1-y/(nyg-1))*0.7+0.1*rng.random(),0,1)
            storm=np.clip(0.2+0.3*rng.random(),0,1)
            hazard=.5*rga+.35*flood+.15*storm
            exposure=1000+80*x+50*y
            vul=np.clip(0.3+0.5*(y/(nyg-1)),0,1)
            mhat=cfg.loss_scale*np.exp(hazard+.35*vul-.55)
            base={"code_commune":r["code_commune"],"department_code":r["department_code"],"area_km2":1.0,"population":exposure,"exposure_proxy":exposure,"log_exposure":np.log1p(exposure),"year":year,"rga_event":0,"flood_event":0,"storm_event":0,"rga_hist":rga*5,"flood_hist":flood*5,"storm_hist":storm*5,"rga_hazard":rga,"flood_hazard":flood,"storm_hazard":storm,"vulnerability_proxy":vul,"hazard_index":hazard,"mhat":mhat,"neighbor_hazard":hazard,"delta_hat":0.6}
            for tau,mult in [(0.5,1.0),(0.75,1.3),(0.90,1.8),(0.95,2.2),(0.99,3.2)]:
                base[f"qhat_{tau:.2f}"]=mhat*mult
            base["tail_mean_ratio"]=3.2
            panel_rows.append(base)
    panel=pd.DataFrame(panel_rows)
    logger.info("Synthetic smoke data ready | communes=%s years=%s", len(gdf), len(years))
    return gdf,panel,adjacency


# -----------------------------------------------------------------------------
# 12. Main experiment orchestration
# -----------------------------------------------------------------------------


def years_by_split(cfg: Config, panel: pd.DataFrame) -> Tuple[List[int], List[int], List[int]]:
    yrs = sorted(int(x) for x in pd.Series(panel.year).dropna().unique())
    if not yrs:
        raise RuntimeError("No years available for temporal split")
    train = [y for y in yrs if y <= cfg.train_end_year]
    sel = [y for y in yrs if cfg.train_end_year < y <= cfg.selection_end_year]
    test = [y for y in yrs if y >= cfg.test_start_year]
    # Hard reviewer-safe separation: no fallback is allowed to create overlaps.
    if set(train) & set(sel) or set(train) & set(test) or set(sel) & set(test):
        raise RuntimeError(f"Temporal leakage: overlapping split years | train={train} sel={sel} test={test}")
    for name, block in [("train", train), ("selection", sel), ("test", test)]:
        if len(block) < cfg.horizon:
            raise RuntimeError(
                f"{name} split has {len(block)} years, fewer than horizon={cfg.horizon}. "
                "Change split dates or horizon; overlapping fallback is forbidden."
            )
    if max(train) >= min(sel) or max(sel) >= min(test):
        raise RuntimeError("Temporal split is not strictly chronological")
    return train, sel, test


def validate_panel_for_training(panel: pd.DataFrame, cfg: Config, logger: logging.Logger) -> pd.DataFrame:
    required = [
        "code_commune", "department_code", "year", "exposure_proxy", "hazard_index",
        "mhat", "tail_mean_ratio", "rga_hazard", "flood_hazard", "storm_hazard", "delta_hat"
    ]
    missing = [c for c in required if c not in panel.columns]
    if missing:
        raise RuntimeError(f"Risk panel missing required columns: {missing}")
    out = panel.copy()
    numeric = [c for c in required if c not in {"code_commune", "department_code"}] + [c for c in out.columns if c.startswith("qhat_")]
    for c in dict.fromkeys(numeric):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    bad = {c: int((~np.isfinite(out[c].to_numpy(dtype=float))).sum()) for c in numeric}
    bad = {k:v for k,v in bad.items() if v}
    if bad:
        logger.warning("Sanitizing non-finite panel values before training | %s", bad)
        for c in bad:
            vals = out[c].to_numpy(dtype=float)
            finite = vals[np.isfinite(vals)]
            fill = float(np.median(finite)) if finite.size else 0.0
            out[c] = np.nan_to_num(vals, nan=fill, posinf=fill, neginf=fill)
    if (out["exposure_proxy"] <= 0).any():
        raise RuntimeError("Non-positive exposure_proxy values remain after panel construction")
    if (out["mhat"] <= 0).any():
        raise RuntimeError("Non-positive mhat values remain after panel construction")
    if out.duplicated(["code_commune", "year"]).any():
        raise RuntimeError("Duplicate commune-year rows detected in risk panel")
    logger.info("Preflight panel validation PASS | rows=%s communes=%s years=%s", len(out), out.code_commune.nunique(), out.year.nunique())
    return out


def run_experiment(
    gdf: gpd.GeoDataFrame,
    panel: pd.DataFrame,
    adjacency: Mapping[str, Sequence[str]],
    cfg: Config,
    paths: Mapping[str, Path],
    logger: logging.Logger,
    device: torch.device,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Tuple[Partition, Dict[int, float]]]]:
    train_years, sel_years, test_years = years_by_split(cfg, panel)
    logger.info("Temporal split | train=%s..%s sel=%s..%s test=%s..%s", min(train_years),max(train_years),min(sel_years),max(sel_years),min(test_years),max(test_years))

    # We train one model per seed and method. Static has no network.
    nets: Dict[Tuple[str,int], QNet] = {}
    train_log_parts=[]
    runtime=[]
    learn_methods=["Mean DQN","QR Mean","Tail-QR Zoning"]
    for method in learn_methods:
        for seed in cfg.seeds:
            t0=time.time()
            net, tlog=train_agent(method,panel,adjacency,train_years,seed,cfg,device,paths,logger)
            nets[(method,seed)]=net
            train_log_parts.append(tlog)
            runtime.append({"method":method,"seed":seed,"phase":"training","runtime_s":time.time()-t0,"parameters":sum(p.numel() for p in net.parameters()),"device":str(device)})

    # Independent test trajectories; common random numbers across methods.
    stress_scenarios=["historical"] if (len(cfg.seeds) == 1 and cfg.train_steps_per_stage <= 5) else ["historical","mean_shift","tail_amplification","spatial_concentration"]
    returns_rows=[]
    step_rows=[]
    final_parts={}
    n_per_seed=max(1, int(math.ceil(cfg.n_test_trajectories/max(len(cfg.seeds),1))))
    traj_id=0
    for stress in stress_scenarios:
        for si,seed in enumerate(cfg.seeds):
            for rep in range(n_per_seed):
                noise_seed=10_000_000 + si*100_000 + rep if cfg.common_random_numbers else 10_000_000 + si*100_000 + rep*31
                for method in METHODS:
                    net=None if method=="Static Zoning" else nets[(method,seed)]
                    t0=time.time()
                    G0,infos,pfin,rfin=run_trajectory(method,net,panel,adjacency,test_years,noise_seed,cfg,device,stress)
                    returns_rows.append({"trajectory_id":traj_id,"method":method,"seed":seed,"rep":rep,"stress":stress,"G0":G0})
                    for info in infos:
                        step_rows.append(info)
                    runtime.append({"method":method,"seed":seed,"phase":f"test_{stress}","runtime_s":time.time()-t0,"parameters":0 if net is None else sum(p.numel() for p in net.parameters()),"device":str(device)})
                    if stress=="historical" and rep==0 and si==0:
                        final_parts[method]=(pfin,rfin)
                traj_id+=1

    returns=pd.DataFrame(returns_rows)
    steps=pd.DataFrame(step_rows)
    train_logs=pd.concat(train_log_parts,ignore_index=True) if train_log_parts else pd.DataFrame()
    runtime_df=pd.DataFrame(runtime)
    returns.to_csv(paths["processed"] / "policy_returns_all.csv",index=False)
    steps.to_csv(paths["processed"] / "policy_steps_all.csv",index=False)
    train_logs.to_csv(paths["processed"] / "training_logs_all.csv",index=False)
    runtime_df.to_csv(paths["processed"] / "runtime_diagnostics.csv",index=False)
    return returns,steps,train_logs,runtime_df,final_parts


def make_manifest(cfg: Config, paths: Mapping[str, Path], logger: logging.Logger, extra: Optional[Dict[str,Any]]=None) -> Path:
    payload={
        "script_version":SCRIPT_VERSION,
        "created_utc":utc_now(),
        "python":sys.version,
        "torch":torch.__version__,
        "cuda_available":torch.cuda.is_available(),
        "cuda_device":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "scientific_status":{
            "insured_claims_observed": bool(cfg.claims_file),
            "loss_environment": "observed insured claims" if cfg.claims_file else cfg.loss_environment_name,
            "causal_interpretation_of_action_dependent_transition": False,
            "global_optimization_claim": False,
            "bellman_representation_of_precommitment_tail_criterion": False,
        },
        "seeds":cfg.seeds,
        "outputs":{},
    }
    for key in ["figures","csv_figures","csv_tables","processed","models"]:
        payload["outputs"][key]=sorted(str(p) for p in paths[key].glob("*"))
    if extra:
        payload.update(extra)
    p=paths["manifests"] / "run_manifest.json"
    p.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    logger.info("Run manifest written | %s",p)
    return p


def production_pipeline(cfg: Config, smoke: bool=False) -> None:
    paths=setup_paths(cfg)
    logger=setup_logger(paths["logs"])
    save_config(cfg,paths)
    logger.info("IME Tail-QR pipeline start | version=%s | smoke=%s",SCRIPT_VERSION,smoke)
    device=select_device(cfg,logger)

    if smoke:
        gdf,panel,adjacency=make_smoke_data(cfg,paths,logger)
        data_manifest=pd.DataFrame([
            {"source_name":"synthetic smoke grid","source_url":"","local_path":"","status":"smoke-only","model_role":"software validation","observed_or_constructed":"synthetic","downloaded_utc":utc_now(),"sha256":"","notes":"Not manuscript evidence"}
        ])
    else:
        data_manifest=download_public_data(cfg,paths,logger)
        gdf=load_communes(paths,cfg,logger)
        pop=load_population(paths,logger)
        catnat=load_catnat_events(paths,logger)
        adjacency=build_adjacency(gdf,paths,cfg,logger)
        panel=build_commune_year_panel(gdf,pop,catnat,cfg,paths,logger)
        panel=add_dependence_summary(panel,adjacency,cfg,paths,logger)
        panel=validate_panel_for_training(panel,cfg,logger)

    returns,steps,train_logs,runtime,final_parts=run_experiment(gdf,panel,adjacency,cfg,paths,logger,device)
    write_signature_tables(cfg,paths,data_manifest,panel,returns,steps,train_logs,runtime,logger)
    build_all_figures(gdf,panel,returns,steps,final_parts,cfg,paths,logger)
    make_manifest(cfg,paths,logger,extra={"smoke_test":smoke})
    logger.info("DONE | results=%s",paths["results"])


# -----------------------------------------------------------------------------
# 13. Rebuild outputs from saved CSVs (no retraining)
# -----------------------------------------------------------------------------


def rebuild_outputs(cfg: Config) -> None:
    paths=setup_paths(cfg)
    logger=setup_logger(paths["logs"])
    returns_p=paths["processed"] / "policy_returns_all.csv"
    steps_p=paths["processed"] / "policy_steps_all.csv"
    panel_p=paths["processed"] / "commune_year_risk_panel_with_dependence_v3.parquet"
    if not returns_p.exists() or not steps_p.exists() or not panel_p.exists():
        raise FileNotFoundError("Need processed policy returns, step logs, and risk panel to rebuild outputs")
    returns=pd.read_csv(returns_p)
    steps=pd.read_csv(steps_p)
    panel=pd.read_parquet(panel_p)
    # Figures 4,5,6,7 can be rebuilt directly from CSVs. Figures 2/3 require geography/partition snapshots.
    figure4_mean_tail_frontier(returns,cfg,paths)
    figure5_return_distribution(returns,cfg,paths)
    figure6_granularity_frontier(steps,returns,cfg,paths)
    figure7_dynamic_adaptation(steps,cfg,paths)
    write_tikz_figure1(paths)
    logger.info("Rebuilt figures 1,4,5,6,7 without retraining")



# =============================================================================
# V11 SCIENTIFIC-FINAL OVERRIDES
# =============================================================================
# The V10 spatial engine is retained.  The overrides below reconcile the code
# with Sections 2--4: policy-conditional evaluation with a frozen continuation,
# finite candidate-policy construction, independent selection on complete G0,
# untouched test evaluation, heterogeneous conditional tails, an encoder that
# explicitly summarizes every component of Rhat_t and the inherited decision,
# and fast Monte Carlo evaluation of frozen policies.

V11_STATE_FEATURES = [
    # Rhat_t: conditional mean surface
    "m_mean","m_sd","m_p10","m_p50","m_p90",
    # Rhat_t: conditional loss quantiles / tail shape
    "q50_mean","q90_mean","q99_mean","tail_ratio_mean","tail_ratio_sd","tail_ratio_p90",
    # Rhat_t: dependence summary
    "delta_mean","delta_sd","delta_p10","delta_p90","neighbor_hazard_mean",
    # X_t and active support summaries
    "rga_mean","flood_mean","storm_mean","vulnerability_mean",
    "exposure_log_mean","exposure_log_sd","active_share",
    # inherited partition / premium state
    "k_scaled","psi_z","uncertainty","prev_rel_mean","prev_rel_sd","prev_rel_min","prev_rel_max",
]
STATE_FEATURES = V11_STATE_FEATURES


def _safe_q(a: np.ndarray, q: float) -> float:
    a=np.asarray(a,dtype=float)
    a=a[np.isfinite(a)]
    return float(np.quantile(a,q)) if a.size else 0.0


def state_vector(
    part: Partition,
    year_df: pd.DataFrame,
    cfg: Config,
    prev_rel: Optional[Mapping[int,float]]=None,
    adjacency: Optional[Mapping[str,Sequence[str]]]=None,
) -> np.ndarray:
    """Finite encoder of S_t=(Rhat_t,X_t,a_t,Z_{t-1},p_{t-1}).

    The encoder is intentionally permutation/label invariant: it summarizes the
    spatial fields and inherited tariff by empirical moments/quantiles while the
    action encoder carries the candidate-specific economic decision.  No future
    or postdecision variable enters this vector.
    """
    m=finite_array(year_df["mhat"].to_numpy(float),fill=0.0)
    qcols={float(c.split('_',1)[1]):c for c in year_df.columns if str(c).startswith('qhat_')}
    def qfield(target: float) -> np.ndarray:
        if not qcols: return m.copy()
        tau=min(qcols,key=lambda x:abs(x-target))
        return finite_array(year_df[qcols[tau]].to_numpy(float),fill=0.0)
    q50,q90,q99=qfield(.50),qfield(.90),qfield(.99)
    tr=finite_array(year_df.get("tail_mean_ratio",pd.Series(np.ones(len(year_df)))).to_numpy(float),fill=1.0)
    de=finite_array(year_df.get("delta_hat",pd.Series(np.zeros(len(year_df)))).to_numpy(float),fill=0.0)
    nh=finite_array(year_df.get("neighbor_hazard",pd.Series(np.zeros(len(year_df)))).to_numpy(float),fill=0.0)
    ex=finite_array(year_df["exposure_proxy"].to_numpy(float),fill=0.0)
    active=ex>0
    relvals=np.asarray(list((prev_rel or indicated_relativities(part,year_df)).values()),dtype=float)
    if relvals.size==0: relvals=np.ones(1)
    vals=np.asarray([
        np.mean(m),np.std(m),_safe_q(m,.10),_safe_q(m,.50),_safe_q(m,.90),
        np.mean(q50),np.mean(q90),np.mean(q99),np.mean(tr),np.std(tr),_safe_q(tr,.90),
        np.mean(de),np.std(de),_safe_q(de,.10),_safe_q(de,.90),np.mean(nh),
        float(np.mean(year_df.get("rga_hazard",0.0))),float(np.mean(year_df.get("flood_hazard",0.0))),
        float(np.mean(year_df.get("storm_hazard",0.0))),float(np.mean(year_df.get("vulnerability_proxy",0.0))),
        np.mean(np.log1p(np.maximum(ex,0))),np.std(np.log1p(np.maximum(ex,0))),float(np.mean(active)),
        part.k/max(cfg.k_max,1),psi_z(part,year_df),uncertainty_c(part,year_df),
        np.mean(relvals),np.std(relvals),np.min(relvals),np.max(relvals),
    ],dtype=np.float32)
    # scale only large monetary risk coordinates; deterministic, no sample leakage
    vals[:8]=np.log1p(np.maximum(vals[:8],0.0))
    vals[20:22]/=10.0
    vals=np.nan_to_num(vals,nan=0.0,posinf=1e6,neginf=-1e6).astype(np.float32)
    return vals


# Preserve original environment methods but make the state encoder receive the
# inherited relativity vector explicitly.
_v10_env_init = ZoningEnvironment.__init__
_v10_env_obs = ZoningEnvironment._obs_actions

def _v11_env_init(self,*args,**kwargs):
    _v10_env_init(self,*args,**kwargs)
    first=self.by_year[self.years[0]]
    self.initial_state=state_vector(self.initial_part,first,self.cfg,self.initial_rel,self.adjacency)
    # update cached tuple with V11 state dimensionality
    if self.cfg.cache_initial_action_set:
        _cache_put_env(_ENV_INITIAL_CACHE,self._env_key,(self.initial_part,dict(self.initial_rel),self.initial_state.copy(),self.initial_actions))
ZoningEnvironment.__init__=_v11_env_init

def _v11_env_obs(self):
    ydf=self._year_df(); year=int(self.years[self.t]); psig=self.part.signature(); skey=(year,psig,tuple(sorted((int(k),round(float(v),10)) for k,v in self.rel.items())))
    s=self._state_cache.get(skey)
    if s is None:
        s=state_vector(self.part,ydf,self.cfg,self.rel,self.adjacency)
        self._state_cache[skey]=s
        if len(self._state_cache)>64: self._state_cache.popitem(last=False)
    else: self._state_cache.move_to_end(skey)
    t0=time.perf_counter(); acts=generate_action_set(self.part,self.rel,ydf,self.adjacency,self.cfg); build_s=time.perf_counter()-t0
    self._profile["actionset"]+=build_s; self._profile_n["actionset"]+=1; self._cache_misses+=1
    if self.logger is not None and (self._cache_misses<=5 or build_s>5.0):
        self.logger.info("ACTIONSET | year=%s actions=%s build=%.3fs reset_hits=%s dynamic_builds=%s",year,len(acts),build_s,self._cache_hits,self._cache_misses)
    if not acts: raise RuntimeError(f"Finite-action availability failure at year={year}")
    assert_finite_array("state_vector",s)
    for i,a in enumerate(acts): assert_finite_array(f"action_features[{i}]",a.features)
    return s.copy(),acts
ZoningEnvironment._obs_actions=_v11_env_obs


def _student_positive_mean(df: float) -> float:
    nu=max(float(df),2.01)
    return float(math.sqrt(nu)*math.gamma((nu+1.0)/2.0)/(math.sqrt(math.pi)*(nu-1.0)*math.gamma(nu/2.0)))


def build_commune_year_panel(communes,population,catnat,cfg,paths,logger):
    """V11 calibrated actuarial environment with spatially heterogeneous tails."""
    cache=paths["processed"] / "commune_year_risk_panel_v11.parquet"
    if cache.exists() and not cfg.force_compute:
        z=pd.read_parquet(cache)
        required={"local_sigma","local_tail_weight","tail_mean_ratio"}
        if required.issubset(z.columns):
            logger.info("Loading V11 cached risk panel")
            return z
    geo=communes[["code_commune","department_code","geometry"]].copy(); geo2154=geo.to_crs(2154)
    geo["area_km2"]=geo2154.geometry.area.values/1e6
    base=geo.drop(columns="geometry").merge(population,on="code_commune",how="left")
    if "population" not in base: base["population"]=np.nan
    fallback=np.maximum(base["area_km2"].to_numpy(),1.0)*100.0
    pop=pd.to_numeric(base["population"],errors="coerce").to_numpy(dtype=float)
    exposure=np.where(np.isfinite(pop)&(pop>0),pop,fallback)
    base["exposure_proxy"]=exposure; base["log_exposure"]=np.log1p(exposure)
    years=np.arange(cfg.start_year,cfg.end_year+1)
    panel=base.loc[base.index.repeat(len(years))].reset_index(drop=True); panel["year"]=np.tile(years,len(base))
    cat=classify_catnat(catnat)
    if not cat.empty:
        agg=cat.groupby(["code_commune","year"],as_index=False)[["rga_event","flood_event","storm_event"]].sum()
        panel=panel.merge(agg,on=["code_commune","year"],how="left")
    for c in ["rga_event","flood_event","storm_event"]:
        if c not in panel: panel[c]=0
        panel[c]=panel[c].fillna(0).astype(float)
    panel=panel.sort_values(["code_commune","year"]).reset_index(drop=True)
    for c in ["rga_event","flood_event","storm_event"]:
        panel[c.replace('_event','_hist')]=(panel.groupby("code_commune",sort=False)[c]
            .transform(lambda x:x.shift(1).rolling(cfg.rolling_window_years,min_periods=1).sum()).fillna(0.0))
    for src,dst in [("rga_hist","rga_hazard"),("flood_hist","flood_hazard"),("storm_hist","storm_hazard")]:
        panel[dst]=panel.groupby("year")[src].rank(pct=True,method="average").fillna(.5)
    density=panel["exposure_proxy"]/panel["area_km2"].clip(lower=.1); panel["_density"]=density
    panel["vulnerability_proxy"]=panel.groupby("year")["_density"].rank(pct=True,method="average").fillna(.5); panel.drop(columns="_density",inplace=True)
    panel["hazard_index"]=.50*panel.rga_hazard+.35*panel.flood_hazard+.15*panel.storm_hazard
    panel["mhat"]=cfg.loss_scale*np.exp(1.00*panel.hazard_index+.35*panel.vulnerability_proxy-.55)
    # Local tail scale/weight makes tail geography non-proportional to mean geography.
    panel["local_sigma"]=cfg.idiosyncratic_sigma*(.65+.45*panel.rga_hazard+.25*panel.flood_hazard+.20*panel.vulnerability_proxy)
    panel["local_tail_weight"]=(.08+.22*panel.rga_hazard+.10*panel.flood_hazard+.08*panel.vulnerability_proxy).clip(.05,.55)
    muplus=_student_positive_mean(cfg.tail_df); sigc=float(cfg.spatial_common_factor)*.20
    for tau in cfg.loss_quantiles:
        z=float(norm.ppf(float(tau))); tq=max(float(student_t.ppf(float(tau),df=max(cfg.tail_df,2.1))),0.0)
        sig=np.sqrt(sigc**2+panel.local_sigma.to_numpy(float)**2)
        logn=np.exp(sig*z-.5*sig**2)
        w=panel.local_tail_weight.to_numpy(float); tf=(1.0+w*tq)/(1.0+w*muplus)
        panel[f"qhat_{tau:.2f}"]=panel.mhat.to_numpy(float)*logn*tf
    qmax=f"qhat_{max(cfg.loss_quantiles):.2f}"; panel["tail_mean_ratio"]=panel[qmax]/panel.mhat.clip(lower=1e-9)
    panel.to_parquet(cache,index=False)
    logger.info("V11 risk panel built | rows=%s communes=%s years=%s tail_ratio_sd=%.4f",len(panel),panel.code_commune.nunique(),panel.year.nunique(),float(panel.tail_mean_ratio.std()))
    return panel


def add_dependence_summary(panel,adjacency,cfg,paths,logger):
    cache=paths["processed"] / "commune_year_risk_panel_with_dependence_v12.parquet"
    if cache.exists() and not cfg.force_compute:
        out=pd.read_parquet(cache)
        if {"delta_hat","neighbor_hazard"}.issubset(out.columns) and np.isfinite(out.delta_hat).all():
            logger.info("Loading V11 validated dependence panel"); return out
    logger.info("Constructing V11 spatial dependence summary Deltahat_t")
    out_parts=[]; diag=[]
    for year,gy in panel.groupby("year",sort=True):
        z=gy.copy(); h=finite_array(z.hazard_index.to_numpy(float),fill=0.0); risk=dict(zip(z.code_commune.astype(str),h))
        nh=[]
        for code,own in zip(z.code_commune.astype(str),h):
            vals=[risk[n] for n in adjacency.get(code,()) if n in risk]
            nh.append(float(np.mean(vals)) if vals else float(own))
        nh=np.asarray(nh); sx=float(np.std(h)); sy=float(np.std(nh))
        if len(h)>=3 and sx>1e-12 and sy>1e-12:
            hx=h-h.mean(); ny=nh-nh.mean(); den=math.sqrt(float(np.dot(hx,hx)*np.dot(ny,ny))); rho=float(np.dot(hx,ny)/den) if den>1e-18 else 0.0
        else: rho=0.0
        rho=float(np.clip(rho,-1,1)) if math.isfinite(rho) else 0.0
        z["neighbor_hazard"]=nh
        # local, F_t-measurable dependence-loading summary; retains annual spatial rho
        # while allowing the encoder to distinguish where common-risk exposure is concentrated.
        z["delta_hat"]=np.clip(rho*(.5+.5*nh),-1.0,1.0)
        out_parts.append(z); diag.append({"year":int(year),"annual_neighbor_rho":rho,"delta_mean":float(z.delta_hat.mean()),"delta_sd":float(z.delta_hat.std())})
    out=pd.concat(out_parts,ignore_index=True); pd.DataFrame(diag).to_csv(paths["metadata"] / "dependence_diagnostics_v11.csv",index=False); out.to_parquet(cache,index=False)
    logger.info("V11 dependence ready | years=%s delta_range=[%.4f, %.4f]",out.year.nunique(),out.delta_hat.min(),out.delta_hat.max()); return out


def simulate_loss_vector(year_df,seed,cfg,stress="historical",exposure_multiplier=None):
    """Mean-correct calibrated loss environment with local tail heterogeneity."""
    rng=np.random.default_rng(int(seed)); n=len(year_df); m=year_df.mhat.to_numpy(float).copy(); hazard=year_df.hazard_index.to_numpy(float)
    if stress=="mean_shift": m*=1.0+cfg.stress_mean_shift
    elif stress=="spatial_concentration":
        q=np.quantile(hazard,.80); m*=1.0+cfg.stress_spatial_concentration*(hazard>=q)
    sig=year_df.get("local_sigma",pd.Series(np.full(n,cfg.idiosyncratic_sigma))).to_numpy(float)
    w=year_df.get("local_tail_weight",pd.Series(np.full(n,.20))).to_numpy(float)
    sigc=float(cfg.spatial_common_factor)*.20
    common_mult=math.exp(sigc*rng.standard_normal()-.5*sigc**2)
    eps=np.exp(sig*rng.standard_normal(n)-.5*sig**2)
    tp=np.maximum(rng.standard_t(df=max(cfg.tail_df,2.1),size=n),0.0); muplus=_student_positive_mean(cfg.tail_df)
    tail_scale=cfg.stress_tail_multiplier if stress=="tail_amplification" else 1.0
    tail_factor=(1.0+w*tail_scale*tp)/(1.0+w*tail_scale*muplus)
    mult=common_mult*eps*tail_factor
    exposure=year_df.exposure_proxy.to_numpy(float)
    if exposure_multiplier is not None: exposure=exposure*np.asarray(exposure_multiplier,float)
    return exposure*m*mult/1000.0


def empirical_tvar_np(losses: np.ndarray, alpha: float) -> float:
    """Exact integrated-quantile TVaR of the empirical equal-mass distribution."""
    x=np.sort(np.asarray(losses,dtype=float)); n=len(x)
    if n==0: return np.nan
    a=float(np.clip(alpha,0.0,np.nextafter(1.0,0.0))); k=min(n-1,int(math.floor(a*n)))
    upper=(k+1)/n; integ=max(upper-a,0.0)*x[k]
    if k+1<n: integ+=float(np.sum(x[k+1:]))/n
    return float(integ/max(1.0-a,1e-15))


def empirical_tvar_from_atoms(atoms: torch.Tensor, alpha: float) -> torch.Tensor:
    losses=torch.sort(-atoms,dim=-1).values; n=losses.shape[-1]; a=float(min(max(alpha,0.0),np.nextafter(1.0,0.0))); k=min(n-1,int(math.floor(a*n)))
    first=((k+1)/n-a)*losses[...,k]
    rest=losses[...,k+1:].sum(dim=-1)/n if k+1<n else torch.zeros_like(first)
    return (first+rest)/max(1.0-a,1e-15)


@dataclass
class PolicyCandidate:
    method: str
    seed: int
    stage: int
    state_dim: int
    action_dim: int
    n_out: int
    mode: str
    state_dict: Dict[str,torch.Tensor]


def _snapshot_net(net: QNet,method:str,seed:int,stage:int,state_dim:int,action_dim:int,n_out:int,mode:str) -> PolicyCandidate:
    sd={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
    return PolicyCandidate(method,seed,stage,state_dim,action_dim,n_out,mode,sd)


def _candidate_net(c: PolicyCandidate,cfg:Config,device:torch.device) -> QNet:
    net=QNet(c.state_dim,c.action_dim,cfg.hidden_dim,c.n_out).to(device); net.load_state_dict(c.state_dict); net.eval(); return net


def _static_idx_from_features(af: np.ndarray, edits: Sequence[str]) -> int:
    keep=[i for i,e in enumerate(edits) if e=="KEEP"]
    return min(keep,key=lambda i:float(af[i,2])) if keep else 0


def _frozen_choose_indices(policy_net: Optional[QNet],policy_kind:str,mode:str,state_chunks,action_chunks,edit_chunks,cfg,device):
    idxs=[]
    if policy_kind=="static":
        return [_static_idx_from_features(a,e) for a,e in zip(action_chunks,edit_chunks)]
    assert policy_net is not None
    sf=np.concatenate([np.repeat(np.asarray(s,dtype=np.float32)[None,:],len(a),axis=0) for s,a in zip(state_chunks,action_chunks)],axis=0)
    af=np.concatenate(action_chunks,axis=0).astype(np.float32,copy=False)
    with torch.no_grad(): out=torch.nan_to_num(policy_net(torch.from_numpy(sf).to(device),torch.from_numpy(af).to(device)),nan=0.0,posinf=1e6,neginf=-1e6)
    off=0
    for a in action_chunks:
        q=out[off:off+len(a)]; off+=len(a)
        score=q[:,0] if q.shape[1]==1 else (q.mean(1)-cfg.lambda_tail*empirical_tvar_from_atoms(q,cfg.alpha) if mode=="tail" else q.mean(1))
        idxs.append(int(torch.argmax(score).item()))
    return idxs


def _batched_frozen_next_values(target,policy_net,policy_kind,policy_mode,batch,n_out,cfg,device):
    result=torch.zeros((len(batch),n_out),dtype=torch.float32,device=device); owners=[]; states=[]; acts=[]; edits=[]
    for i,b in enumerate(batch):
        # tuple: s,a,r,s2,done,next_af,next_edits
        if b[4] or b[5] is None or len(b[5])==0: continue
        owners.append(i); states.append(np.asarray(b[3],dtype=np.float32)); acts.append(np.asarray(b[5],dtype=np.float32)); edits.append(b[6])
    if not owners: return result
    idxs=_frozen_choose_indices(policy_net,policy_kind,policy_mode,states,acts,edits,cfg,device)
    sf=np.stack([states[j] for j in range(len(states))]).astype(np.float32)
    af=np.stack([acts[j][idxs[j]] for j in range(len(acts))]).astype(np.float32)
    with torch.no_grad(): val=torch.nan_to_num(target(torch.from_numpy(sf).to(device),torch.from_numpy(af).to(device)),nan=0.0,posinf=1e6,neginf=-1e6)
    for row,owner in enumerate(owners): result[owner]=val[row]
    return result


def _choose_frozen_action(policy_net,policy_kind,mode,state,actions,cfg,device,epsilon,rng):
    if len(actions)==1:return 0
    if rng.random()<float(np.clip(epsilon,0,1)): return int(rng.integers(0,len(actions)))
    if policy_kind=="static": return static_action(actions)
    return int(np.argmax(score_actions(policy_net,state,actions,mode,cfg,device)))


def train_agent(method,panel,adjacency,train_years,seed,cfg,device,paths,logger):
    """Stage-wise policy-conditional evaluator fitting with frozen continuation."""
    set_seed(seed,cfg.deterministic_torch); rng=np.random.default_rng(seed)
    env=ZoningEnvironment(panel,adjacency,cfg,train_years[:cfg.horizon],stress="historical",logger=logger); state,actions=env.reset(seed)
    state_dim=len(state); action_dim=len(actions[0].features); n_out=1 if method=="Mean DQN" else cfg.n_quantiles; construct_mode="tail" if method=="Tail-QR Zoning" else "mean"
    net=QNet(state_dim,action_dim,cfg.hidden_dim,n_out).to(device); opt=torch.optim.AdamW(net.parameters(),lr=cfg.learning_rate,weight_decay=cfg.weight_decay)
    taus=((torch.arange(cfg.n_quantiles,device=device).float()+.5)/cfg.n_quantiles); logs=[]; candidates=[]; global_step=0; episode_seed=seed
    frozen_kind="static"; frozen_net=None; frozen_mode=construct_mode; t_all=time.time()
    for stage in range(int(cfg.evaluation_stages)):
        # Freeze continuation for the whole evaluator-fitting stage.
        if frozen_net is not None: frozen_net.eval(); [p.requires_grad_(False) for p in frozen_net.parameters()]
        target=copy.deepcopy(net).to(device); target.eval(); replay=ReplayBuffer(cfg.replay_capacity)
        state,actions=env.reset(episode_seed); stage_t=time.time(); update_time=0.0
        for local in range(int(cfg.train_steps_per_stage)):
            global_step+=1; eps=epsilon_schedule(global_step-1,cfg)
            idx=_choose_frozen_action(frozen_net,frozen_kind,frozen_mode,state,actions,cfg,device,eps,rng); act=actions[idx]
            next_state,reward,done,info,next_actions=env.step(act)
            if done or not next_actions:
                naf=np.empty((0,action_dim),dtype=np.float32); ned=[]
            else:
                naf=np.stack([x.features for x in next_actions]).astype(np.float32,copy=False); ned=[x.edit_type for x in next_actions]
            replay.add((state.astype(np.float32,copy=True),act.features.astype(np.float32,copy=True),float(reward),None if done else next_state.astype(np.float32,copy=True),bool(done),naf,ned))
            if done:
                episode_seed+=1; state,actions=env.reset(episode_seed)
            else: state,actions=next_state,next_actions
            loss_value=np.nan; threshold=max(cfg.batch_size,min(cfg.warmup_steps,cfg.batch_size*2))
            if len(replay)>=threshold:
                tu=time.perf_counter(); batch=replay.sample(cfg.batch_size,rng)
                s=torch.from_numpy(np.stack([b[0] for b in batch]).astype(np.float32)).to(device); a=torch.from_numpy(np.stack([b[1] for b in batch]).astype(np.float32)).to(device)
                r=torch.from_numpy(np.asarray([b[2] for b in batch],dtype=np.float32)).to(device); d=torch.from_numpy(np.asarray([b[4] for b in batch],dtype=np.float32)).to(device)
                pred=torch.nan_to_num(net(s,a),nan=0.0,posinf=1e6,neginf=-1e6)
                with torch.no_grad():
                    nv=_batched_frozen_next_values(target,frozen_net,frozen_kind,frozen_mode,batch,n_out,cfg,device); targ=r[:,None]+cfg.gamma*(1.0-d[:,None])*nv
                loss=F.smooth_l1_loss(pred[:,0],targ[:,0]) if n_out==1 else quantile_huber_loss(pred,targ,taus,cfg.huber_kappa)
                if torch.isfinite(loss):
                    opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(),cfg.gradient_clip)
                    if all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters()): opt.step(); loss_value=float(loss.detach().cpu())
                    else: opt.zero_grad(set_to_none=True)
                update_time+=time.perf_counter()-tu
            if (local+1)%cfg.target_refresh==0: target.load_state_dict(net.state_dict())
            if (local+1)%cfg.log_every==0 or local==0:
                elapsed=time.time()-t_all; sps=global_step/max(elapsed,1e-9); prof=env.profile_summary()
                logger.info("EVALSTAGE | method=%s seed=%s stage=%s/%s continuation=%s step=%s/%s loss=%s rate=%.2f step/s actionset_avg=%.3fs",method,seed,stage+1,cfg.evaluation_stages,frozen_kind,local+1,cfg.train_steps_per_stage,f"{loss_value:.5g}" if np.isfinite(loss_value) else "NA",sps,prof.get("actionset_mean_s",0.0))
                logs.append({"method":method,"seed":seed,"stage":stage+1,"continuation_kind":frozen_kind,"step":global_step,"stage_step":local+1,"epsilon":eps,"loss":loss_value,"elapsed_s":elapsed,"steps_per_s":sps,"update_total_s":update_time,**prof})
        # evaluator -> admissible candidate; candidate is frozen before next stage
        cand=_snapshot_net(net,method,seed,stage+1,state_dim,action_dim,n_out,construct_mode); candidates.append(cand)
        stage_path=paths["models"] / f"{method.lower().replace(' ','_').replace('-','_')}_seed{seed}_stage{stage+1}.pt"
        torch.save({"state_dict":cand.state_dict,"state_dim":state_dim,"action_dim":action_dim,"n_out":n_out,"method":method,"seed":seed,"stage":stage+1,"continuation_kind":frozen_kind,"candidate_mode":construct_mode},stage_path)
        frozen_net=copy.deepcopy(net).to(device); frozen_kind="network"; frozen_mode=construct_mode
    return candidates,pd.DataFrame(logs)


def _policy_action_index(method,net,state,actions,cfg,device):
    if method=="Static Zoning": return static_action(actions)
    mode="tail" if method=="Tail-QR Zoning" else "mean"; scores=score_actions(net,state,actions,mode,cfg,device); return int(np.flatnonzero(scores==np.max(scores))[0])


def build_policy_path(method,net,panel,adjacency,years,cfg,device,stress="historical"):
    """Construct the deterministic decision path once; losses do not drive S_{t+1}."""
    env=ZoningEnvironment(panel,adjacency,cfg,years[:cfg.horizon],stress=stress); state,actions=env.reset(987654321); recs=[]
    while True:
        idx=_policy_action_index(method,net,state,actions,cfg,device); act=actions[idx]; ydf=env._year_df(); expmult=env.exposure_mult.copy(); dc=decision_cost(act,env.part,env.rel,ydf,cfg); ctx=_year_arrays(ydf,cfg); mu=float(np.dot(ctx["exposure"],ctx["mhat"])/1000.0); loading=cfg.loading_rate*mu
        next_state,_,done,info,next_actions=env.step(act)
        recs.append({"year":int(info["year"]),"action_id":act.action_id,"edit_type":act.edit_type,"K":int(info["K"]),"ydf":ydf,"exposure_mult":expmult,"mu_hat_total":mu,"loading":loading,**dc})
        if done: break
        state,actions=next_state,next_actions
    return recs,env.part.copy(),dict(env.rel)


def evaluate_policy_path(method,path,noise_seeds,cfg,stress,seed_label,stage_label=None,collect_steps=True):
    returns=[]; steps=[]
    for rep,noise_seed in enumerate(noise_seeds):
        disc=1.0; G=0.0; tid=int(noise_seed)
        for tt,r in enumerate(path):
            loss=float(np.sum(simulate_loss_vector(r["ydf"],int(noise_seed)*100003+tt*101,cfg,stress=stress,exposure_multiplier=r["exposure_mult"])))
            u=r["mu_hat_total"]+r["loading"]-loss; rew=u-r["decision_cost"]; G+=disc*rew
            if collect_steps:
                steps.append({"trajectory_id":tid,"method":method,"seed":seed_label,"candidate_stage":stage_label,"rep":rep,"stress":stress,"year":r["year"],"action_id":r["action_id"],"edit_type":r["edit_type"],"K":r["K"],"underwriting_result":u,"loss_total":loss,"mu_hat_total":r["mu_hat_total"],"reward":rew,"discount":disc,"psi_z":r["psi_z"],"psi_p":r["psi_p"],"c_zonal":r["c_zonal"],"gamma_p":r["gamma_p"],"z_delta":r["z_delta"],"decision_cost":r["decision_cost"]})
            disc*=cfg.gamma
        returns.append(G)
    return np.asarray(returns,float),steps


def select_candidate_policy(method,candidates,panel,adjacency,sel_years,seed,cfg,device,paths,logger):
    n=max(1,int(math.ceil(cfg.n_selection_trajectories/max(len(cfg.seeds),1)))); noise=[20_000_000+seed*10000+j for j in range(n)]; rows=[]; best=None; bestj=-np.inf
    for cand in candidates:
        net=_candidate_net(cand,cfg,device); path,_,_=build_policy_path(method,net,panel,adjacency,sel_years,cfg,device,"historical"); ret,_=evaluate_policy_path(method,path,noise,cfg,"historical",seed,cand.stage,collect_steps=False)
        mean=float(ret.mean()); tv=empirical_tvar_np(-ret,cfg.alpha); lam=cfg.lambda_tail if method=="Tail-QR Zoning" else 0.0; j=mean-lam*tv
        rows.append({"method":method,"seed":seed,"candidate_stage":cand.stage,"n_selection":len(ret),"mean_G0_sel":mean,"TVaR_alpha_adverse_sel":tv,"lambda_selection":lam,"J_sel":j})
        if j>bestj: bestj=j; best=cand
        del net
        if device.type=="cuda": torch.cuda.empty_cache()
    if best is None: raise RuntimeError(f"No continuation-admissible candidate selected for {method} seed={seed}")
    for r in rows:r["selected"]=(r["candidate_stage"]==best.stage)
    logger.info("SELECT | method=%s seed=%s selected_stage=%s Jsel=%.6g candidates=%s",method,seed,best.stage,bestj,len(candidates))
    return best,pd.DataFrame(rows)


def _save_partition_snapshots(final_parts,paths):
    rows=[]
    for method,(p,r) in final_parts.items():
        for code,z in p.zone_of.items(): rows.append({"method":method,"code_commune":code,"zone":int(z),"relativity":float(r.get(int(z),1.0))})
    pd.DataFrame(rows).to_csv(paths["processed"] / "final_partition_snapshots.csv",index=False)


def run_experiment(gdf,panel,adjacency,cfg,paths,logger,device):
    train_years,sel_years,test_years=years_by_split(cfg,panel); logger.info("Temporal split | train=%s..%s sel=%s..%s test=%s..%s",min(train_years),max(train_years),min(sel_years),max(sel_years),min(test_years),max(test_years))
    learn=["Mean DQN","QR Mean","Tail-QR Zoning"]; selected={}; train_parts=[]; sel_parts=[]; runtime=[]; first_seed_phase=[]
    # seed outer loop enables a meaningful full-pipeline runtime gate after one replica of all learned methods
    for si,seed in enumerate(cfg.seeds):
        seed_t=time.time()
        for method in learn:
            t0=time.time(); cands,tlog=train_agent(method,panel,adjacency,train_years,seed,cfg,device,paths,logger); train_parts.append(tlog)
            best,slog=select_candidate_policy(method,cands,panel,adjacency,sel_years,seed,cfg,device,paths,logger); sel_parts.append(slog); selected[(method,seed)]=best
            runtime.append({"method":method,"seed":seed,"phase":"train_plus_selection","runtime_s":time.time()-t0,"candidate_stage":best.stage,"device":str(device)})
        if si==0:
            first_seed_elapsed=time.time()-seed_t; projected=first_seed_elapsed*len(cfg.seeds)/3600.0
            logger.info("RUNTIMEGATE | first-seed learned-method block=%.2fh projected_10seed_train_selection=%.2fh limit=%.2fh",first_seed_elapsed/3600,projected,cfg.runtime_gate_hours)
            if cfg.enforce_runtime_gate and projected>cfg.runtime_gate_hours:
                raise RuntimeError(f"V11 runtime gate failed: projected train+selection {projected:.2f}h > {cfg.runtime_gate_hours:.2f}h. Reduce train_steps_per_stage or disable enforce_runtime_gate explicitly after review.")
    # Final untouched test.  Frozen policy path is built once per selected policy/stress; MC varies losses only.
    returns_rows=[]; step_rows=[]; final_parts={}; traj_counter=0
    stresses=["historical","mean_shift","tail_amplification","spatial_concentration"]
    for stress in stresses:
        total_n=cfg.n_test_trajectories if stress=="historical" else cfg.n_stress_trajectories; n_per_seed=max(1,int(math.ceil(total_n/max(len(cfg.seeds),1))))
        for si,seed in enumerate(cfg.seeds):
            noise=[30_000_000+si*100_000+j for j in range(n_per_seed)]
            for method in METHODS:
                if method=="Static Zoning": net=None; stage=0
                else: cand=selected[(method,seed)]; net=_candidate_net(cand,cfg,device); stage=cand.stage
                t0=time.time(); path,pfin,rfin=build_policy_path(method,net,panel,adjacency,test_years,cfg,device,stress); ret,st=evaluate_policy_path(method,path,noise,cfg,stress,seed,stage,collect_steps=True)
                for rep,G in enumerate(ret): returns_rows.append({"trajectory_id":30_000_000+si*100_000+rep,"method":method,"seed":seed,"rep":rep,"stress":stress,"G0":float(G),"selected_stage":stage})
                step_rows.extend(st); runtime.append({"method":method,"seed":seed,"phase":f"test_{stress}","runtime_s":time.time()-t0,"candidate_stage":stage,"device":str(device)})
                if stress=="historical" and si==0: final_parts[method]=(pfin,rfin)
                if net is not None: del net
                if device.type=="cuda": torch.cuda.empty_cache()
    returns=pd.DataFrame(returns_rows); steps=pd.DataFrame(step_rows); train_logs=pd.concat(train_parts,ignore_index=True) if train_parts else pd.DataFrame(); selection=pd.concat(sel_parts,ignore_index=True) if sel_parts else pd.DataFrame(); runtime_df=pd.DataFrame(runtime)
    returns.to_csv(paths["processed"] / "policy_returns_all.csv",index=False); steps.to_csv(paths["processed"] / "policy_steps_all.csv",index=False); train_logs.to_csv(paths["processed"] / "training_logs_all.csv",index=False); selection.to_csv(paths["processed"] / "candidate_selection.csv",index=False); runtime_df.to_csv(paths["processed"] / "runtime_diagnostics.csv",index=False); _save_partition_snapshots(final_parts,paths)
    return returns,steps,train_logs,runtime_df,final_parts


# Augment signature tables with selection diagnostics and robustness in Table 7.
_v10_write_signature_tables=write_signature_tables

def write_signature_tables(cfg,paths,data_manifest,panel,returns,steps,train_logs,runtime_rows,logger):
    _v10_write_signature_tables(cfg,paths,data_manifest,panel,returns,steps,train_logs,runtime_rows,logger)
    ctab=paths["csv_tables"]; selp=paths["processed"] / "candidate_selection.csv"; sel=pd.read_csv(selp) if selp.exists() else pd.DataFrame()
    # Table 3 gets explicit train/selection/test roles.
    t3p=ctab / "table3_experimental_design.csv"; t3=pd.read_csv(t3p)
    t3["selection_rule"]=np.where(t3.policy.eq("Tail-QR Zoning"),"max complete-return J on independent selection trajectories",np.where(t3.policy.eq("Static Zoning"),"prespecified static benchmark","max complete-return mean on independent selection trajectories")); t3["test_policy_frozen"]="yes"; t3.to_csv(t3p,index=False)
    # Table 4: stronger calibration MC (20 replicates/year rather than 3) and tail-resolution record.
    rows=[]; sample_years=sorted(panel.year.unique())[-min(5,panel.year.nunique()):]; seeds=list(range(7001,7021))
    for y in sample_years:
        g=panel[panel.year==y].reset_index(drop=True); sims=np.stack([simulate_loss_vector(g,s+int(y),cfg) for s in seeds]); exp=g.exposure_proxy.to_numpy(float); cost=sims*1000/np.maximum(exp[None,:],1e-9); r={"year":y,"mean_abs_calibration_error":float(np.mean(np.abs(cost.mean(0)-g.mhat.to_numpy(float)))),"delta_hat_mean":float(g.delta_hat.mean()),"delta_hat_sd":float(g.delta_hat.std()),"tail_ratio_sd":float(g.tail_mean_ratio.std()),"n_communes":len(g),"mc_replications":len(seeds),"effective_tail_quantiles":cfg.n_quantiles*(1-cfg.alpha),"environment":cfg.loss_environment_name}
        for tau in [x for x in cfg.loss_quantiles if x>=.90]: r[f"coverage_q{tau:.2f}"]=float(np.mean(cost<=g[f"qhat_{tau:.2f}"].to_numpy(float)[None,:]))
        rows.append(r)
    pd.DataFrame(rows).to_csv(ctab / "table4_risk_calibration.csv",index=False)
    # Table 7 combines stress robustness, selection-stage stability and runtime.
    robust=summarize_returns(returns,cfg); robust["panel"]="stress_performance"
    if not sel.empty:
        ss=sel[sel.selected.astype(bool)].groupby(["method","candidate_stage"]).size().rename("selected_count").reset_index(); ss["panel"]="selection_stability"
    else:ss=pd.DataFrame()
    rt=runtime_rows.groupby(["method","phase"],as_index=False).runtime_s.agg(["mean","std","count"]).reset_index(); rt["panel"]="runtime"
    out7=pd.concat([robust,ss,rt],ignore_index=True,sort=False); out7.to_csv(ctab / "table7_robustness_ablation_computation.csv",index=False)


def figure4_mean_tail_frontier(returns,cfg,paths):
    d=returns[returns.stress=="historical"].copy(); summ=summarize_returns(d,cfg)
    # seed-level uncertainty
    seedrows=[]
    for (m,s),g in d.groupby(["method","seed"]):
        r=g.G0.to_numpy(float); seedrows.append({"method":m,"seed":s,"mean_G0":r.mean(),"TVaR":empirical_tvar_np(-r,cfg.alpha)})
    sd=pd.DataFrame(seedrows); agg=sd.groupby("method").agg(mean_G0=("mean_G0","mean"),mean_G0_sd=("mean_G0","std"),TVaR_alpha_adverse=("TVaR","mean"),TVaR_sd=("TVaR","std")).reset_index(); agg.to_csv(paths["csv_figures"] / "figure4_mean_tail_frontier.csv",index=False)
    fig,ax=plt.subplots(figsize=(7.5,6))
    for _,r in agg.iterrows(): ax.errorbar(r.mean_G0,r.TVaR_alpha_adverse,xerr=r.mean_G0_sd,yerr=r.TVaR_sd,fmt='o',capsize=3,label=r.method)
    ax.set_xlabel("Mean cumulative actuarial return"); ax.set_ylabel(f"TVaR$_{{{cfg.alpha:.2f}}}$ of adverse cumulative return"); ax.set_title("Mean-return versus downside-tail policy frontier"); ax.grid(alpha=.2); ax.legend(); savefig(fig,paths["figures"] / "figure4_mean_tail_frontier",cfg)


def figure5_return_distribution(returns,cfg,paths):
    d=returns[returns.stress=="historical"].copy(); d["adverse_return"]=-d.G0; marks=[]
    for m,g in d.groupby("method"):
        x=g.adverse_return.to_numpy(float); marks.append({"method":m,"VaR":float(np.quantile(x,cfg.alpha,method='higher')),"TVaR":empirical_tvar_np(x,cfg.alpha)})
    pd.DataFrame(marks).to_csv(paths["csv_figures"] / "figure5_tail_markers.csv",index=False); d.to_csv(paths["csv_figures"] / "figure5_adverse_return_ecdf.csv",index=False)
    fig,ax=plt.subplots(figsize=(8,6))
    for m,g in d.groupby("method"):
        x=np.sort(g.adverse_return.to_numpy(float)); y=np.arange(1,len(x)+1)/max(len(x),1); ax.plot(x,y,label=m)
    ax.axhline(cfg.alpha,linestyle='--',linewidth=1); ax.set_xlabel("Adverse cumulative return $-G_0$"); ax.set_ylabel("Empirical CDF"); ax.set_title("Cumulative-return distribution and adverse tail"); ax.legend(); ax.grid(alpha=.2); savefig(fig,paths["figures"] / "figure5_adverse_return_distribution",cfg)


def rebuild_outputs(cfg:Config):
    paths=setup_paths(cfg); logger=setup_logger(paths["logs"]); rp=paths["processed"] / "policy_returns_all.csv"; sp=paths["processed"] / "policy_steps_all.csv"; pp=paths["processed"] / "commune_year_risk_panel_with_dependence_v12.parquet"; pcsv=paths["processed"] / "commune_year_risk_panel_with_dependence_v12.csv"
    if not (rp.exists() and sp.exists() and (pp.exists() or pcsv.exists())): raise FileNotFoundError("Need V11 processed returns, steps and risk panel")
    returns=pd.read_csv(rp); steps=pd.read_csv(sp); panel=pd.read_parquet(pp) if pp.exists() else pd.read_csv(pcsv,dtype={"code_commune":str}); write_tikz_figure1(paths); figure4_mean_tail_frontier(returns,cfg,paths); figure5_return_distribution(returns,cfg,paths); figure6_granularity_frontier(steps,returns,cfg,paths); figure7_dynamic_adaptation(steps,cfg,paths)
    # Figures 2--3 are reproducible when saved geometry/snapshots are present.
    gp=paths["processed"] / "communes_geometry_v11.geojson"; fp=paths["processed"] / "final_partition_snapshots.csv"
    if gp.exists():
        gdf=gpd.read_file(gp); figure2_risk_atlas(gdf,panel,cfg,paths)
        if fp.exists():
            z=pd.read_csv(fp,dtype={"code_commune":str}); final={}
            for m,g in z.groupby("method"):
                p=Partition(dict(zip(g.code_commune.astype(str),g.zone.astype(int)))); r=g.groupby("zone").relativity.first().to_dict(); final[m]=(p,{int(k):float(v) for k,v in r.items()})
            figure3_zoning_maps(gdf,panel,final,cfg,paths)
    logger.info("V11 outputs rebuilt from saved artifacts without retraining")


# Wrap production pipeline to persist geometry used by Figures 2--3 and enrich manifest.
_v10_production_pipeline=production_pipeline

def production_pipeline(cfg:Config,smoke:bool=False):
    paths=setup_paths(cfg); logger=setup_logger(paths["logs"]); save_config(cfg,paths); logger.info("IME Tail-QR pipeline start | version=%s | smoke=%s",SCRIPT_VERSION,smoke); device=select_device(cfg,logger)
    if smoke:
        gdf,panel,adjacency=make_smoke_data(cfg,paths,logger); data_manifest=pd.DataFrame([{"source_name":"synthetic smoke grid","source_url":"","local_path":"","status":"smoke-only","model_role":"software validation","observed_or_constructed":"synthetic","downloaded_utc":utc_now(),"sha256":"","notes":"Not manuscript evidence"}])
        panel.to_csv(paths["processed"] / "commune_year_risk_panel_with_dependence_v12.csv",index=False)
        gdf[["code_commune","geometry"]].to_file(paths["processed"] / "communes_geometry_v11.geojson",driver="GeoJSON")
    else:
        if cfg.claims_file:
            raise RuntimeError("V11 baseline does not silently ingest claims_file. A claims extension requires explicit commune-year insured-loss AND exposure schema validation before use.")
        data_manifest=download_public_data(cfg,paths,logger); gdf=load_communes(paths,cfg,logger); pop=load_population(paths,logger); catnat=load_catnat_events(paths,logger); adjacency=build_adjacency(gdf,paths,cfg,logger); panel=build_commune_year_panel(gdf,pop,catnat,cfg,paths,logger); panel=add_dependence_summary(panel,adjacency,cfg,paths,logger); panel=validate_panel_for_training(panel,cfg,logger)
        # exact geometry backing for no-retraining rebuild of maps
        gdf[["code_commune","geometry"]].to_file(paths["processed"] / "communes_geometry_v11.geojson",driver="GeoJSON")
    returns,steps,train_logs,runtime,final_parts=run_experiment(gdf,panel,adjacency,cfg,paths,logger,device); write_signature_tables(cfg,paths,data_manifest,panel,returns,steps,train_logs,runtime,logger); build_all_figures(gdf,panel,returns,steps,final_parts,cfg,paths,logger)
    make_manifest(cfg,paths,logger,extra={"smoke_test":smoke,"v12_scientific_protocol":{"frozen_continuation":True,"independent_candidate_selection":True,"test_policy_frozen":True,"exact_empirical_tvar":True,"heterogeneous_tail_surface":True,"fast_mc_requires_loss_not_in_next_state":True,"effective_tail_quantiles":cfg.n_quantiles*(1-cfg.alpha)}}); logger.info("DONE | results=%s",paths["results"])



# -----------------------------------------------------------------------------
# V11 robustness: E0/E1 and genuine admissible granularity sweep
# -----------------------------------------------------------------------------

def _fixed_partition_path(part:Partition,panel:pd.DataFrame,years:Sequence[int],cfg:Config,adjacency,controlled_response:bool=True):
    by={int(y):g.reset_index(drop=True) for y,g in panel[panel.year.isin(years[:cfg.horizon])].groupby('year',sort=True)}
    first=by[int(years[0])]; prev_part=initial_department_partition(first); prev_rel=indicated_relativities(prev_part,first); expmult=np.ones(len(first),dtype=float); recs=[]
    for tt,y in enumerate(years[:cfg.horizon]):
        ydf=by[int(y)]; rel=indicated_relativities(part,ydf); a=Action(f'FIXED_K{part.k}_Y{y}','KEEP',part,rel); dc=decision_cost(a,prev_part,prev_rel,ydf,cfg); ctx=_year_arrays(ydf,cfg); mu=float(np.dot(ctx['exposure'],ctx['mhat'])/1000.0); loading=cfg.loading_rate*mu
        recs.append({'year':int(y),'action_id':a.action_id,'edit_type':'KEEP','K':part.k,'ydf':ydf,'exposure_mult':expmult.copy(),'mu_hat_total':mu,'loading':loading,**dc})
        if controlled_response:
            z=_zone_assignment(part,ydf).astype(np.int64); valid=z>=0; rloc=np.ones(len(z))
            if valid.any():
                rv=np.ones(int(z[valid].max())+1)
                for k,v in rel.items():
                    if 0<=int(k)<len(rv):rv[int(k)]=float(v)
                rloc[valid]=rv[z[valid]]
            expmult=np.clip(expmult*np.exp(-cfg.exposure_elasticity*(rloc-1.0)),.80,1.20)
        prev_part=part; prev_rel=rel
    return recs


def _unique_edit_actions(actions,edit_type):
    out=[]; seen=set()
    for a in actions:
        if a.edit_type!=edit_type:continue
        desc=a.lazy_descriptor if a.lazy_descriptor is not None else a.action_id.split('|PRICE=')[0]
        key=repr(desc)
        if key in seen:continue
        seen.add(key);out.append(a)
    return out


def _granularity_chain(panel,adjacency,year,cfg,e_min,logger):
    ydf=panel[panel.year==year].reset_index(drop=True); base=initial_department_partition(ydf); rel0=indicated_relativities(base,ydf); records={base.k:(base,rel0)}
    # local sweep config does not alter the RL experiment; it only enumerates admissible comparator partitions.
    cg=copy.deepcopy(cfg); cg.e_min=float(e_min); cg.m_geographic_total=max(30,cfg.m_geographic_total); cg.max_action_set=max(90,cfg.max_action_set); cg.m_merge_total=max(20,cfg.m_merge_total); cg.m_split_target_zones=max(15,cfg.m_split_target_zones)
    # refinement chain
    part=base; rel=rel0
    for _ in range(max(0,min(cg.k_max,110)-base.k)):
        acts=generate_action_set(part,rel,ydf,adjacency,cg); opts=_unique_edit_actions(acts,'SPLIT')
        if not opts:break
        best=min(opts,key=lambda a:float((a.lazy_metrics or _action_metric_bundle(a,part,rel,ydf)).get('psi_z',np.inf)))
        part=_materialize_action_if_needed(best,part); rel=indicated_relativities(part,ydf); records[part.k]=(part,rel)
    # coarsening chain to at most K=86
    part=base; rel=rel0
    for _ in range(max(0,base.k-86)):
        acts=generate_action_set(part,rel,ydf,adjacency,cg); opts=_unique_edit_actions(acts,'MERGE')
        if not opts:break
        best=min(opts,key=lambda a:float((a.lazy_metrics or _action_metric_bundle(a,part,rel,ydf)).get('psi_z',np.inf)))
        part=_materialize_action_if_needed(best,part); rel=indicated_relativities(part,ydf); records[part.k]=(part,rel)
    logger.info('GRANULARITYCHAIN | E_min=%.0f K_range=%s..%s n=%s',e_min,min(records),max(records),len(records)); return records


def run_granularity_sweep(panel,adjacency,test_years,cfg,paths,logger):
    year=int(test_years[0]); egrid=sorted(set([max(500.0,cfg.e_min*.6),cfg.e_min,cfg.e_min*2.0])); targets=[88,92,96,100,104,108]; rows=[]; nmc=max(20,int(math.ceil(cfg.n_stress_trajectories/max(len(cfg.seeds),1))))
    noise=[44_000_000+j for j in range(nmc)]
    for emin in egrid:
        chain=_granularity_chain(panel,adjacency,year,cfg,emin,logger)
        ks=sorted(chain)
        for target in targets:
            if not ks:continue
            k=min(ks,key=lambda x:abs(x-target)); part,_=chain[k]; path=_fixed_partition_path(part,panel,test_years,cfg,adjacency,controlled_response=cfg.controlled_response_baseline); ret,_=evaluate_policy_path('Granularity sweep',path,noise,cfg,'historical',0,k,collect_steps=False)
            ydf=panel[panel.year==year].reset_index(drop=True); rows.append({'E_min':float(emin),'target_K':target,'K':part.k,'mean_G0':float(ret.mean()),'TVaR_alpha_adverse':empirical_tvar_np(-ret,cfg.alpha),'J_alpha_lambda':float(ret.mean()-cfg.lambda_tail*empirical_tvar_np(-ret,cfg.alpha)),'psi_z_t0':psi_z(part,ydf),'uncertainty_t0':uncertainty_c(part,ydf),'n_mc':len(ret)})
    out=pd.DataFrame(rows).drop_duplicates(['E_min','K']); out.to_csv(paths['processed'] / 'granularity_sweep.csv',index=False); return out


def figure6_granularity_frontier(steps,returns,cfg,paths):
    gp=paths['processed'] / 'granularity_sweep.csv'
    if gp.exists():
        m=pd.read_csv(gp); m.to_csv(paths['csv_figures'] / 'figure6_granularity_frontier.csv',index=False); fig,ax=plt.subplots(figsize=(8,6))
        for emin,g in m.groupby('E_min'):
            g=g.sort_values('K'); ax.plot(g.K,g.J_alpha_lambda,marker='o',label=f'$E_{{min}}$={emin:.0f}')
        ax.set_xlabel('Number of geographic zones $K$'); ax.set_ylabel('Mean minus tail penalty on cumulative return'); ax.set_title('Actuarial granularity frontier under admissible pooling constraints'); ax.legend(); ax.grid(alpha=.2); savefig(fig,paths['figures'] / 'figure6_granularity_frontier',cfg); return
    # defensive fallback only for legacy rebuilds
    s=steps.groupby(['method','stress','seed']).agg(K=('K','mean'),adjustment=('gamma_p','mean'),turnover=('z_delta','mean')).reset_index(); r=returns.groupby(['method','stress','seed'],as_index=False).G0.mean(); m=s.merge(r,on=['method','stress','seed']); m['burden']=cfg.c_p*m.adjustment+cfg.c_z*m.turnover; m['adverse']=-m.G0; m.to_csv(paths['csv_figures'] / 'figure6_granularity_frontier.csv',index=False); fig,ax=plt.subplots(figsize=(8,6));
    for method,g in m[m.stress=='historical'].groupby('method'):ax.scatter(g.burden,g.adverse,label=method)
    ax.legend();savefig(fig,paths['figures'] / 'figure6_granularity_frontier',cfg)


# Replace run_experiment once more to add E0 robustness and the granularity sweep.
_v11_run_experiment_core=run_experiment

def run_experiment(gdf,panel,adjacency,cfg,paths,logger,device):
    returns,steps,train_logs,runtime_df,final_parts=_v11_run_experiment_core(gdf,panel,adjacency,cfg,paths,logger,device)
    _,_,test_years=years_by_split(cfg,panel)
    # E0 nested benchmark: selected policies are held fixed, but tariff decisions no longer
    # feed exposure multipliers into future losses. No retraining or reselection occurs.
    selp=paths['processed'] / 'candidate_selection.csv'; sel=pd.read_csv(selp); chosen=sel[sel.selected.astype(bool)] if not sel.empty else pd.DataFrame(); extra_r=[]; extra_s=[]; n_per=max(1,int(math.ceil(cfg.n_stress_trajectories/max(len(cfg.seeds),1))))
    for si,seed in enumerate(cfg.seeds):
        noise=[35_000_000+si*100000+j for j in range(n_per)]
        for method in METHODS:
            if method=='Static Zoning':net=None;stage=0
            else:
                row=chosen[(chosen.method==method)&(chosen.seed==seed)]
                if row.empty:continue
                stage=int(row.candidate_stage.iloc[0]); mp=paths['models'] / f"{method.lower().replace(' ','_').replace('-','_')}_seed{seed}_stage{stage}.pt"; ck=torch.load(mp,map_location='cpu'); cand=PolicyCandidate(method,seed,stage,int(ck['state_dim']),int(ck['action_dim']),int(ck['n_out']),'tail' if method=='Tail-QR Zoning' else 'mean',ck['state_dict']);net=_candidate_net(cand,cfg,device)
            # build selected action path under E1, then remove the exposure feedback for E0 loss simulation
            path,_,_=build_policy_path(method,net,panel,adjacency,test_years,cfg,device,'historical')
            for r in path:r['exposure_mult']=np.ones_like(r['exposure_mult'])
            ret,st=evaluate_policy_path(method,path,noise,cfg,'exogenous_transition',seed,stage,collect_steps=True)
            for rep,G in enumerate(ret):extra_r.append({'trajectory_id':35_000_000+si*100000+rep,'method':method,'seed':seed,'rep':rep,'stress':'exogenous_transition','G0':float(G),'selected_stage':stage})
            extra_s.extend(st)
            if net is not None:del net
    if extra_r:
        returns=pd.concat([returns,pd.DataFrame(extra_r)],ignore_index=True);steps=pd.concat([steps,pd.DataFrame(extra_s)],ignore_index=True);returns.to_csv(paths['processed'] / 'policy_returns_all.csv',index=False);steps.to_csv(paths['processed'] / 'policy_steps_all.csv',index=False)
    run_granularity_sweep(panel,adjacency,test_years,cfg,paths,logger)
    return returns,steps,train_logs,runtime_df,final_parts


# Canonical Table 7 is the combined robustness/ablation/computation table only.
_v11_write_tables_core=write_signature_tables

def write_signature_tables(cfg,paths,data_manifest,panel,returns,steps,train_logs,runtime_rows,logger):
    _v11_write_tables_core(cfg,paths,data_manifest,panel,returns,steps,train_logs,runtime_rows,logger)
    legacy=paths['csv_tables'] / 'table7_computational_diagnostics.csv'
    if legacy.exists():legacy.unlink()



# =============================================================================
# V12 FINAL SCIENTIFIC OVERRIDES
# =============================================================================
# The V11 results revealed a genuine identification/design problem rather than a
# plotting problem: mean and tail policies saw almost the same effective action
# information and the finite action screen was predominantly mean-risk driven.
# V12 makes the tail-sensitive control problem identifiable without inserting a
# tail term into the one-period payoff.  The changes are: (i) exact marginal
# calibration of qhat_tau to the loss simulator, (ii) tail-aware but still finite
# geographic screening, (iii) tail-shape pricing perturbations, (iv) action
# features that retain the alignment between pricing and tail geography, and
# (v) monetary-equivalent decision-cost coefficients with no hidden reward scale.

_V11_build_commune_year_panel = build_commune_year_panel
_V11_add_dependence_summary = add_dependence_summary
_V11_generate_geographic_candidates = generate_geographic_candidates


def _v12_exact_quantile_surface(panel: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Recalibrate qhat_tau to the exact marginal multiplier used by simulate_loss_vector.

    V11 multiplied componentwise quantiles, which is conservative for products and
    generated over-coverage.  V12 evaluates the product distribution directly on a
    deterministic (sigma, tail-weight) grid and maps each commune to the nearest cell.
    This changes only Rhat_t calibration; it does not use future observations.
    """
    out = panel.copy()
    sig = out["local_sigma"].to_numpy(float)
    ww = out["local_tail_weight"].to_numpy(float)
    sg = np.linspace(max(0.03, float(np.nanmin(sig))), float(np.nanmax(sig)), 19)
    wg = np.linspace(max(0.01, float(np.nanmin(ww))), float(np.nanmax(ww)), 19)
    taus = np.asarray(cfg.loss_quantiles, dtype=float)
    rng = np.random.default_rng(9132026)
    nmc = 80_000
    zc = rng.standard_normal(nmc)
    zi = rng.standard_normal(nmc)
    tp = np.maximum(rng.standard_t(df=max(cfg.tail_df,2.1), size=nmc), 0.0)
    muplus = _student_positive_mean(cfg.tail_df)
    sigc = float(cfg.spatial_common_factor) * .20
    common = np.exp(sigc*zc - .5*sigc**2)
    qgrid = np.empty((len(sg),len(wg),len(taus)), dtype=float)
    for i,s in enumerate(sg):
        eps = np.exp(s*zi - .5*s*s)
        base = common*eps
        for j,w in enumerate(wg):
            tf = (1.0+w*tp)/(1.0+w*muplus)
            qgrid[i,j,:] = np.quantile(base*tf, taus)
    si = np.abs(sig[:,None]-sg[None,:]).argmin(axis=1)
    wi = np.abs(ww[:,None]-wg[None,:]).argmin(axis=1)
    m = out["mhat"].to_numpy(float)
    for k,tau in enumerate(taus):
        out[f"qhat_{tau:.2f}"] = m*qgrid[si,wi,k]
    qmax=f"qhat_{max(cfg.loss_quantiles):.2f}"
    out["tail_mean_ratio"] = out[qmax]/out["mhat"].clip(lower=1e-12)
    return out


def build_commune_year_panel(communes,population,catnat,cfg,paths,logger):
    cache = paths["processed"] / "commune_year_risk_panel_v12.parquet"
    if cache.exists() and not cfg.force_compute:
        z=pd.read_parquet(cache)
        required={"local_sigma","local_tail_weight","tail_mean_ratio","qhat_0.99"}
        if required.issubset(z.columns):
            logger.info("Loading V12 calibrated risk panel")
            return z
    z=_V11_build_commune_year_panel(communes,population,catnat,cfg,paths,logger)
    z=_v12_exact_quantile_surface(z,cfg)
    z.to_parquet(cache,index=False)
    logger.info("V12 exact marginal quantile calibration ready | tail_ratio_sd=%.4f",float(z.tail_mean_ratio.std()))
    return z


def add_dependence_summary(panel,adjacency,cfg,paths,logger):
    """V12 dependence panel: preserve the V12-recalibrated quantiles."""
    cache=paths["processed"] / "commune_year_risk_panel_with_dependence_v12.parquet"
    if cache.exists() and not cfg.force_compute:
        out=pd.read_parquet(cache)
        if {"delta_hat","neighbor_hazard","qhat_0.99"}.issubset(out.columns) and np.isfinite(out.delta_hat).all():
            logger.info("Loading V12 validated dependence panel"); return out
    logger.info("Constructing V12 spatial dependence summary Deltahat_t")
    out_parts=[]; diag=[]
    for year,gy in panel.groupby("year",sort=True):
        z=gy.copy(); h=finite_array(z.hazard_index.to_numpy(float),fill=0.0); risk=dict(zip(z.code_commune.astype(str),h))
        nh=[]
        for code,own in zip(z.code_commune.astype(str),h):
            vals=[risk[n] for n in adjacency.get(code,()) if n in risk]
            nh.append(float(np.mean(vals)) if vals else float(own))
        nh=np.asarray(nh); sx=float(np.std(h)); sy=float(np.std(nh))
        if len(h)>=3 and sx>1e-12 and sy>1e-12:
            hx=h-h.mean(); ny=nh-nh.mean(); den=math.sqrt(float(np.dot(hx,hx)*np.dot(ny,ny))); rho=float(np.dot(hx,ny)/den) if den>1e-18 else 0.0
        else: rho=0.0
        rho=float(np.clip(rho,-1,1)) if math.isfinite(rho) else 0.0
        z["neighbor_hazard"]=nh; z["delta_hat"]=np.clip(rho*(.5+.5*nh),-1.0,1.0)
        out_parts.append(z); diag.append({"year":int(year),"annual_neighbor_rho":rho,"delta_mean":float(z.delta_hat.mean()),"delta_sd":float(z.delta_hat.std())})
    out=pd.concat(out_parts,ignore_index=True); pd.DataFrame(diag).to_csv(paths["metadata"] / "dependence_diagnostics_v12.csv",index=False); out.to_parquet(cache,index=False)
    logger.info("V12 dependence ready | years=%s delta_range=[%.4f, %.4f]",out.year.nunique(),out.delta_hat.min(),out.delta_hat.max()); return out


def _v12_pricing_candidates_from_stats(zs: pd.DataFrame, cfg: Config) -> List[Dict[int,float]]:
    """Pricing menu with distinct mean-level and tail-shape perturbations.

    Aggregate premium normalization and the delta_r feasibility constraint are
    preserved exactly.  Tail-shape tilts use q99/m rather than q99 alone, so the
    finite action set can contain actions that differ in tail targeting even when
    mean risk is similar.
    """
    if zs.empty: return []
    indicated=dict(zip(zs.zone.astype(int),zs.indicated_relativity.astype(float)))
    weights=dict(zip(zs.zone.astype(int),zs.weight.astype(float)))
    out=[indicated]
    if len(indicated)<=1 or cfg.m_pricing_per_partition<=1: return out
    zones=sorted(indicated)
    q99={int(k):float(v) for k,v in zip(zs.zone,zs.q99_zone)}
    m={int(k):float(v) for k,v in zip(zs.zone,zs.mhat_zone)}
    tr={k:q99[k]/max(m[k],1e-12) for k in zones}
    def direction(metric):
        med=float(np.median([metric[k] for k in zones]))
        return {k:(1.0 if metric[k]>=med else -1.0) for k in zones}
    menus=[(direction(q99), cfg.h), (direction(tr), cfg.h), (direction(tr), -cfg.h)]
    for direc,level in menus:
        cand={k:indicated[k]*math.exp(level*direc[k]) for k in zones}
        cand=pricing_normalize(cand,weights)
        ok=all(abs(math.log(max(cand[k],1e-12)/max(indicated[k],1e-12)))<=cfg.delta_r+1e-9 for k in zones)
        if ok and not any(max(abs(cand[k]-x.get(k,cand[k])) for k in zones)<1e-10 for x in out): out.append(cand)
        if len(out)>=cfg.m_pricing_per_partition: break
    return out[:cfg.m_pricing_per_partition]


def _pricing_candidates_from_stats(zs: pd.DataFrame, cfg: Config) -> List[Dict[int,float]]:
    return _v12_pricing_candidates_from_stats(zs,cfg)


def pricing_candidates(part: Partition, year_df: pd.DataFrame, cfg: Config) -> List[Dict[int,float]]:
    return _v12_pricing_candidates_from_stats(zone_statistics(part,year_df),cfg)


def _v12_zone_tail_moments(part: Partition, year_df: pd.DataFrame) -> Tuple[Dict[str,np.ndarray],Dict[str,Any]]:
    ctx=_year_arrays(year_df,cfg=None)
    z=_zone_assignment(part,year_df).astype(np.int64); valid=z>=0
    q=ctx["q99"][valid]; m=ctx["mhat"][valid]; e=ctx["exposure"][valid]; zi=z[valid]
    qbar=float(np.dot(e,q)/max(e.sum(),1e-12)); tr=q/max(m,1e-12)
    # q99 relativity carries both level and local tail shape; tr isolates shape.
    qr=q/max(qbar,1e-12)
    k=int(zi.max())+1 if zi.size else 0
    def moments(r):
        es=np.bincount(zi,weights=e,minlength=k); er=np.bincount(zi,weights=e*r,minlength=k); er2=np.bincount(zi,weights=e*r*r,minlength=k)
        within=np.maximum(er2-er*er/np.maximum(es,1e-12),0.0)
        return {"e":es,"er":er,"er2":er2,"within":within}
    return moments(qr), {"ctx":ctx,"qrel":qr,"tailshape":tr,"valid_index":np.flatnonzero(valid)}


def generate_geographic_candidates(part,year_df,adjacency,cfg):
    """V12 diversified finite screen: preserve mean candidates and reserve tail slots."""
    base=_V11_generate_geographic_candidates(part,year_df,adjacency,cfg)
    keep=[x for x in base if x[0]=="KEEP"][:1]
    base_non=[x for x in base if x[0]!="KEEP"]
    ctx=_year_arrays(year_df,cfg); st=_partition_structure(part,adjacency,getattr(cfg,"structure_cache_size",8)); groups=st["groups"]
    zs=zone_statistics(part,year_df); zone_exp=dict(zip(zs.zone.astype(int),zs.exposure.astype(float)))
    moments=_zone_moments_for_screening(part,year_df)
    # Build a tail-relative context compatible with the exact shift formula.
    qbar=float(np.dot(ctx["exposure"],ctx["q99"])/max(ctx["total_e"],1e-12)); qrel=ctx["q99"]/max(qbar,1e-12)
    tail_ctx=dict(ctx); tail_ctx["local_rel"]=qrel
    zarr=_zone_assignment(part,year_df).astype(np.int64); k=int(zarr.max())+1
    e=ctx["exposure"]; valid=zarr>=0; zi=zarr[valid]
    es=np.bincount(zi,weights=e[valid],minlength=k); er=np.bincount(zi,weights=e[valid]*qrel[valid],minlength=k); er2=np.bincount(zi,weights=e[valid]*qrel[valid]**2,minlength=k)
    tail_mom={"e":es,"er":er,"er2":er2,"within":np.maximum(er2-er*er/np.maximum(es,1e-12),0.0)}
    idxmap=ctx["code_to_index"]
    extras=[]; seen={x[0] for x in base}
    # Tail-aware one-commune shifts.
    art={kk:_articulation_points_subset(groups.get(kk,[]),adjacency) for kk in groups}
    cand=[]
    for node,kf,kt in st["boundary"]:
        ii=idxmap.get(node)
        if ii is None or node in art.get(kf,set()) or len(groups.get(kf,[]))<=1: continue
        en=float(e[ii])
        if zone_exp.get(kf,0.0)-en<cfg.e_min: continue
        dt=_shift_delta_psi_exact(ii,kf,kt,tail_mom,tail_ctx)
        dm=_shift_delta_psi_exact(ii,kf,kt,moments,ctx)
        cand.append((dt,dm,node,kf,kt))
    for dt,dm,node,kf,kt in sorted(cand,reverse=True)[:8]:
        aid=f"SHIFT|unit={node}|from={kf}|to={kt}"
        if aid in seen: continue
        extras.append((aid,None,{"unit":node,"from":kf,"to":kt,"delta_psi_z":dm,"tail_screen_score":dt,"_lazy_descriptor":("SHIFT",node,kf,kt)})); seen.add(aid)
    # Tail-driven splits in zones with the largest q99-relative dispersion.
    zone_disp=[]
    for kk,nodes in groups.items():
        ids=[idxmap[n] for n in nodes if n in idxmap]
        if len(ids)>=4:
            ee=e[ids]; rr=qrel[ids]; mu=np.dot(ee,rr)/max(ee.sum(),1e-12); zone_disp.append((float(np.dot(ee,(rr-mu)**2)/max(ee.sum(),1e-12)),kk))
    qmap=dict(zip(ctx["codes"],ctx["q99"]))
    for _,kk in sorted(zone_disp,reverse=True)[:6]:
        sp=split_groups_two_seed(kk,groups[kk],qmap,adjacency)
        if sp is None: continue
        glo,ghi=sp; hi=np.asarray([idxmap[n] for n in ghi],dtype=int); par=np.asarray([idxmap[n] for n in groups[kk]],dtype=int)
        if hi.size==0 or hi.size>=par.size: continue
        ehi=float(e[hi].sum()); epar=float(e[par].sum()); elo=epar-ehi
        if min(ehi,elo)<cfg.e_min: continue
        # Exact mean Psi gain for lazy accounting.
        pe=e[par]; pr=ctx["local_rel"][par]; old=max(float(np.dot(pe,pr*pr)-np.dot(pe,pr)**2/max(pe.sum(),1e-12)),0.0)
        he=e[hi]; hr=ctx["local_rel"][hi]; wh=max(float(np.dot(he,hr*hr)-np.dot(he,hr)**2/max(he.sum(),1e-12)),0.0)
        mask=np.isin(par,hi,invert=True); lo=par[mask]; le=e[lo]; lr=ctx["local_rel"][lo]; wl=max(float(np.dot(le,lr*lr)-np.dot(le,lr)**2/max(le.sum(),1e-12)),0.0)
        dm=(old-wh-wl)/max(ctx["total_e"],1e-12)
        aid=f"SPLIT|zone={kk}|variant=tail12"
        if aid in seen: continue
        extras.append((aid,None,{"zone":kk,"variant":"tail12","delta_psi_z":dm,"tail_screen_score":float(_safe_q(qrel[par],.90)-_safe_q(qrel[par],.10)),"_lazy_descriptor":("SPLIT",int(kk),tuple(ghi))}));seen.add(aid)
    # Fixed finite budget: one KEEP, majority mean-screened, explicit tail quota.
    budget=max(1,int(cfg.m_geographic_total)); tail_slots=min(8,max(0,budget//3)); mean_slots=max(0,budget-len(keep)-tail_slots)
    out=keep+base_non[:mean_slots]+extras[:tail_slots]
    # Fill any unused quota deterministically from remaining base/extras.
    ids={x[0] for x in out}
    for x in base_non+extras:
        if len(out)>=budget: break
        if x[0] not in ids: out.append(x);ids.add(x[0])
    return out[:budget]


def _v12_action_stats(action: Action, year_df: pd.DataFrame) -> pd.DataFrame:
    if action.lazy_zone_stats is not None: return action.lazy_zone_stats
    if action.partition is None: return pd.DataFrame()
    return zone_statistics(action.partition,year_df)


def action_features(action,prev_part,prev_rel,year_df,cfg):
    """V12 action encoder retains tail targeting/alignment, not only aggregate costs."""
    m=_action_metric_bundle(action,prev_part,prev_rel,year_df)
    zs=_v12_action_stats(action,year_df)
    rel_vals=np.fromiter((float(v) for v in action.relativities.values()),dtype=float)
    if zs.empty:
        tail_sd=tail_align=q99_align=maxdev=0.0
    else:
        zones=zs.zone.astype(int).to_numpy(); w=zs.weight.to_numpy(float); mh=zs.mhat_zone.to_numpy(float); q=zs.q99_zone.to_numpy(float); tr=q/np.maximum(mh,1e-12)
        ind=zs.indicated_relativity.to_numpy(float); rr=np.asarray([float(action.relativities.get(int(k),ind[j])) for j,k in enumerate(zones)])
        dev=np.log(np.maximum(rr,1e-12)/np.maximum(ind,1e-12)); maxdev=float(np.max(np.abs(dev))) if dev.size else 0.0
        tail_sd=float(np.sqrt(np.dot(w,(tr-np.dot(w,tr))**2))) if tr.size else 0.0
        def align(x):
            xc=x-np.dot(w,x); sd=math.sqrt(max(float(np.dot(w,xc*xc)),1e-12)); return float(np.dot(w,(xc/sd)*dev))
        tail_align=align(tr); q99_align=align(q/np.maximum(np.dot(w,q),1e-12))
    edit=[float(action.edit_type==t) for t in ["KEEP","SPLIT","MERGE","SHIFT"]]
    arr=np.asarray([
        (action.lazy_k if action.lazy_k is not None else action.partition.k)/max(cfg.k_max,1),
        m["psi_z"],m["psi_p"],m["c_zonal"],m["gamma_p"],m["z_delta"],
        float(rel_vals.mean()) if rel_vals.size else 1.0,float(rel_vals.std()) if rel_vals.size else 0.0,
        tail_sd,tail_align,q99_align,maxdev,float(action.metadata.get("tail_screen_score",0.0)),*edit
    ],dtype=np.float32)
    return np.nan_to_num(arr,nan=0.0,posinf=1e6,neginf=-1e6).astype(np.float32)


# More informative diagnostics for the final paper.
_V11_write_signature_tables_final = write_signature_tables

def write_signature_tables(cfg,paths,data_manifest,panel,returns,steps,train_logs,runtime_rows,logger):
    _V11_write_signature_tables_final(cfg,paths,data_manifest,panel,returns,steps,train_logs,runtime_rows,logger)
    # Pairwise action-path divergence on the untouched historical test.
    h=steps[steps.stress.eq("historical")].copy()
    rows=[]
    if not h.empty:
        path=h.groupby(["method","seed","year"],as_index=False).first()
        methods=sorted(path.method.unique())
        for i,a in enumerate(methods):
            for b in methods[i+1:]:
                aa=path[path.method.eq(a)][["seed","year","action_id","K"]].rename(columns={"action_id":"a","K":"Ka"})
                bb=path[path.method.eq(b)][["seed","year","action_id","K"]].rename(columns={"action_id":"b","K":"Kb"})
                z=aa.merge(bb,on=["seed","year"])
                rows.append({"method_a":a,"method_b":b,"n_state_years":len(z),"action_disagreement_rate":float(np.mean(z.a!=z.b)) if len(z) else np.nan,"K_disagreement_rate":float(np.mean(z.Ka!=z.Kb)) if len(z) else np.nan})
    pd.DataFrame(rows).to_csv(paths["csv_tables"] / "policy_action_divergence.csv",index=False)



# =============================================================================
# V13 FINAL ANALYSIS OVERRIDES
# =============================================================================
# V13 leaves the V12.1 baseline training problem unchanged.  It adds two
# reviewer-facing analyses that were not identified cleanly in V12.1:
#   (i) a direct admissible granularity chain that does not depend on the RL
#       top-M action screen, and
#   (ii) post-training preference sensitivity over (alpha, lambda) and tail
#        severity, using the frozen quantile candidate family and independent
#        selection/test blocks.  The latter is deliberately labelled as a
#        post-training sensitivity, not as full retraining for each preference.

def _v13_direct_best_split(
    part: Partition,
    year_df: pd.DataFrame,
    adjacency: Mapping[str, Sequence[str]],
    cfg: Config,
) -> Optional[Tuple[Partition, float, str]]:
    """Return the feasible split with the largest exact reduction in Psi^Z.

    This enumerates all current zones and both mean/q99 two-seed cuts.  It is
    used only for the comparator granularity path, not for RL action selection.
    """
    if part.k >= cfg.k_max:
        return None
    ctx = _year_arrays(year_df, cfg)
    groups = part.zones()
    idx = ctx["code_to_index"]
    e = ctx["exposure"]
    r = ctx["local_rel"]
    total_e = max(float(ctx["total_e"]), 1e-12)
    risk_maps = [
        ("mean", dict(zip(ctx["codes"], ctx["mhat"]))),
        ("q99", dict(zip(ctx["codes"], ctx["q99"]))),
    ]
    best = None
    for zone, nodes in sorted(groups.items()):
        if len(nodes) < 4:
            continue
        par = np.asarray([idx[n] for n in nodes if n in idx], dtype=np.int64)
        if par.size < 4:
            continue
        pe = e[par]; pr = r[par]
        p_e = float(pe.sum())
        p_er = float(np.dot(pe, pr))
        p_er2 = float(np.dot(pe, pr * pr))
        old = max(p_er2 - p_er * p_er / max(p_e, 1e-12), 0.0)
        for tag, rmap in risk_maps:
            sp = split_groups_two_seed(int(zone), nodes, rmap, adjacency)
            if sp is None:
                continue
            glo, ghi = sp
            hi = np.asarray([idx[n] for n in ghi if n in idx], dtype=np.int64)
            if hi.size == 0 or hi.size >= par.size:
                continue
            he = e[hi]; hr = r[hi]
            e_hi = float(he.sum()); e_lo = p_e - e_hi
            if min(e_hi, e_lo) < float(cfg.e_min):
                continue
            er_hi = float(np.dot(he, hr)); er2_hi = float(np.dot(he, hr * hr))
            er_lo = p_er - er_hi; er2_lo = p_er2 - er2_hi
            w_hi = max(er2_hi - er_hi * er_hi / max(e_hi, 1e-12), 0.0)
            w_lo = max(er2_lo - er_lo * er_lo / max(e_lo, 1e-12), 0.0)
            dpsi = float((old - w_hi - w_lo) / total_e)
            key = (dpsi, -int(zone), tag)
            if best is None or key > best[0]:
                best = (key, int(zone), tuple(ghi), tag, dpsi)
    if best is None:
        return None
    _, zone, ghi, tag, dpsi = best
    new = _materialize_deferred_edit(part, ("SPLIT", int(zone), tuple(ghi)))
    if not partition_admissible(new, year_df, adjacency, cfg):
        raise RuntimeError("V13 direct split produced an inadmissible partition")
    return new, float(dpsi), str(tag)


def _v13_direct_best_merge(
    part: Partition,
    year_df: pd.DataFrame,
    adjacency: Mapping[str, Sequence[str]],
    cfg: Config,
) -> Optional[Tuple[Partition, float, Tuple[int, int]]]:
    """Return adjacent merge with smallest exact increase in Psi^Z."""
    if part.k <= 1:
        return None
    zs = zone_statistics(part, year_df).set_index("zone")
    total_e = max(float(zs["exposure"].sum()), 1e-12)
    best = None
    for ka, kb in zone_neighbor_pairs(part, adjacency):
        if ka not in zs.index or kb not in zs.index:
            continue
        ea = float(zs.loc[ka, "exposure"]); eb = float(zs.loc[kb, "exposure"])
        ra = float(zs.loc[ka, "indicated_relativity"])
        rb = float(zs.loc[kb, "indicated_relativity"])
        increase = float((ea * eb / max(ea + eb, 1e-12)) * (ra - rb) ** 2 / total_e)
        key = (increase, int(ka), int(kb))
        if best is None or key < best[0]:
            best = (key, int(ka), int(kb), increase)
    if best is None:
        return None
    _, ka, kb, increase = best
    new = merge_partition(part, ka, kb)
    if not partition_admissible(new, year_df, adjacency, cfg):
        raise RuntimeError("V13 direct merge produced an inadmissible partition")
    return new, float(increase), (ka, kb)


def _granularity_chain(panel, adjacency, year, cfg, e_min, logger):
    """V13 direct greedy admissible chain, independent of RL screening."""
    ydf = panel[panel.year == year].reset_index(drop=True)
    base = initial_department_partition(ydf)
    cg = copy.deepcopy(cfg)
    cg.e_min = float(e_min)
    records: Dict[int, Tuple[Partition, Dict[int, float]]] = {
        base.k: (base, indicated_relativities(base, ydf))
    }

    part = base
    while part.k < min(int(cg.k_max), 110):
        nxt = _v13_direct_best_split(part, ydf, adjacency, cg)
        if nxt is None:
            break
        part, _, _ = nxt
        records[part.k] = (part, indicated_relativities(part, ydf))

    part = base
    while part.k > 86:
        nxt = _v13_direct_best_merge(part, ydf, adjacency, cg)
        if nxt is None:
            break
        part, _, _ = nxt
        records[part.k] = (part, indicated_relativities(part, ydf))

    logger.info(
        "V13 GRANULARITYCHAIN | E_min=%.0f K_range=%s..%s n=%s",
        e_min, min(records), max(records), len(records)
    )
    return records


def run_granularity_sweep(panel, adjacency, test_years, cfg, paths, logger):
    year = int(test_years[0])
    egrid = sorted(set([max(500.0, cfg.e_min * .6), cfg.e_min, cfg.e_min * 2.0]))
    targets = [88, 92, 96, 100, 104, 108]
    rows = []
    nmc = max(40, int(math.ceil(cfg.v13_granularity_mc_total / max(len(cfg.seeds), 1))))
    noise = [44_000_000 + j for j in range(nmc)]
    for emin in egrid:
        chain = _granularity_chain(panel, adjacency, year, cfg, emin, logger)
        ks = sorted(chain)
        for target in targets:
            if not ks:
                continue
            k = min(ks, key=lambda x: (abs(x - target), x))
            part, _ = chain[k]
            path = _fixed_partition_path(
                part, panel, test_years, cfg, adjacency,
                controlled_response=cfg.controlled_response_baseline
            )
            ret, _ = evaluate_policy_path(
                "Granularity sweep", path, noise, cfg, "historical", 0, k,
                collect_steps=False
            )
            ydf = panel[panel.year == year].reset_index(drop=True)
            tv = empirical_tvar_np(-ret, cfg.alpha)
            rows.append({
                "E_min": float(emin),
                "target_K": int(target),
                "K": int(part.k),
                "mean_G0": float(ret.mean()),
                "TVaR_alpha_adverse": float(tv),
                "J_alpha_lambda": float(ret.mean() - cfg.lambda_tail * tv),
                "psi_z_t0": float(psi_z(part, ydf)),
                "uncertainty_t0": float(uncertainty_c(part, ydf)),
                "n_mc": int(len(ret)),
                "frontier_type": "greedy_admissible_direct",
            })
    out = pd.DataFrame(rows).drop_duplicates(["E_min", "K"]).sort_values(["E_min", "K"])
    out.to_csv(paths["processed"] / "granularity_sweep.csv", index=False)
    out.to_csv(paths["csv_figures"] / "figure6_granularity_frontier.csv", index=False)
    return out


def _v13_load_candidate(method, seed, stage, cfg, paths, device) -> PolicyCandidate:
    mp = paths["models"] / f"{method.lower().replace(' ','_').replace('-','_')}_seed{seed}_stage{stage}.pt"
    if not mp.exists():
        raise FileNotFoundError(f"Missing V13 candidate checkpoint: {mp}")
    ck = torch.load(mp, map_location="cpu")
    return PolicyCandidate(
        method, int(seed), int(stage), int(ck["state_dim"]), int(ck["action_dim"]),
        int(ck["n_out"]), "tail" if method == "Tail-QR Zoning" else "mean",
        ck["state_dict"]
    )


def _v13_selected_stage_table(paths: Mapping[str, Path]) -> pd.DataFrame:
    p = paths["processed"] / "candidate_selection.csv"
    if not p.exists():
        raise FileNotFoundError(
            f"Missing {p}. Run the baseline V12.1/V13 training first."
        )
    x = pd.read_csv(p)
    if "selected" not in x:
        raise RuntimeError("candidate_selection.csv has no selected column")
    mask = x["selected"].astype(str).str.lower().isin(["true", "1", "yes"])
    return x.loc[mask].copy()


def _v13_test_noise(cfg: Config, seed_index: int, total: int) -> List[int]:
    n = max(20, int(math.ceil(total / max(len(cfg.seeds), 1))))
    return [55_000_000 + seed_index * 100_000 + j for j in range(n)]


def run_v13_preference_sensitivity(
    panel: pd.DataFrame,
    adjacency: Mapping[str, Sequence[str]],
    cfg: Config,
    paths: Mapping[str, Path],
    logger: logging.Logger,
    device: torch.device,
) -> pd.DataFrame:
    """Post-training (alpha, lambda, tail-severity) sensitivity.

    Candidate evaluators are frozen. For each (alpha, lambda), Tail-QR candidate
    stages are re-ranked on the independent selection years using that preference,
    then the selected stage is evaluated on the untouched test years. QR Mean,
    Mean DQN and Static retain their independently selected baseline policies.
    This is not claimed to equal full retraining at each preference.
    """
    _, sel_years, test_years = years_by_split(cfg, panel)
    selected = _v13_selected_stage_table(paths)
    rows = []
    stage_rows = []

    # Baseline stages for mean-ranked policies do not depend on alpha/lambda.
    base_stage = {}
    for method in ["Mean DQN", "QR Mean"]:
        for seed in cfg.seeds:
            r = selected[(selected.method == method) & (selected.seed == seed)]
            if r.empty:
                raise RuntimeError(f"Missing selected baseline stage for {method}, seed={seed}")
            base_stage[(method, seed)] = int(r.candidate_stage.iloc[0])

    for alpha in [float(x) for x in cfg.v13_alpha_grid]:
        for lam in [float(x) for x in cfg.v13_lambda_grid]:
            sens_cfg = copy.deepcopy(cfg)
            sens_cfg.alpha = alpha
            sens_cfg.lambda_tail = lam
            for si, seed in enumerate(cfg.seeds):
                # Tail-QR: re-rank the frozen candidate family on selection data.
                nsel = max(30, int(math.ceil(cfg.v13_sensitivity_mc_total / max(len(cfg.seeds), 1))))
                sel_noise = [54_000_000 + si * 100_000 + j for j in range(nsel)]
                best_stage = None
                best_j = -np.inf
                for stage in range(1, int(cfg.evaluation_stages) + 1):
                    cand = _v13_load_candidate("Tail-QR Zoning", seed, stage, sens_cfg, paths, device)
                    net = _candidate_net(cand, sens_cfg, device)
                    path, _, _ = build_policy_path(
                        "Tail-QR Zoning", net, panel, adjacency, sel_years,
                        sens_cfg, device, "historical"
                    )
                    ret, _ = evaluate_policy_path(
                        "Tail-QR Zoning", path, sel_noise, sens_cfg,
                        "historical", seed, stage, collect_steps=False
                    )
                    mu = float(ret.mean())
                    tv = float(empirical_tvar_np(-ret, alpha))
                    j = float(mu - lam * tv)
                    stage_rows.append({
                        "alpha": alpha, "lambda": lam, "seed": int(seed),
                        "candidate_stage": int(stage), "mean_G0_sel": mu,
                        "TVaR_alpha_adverse_sel": tv, "J_sel": j,
                    })
                    if j > best_j:
                        best_j = j
                        best_stage = stage
                    del net
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

                test_noise = _v13_test_noise(cfg, si, cfg.v13_sensitivity_mc_total)
                for tail_mult in [float(x) for x in cfg.v13_tail_multiplier_grid]:
                    eval_cfg = copy.deepcopy(sens_cfg)
                    eval_cfg.stress_tail_multiplier = tail_mult
                    stress = "historical" if abs(tail_mult - 1.0) < 1e-12 else "tail_amplification"

                    # Static + mean-ranked baselines.
                    for method in ["Static Zoning", "Mean DQN", "QR Mean", "Tail-QR Zoning"]:
                        if method == "Static Zoning":
                            net = None
                            stage = 0
                        elif method == "Tail-QR Zoning":
                            stage = int(best_stage)
                            cand = _v13_load_candidate(method, seed, stage, eval_cfg, paths, device)
                            net = _candidate_net(cand, eval_cfg, device)
                        else:
                            stage = int(base_stage[(method, seed)])
                            cand = _v13_load_candidate(method, seed, stage, eval_cfg, paths, device)
                            net = _candidate_net(cand, eval_cfg, device)

                        path, _, _ = build_policy_path(
                            method, net, panel, adjacency, test_years,
                            eval_cfg, device, "historical"
                        )
                        ret, _ = evaluate_policy_path(
                            method, path, test_noise, eval_cfg, stress,
                            seed, stage, collect_steps=False
                        )
                        mu = float(ret.mean())
                        tv = float(empirical_tvar_np(-ret, alpha))
                        j = float(mu - lam * tv)
                        rows.append({
                            "alpha": alpha,
                            "lambda": lam,
                            "tail_multiplier": tail_mult,
                            "seed": int(seed),
                            "method": method,
                            "selected_stage": int(stage),
                            "mean_G0": mu,
                            "TVaR_alpha_adverse": tv,
                            "J_alpha_lambda": j,
                            "n_test": int(len(ret)),
                            "sensitivity_type": "post_training_frozen_candidate_family",
                        })
                        if net is not None:
                            del net
                            if device.type == "cuda":
                                torch.cuda.empty_cache()

    raw = pd.DataFrame(rows)
    stages = pd.DataFrame(stage_rows)
    raw.to_csv(paths["processed"] / "v13_preference_sensitivity_raw.csv", index=False)
    stages.to_csv(paths["processed"] / "v13_tail_candidate_selection_sensitivity.csv", index=False)

    # Aggregate across seeds and derive paired Tail-QR minus QR Mean contrasts.
    agg = raw.groupby(["alpha", "lambda", "tail_multiplier", "method"], as_index=False).agg(
        mean_G0=("mean_G0", "mean"),
        sd_mean_G0=("mean_G0", "std"),
        TVaR_alpha_adverse=("TVaR_alpha_adverse", "mean"),
        sd_TVaR=("TVaR_alpha_adverse", "std"),
        J_alpha_lambda=("J_alpha_lambda", "mean"),
        sd_J=("J_alpha_lambda", "std"),
        n_seeds=("seed", "nunique"),
    )
    agg.to_csv(paths["csv_tables"] / "v13_preference_sensitivity.csv", index=False)

    piv = raw.pivot_table(
        index=["alpha", "lambda", "tail_multiplier", "seed"],
        columns="method", values=["mean_G0", "TVaR_alpha_adverse", "J_alpha_lambda"]
    )
    contrast_rows = []
    for idx, rr in piv.iterrows():
        a, l, tm, seed = idx
        try:
            contrast_rows.append({
                "alpha": a, "lambda": l, "tail_multiplier": tm, "seed": int(seed),
                "delta_mean_tail_minus_qr": float(rr[("mean_G0", "Tail-QR Zoning")] - rr[("mean_G0", "QR Mean")]),
                "delta_tvar_tail_minus_qr": float(rr[("TVaR_alpha_adverse", "Tail-QR Zoning")] - rr[("TVaR_alpha_adverse", "QR Mean")]),
                "delta_J_tail_minus_qr": float(rr[("J_alpha_lambda", "Tail-QR Zoning")] - rr[("J_alpha_lambda", "QR Mean")]),
            })
        except KeyError:
            continue
    contrasts = pd.DataFrame(contrast_rows)
    contrasts.to_csv(paths["processed"] / "v13_tail_vs_qr_paired.csv", index=False)
    phase = contrasts.groupby(["alpha", "lambda", "tail_multiplier"], as_index=False).agg(
        delta_mean_tail_minus_qr=("delta_mean_tail_minus_qr", "mean"),
        delta_tvar_tail_minus_qr=("delta_tvar_tail_minus_qr", "mean"),
        delta_J_tail_minus_qr=("delta_J_tail_minus_qr", "mean"),
        sd_delta_J=("delta_J_tail_minus_qr", "std"),
        tail_qr_win_rate=("delta_J_tail_minus_qr", lambda x: float(np.mean(np.asarray(x) > 0))),
        n_seeds=("seed", "nunique"),
    )
    phase["preferred_by_J"] = np.where(
        phase["delta_J_tail_minus_qr"] > 0, "Tail-QR Zoning", "QR Mean"
    )
    phase.to_csv(paths["csv_figures"] / "v13_phase_diagram.csv", index=False)
    logger.info(
        "V13 sensitivity complete | cells=%s | Tail-QR preferred cells=%s/%s",
        len(phase), int((phase.preferred_by_J == "Tail-QR Zoning").sum()), len(phase)
    )
    return phase


def figure_v13_phase_diagram(cfg: Config, paths: Mapping[str, Path]) -> None:
    p = paths["csv_figures"] / "v13_phase_diagram.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    for tm in sorted(d.tail_multiplier.unique()):
        g = d[d.tail_multiplier == tm]
        mat = g.pivot(index="alpha", columns="lambda", values="delta_J_tail_minus_qr").sort_index()
        fig, ax = plt.subplots(figsize=(8, 5.5))
        im = ax.imshow(mat.to_numpy(), aspect="auto", origin="lower")
        ax.set_xticks(range(len(mat.columns)), [f"{x:g}" for x in mat.columns])
        ax.set_yticks(range(len(mat.index)), [f"{x:g}" for x in mat.index])
        ax.set_xlabel(r"Tail preference $\lambda$")
        ax.set_ylabel(r"Tail level $\alpha$")
        ax.set_title(
            rf"Tail-QR minus QR Mean criterion, tail multiplier={tm:g}"
        )
        cb = fig.colorbar(im, ax=ax)
        cb.set_label(r"$\Delta J_{\alpha,\lambda}$ (Tail-QR $-$ QR Mean)")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat.iloc[i, j]
                ax.text(j, i, f"{val/1e3:.1f}k", ha="center", va="center", fontsize=8)
        savefig(
            fig,
            paths["figures"] / f"v13_phase_diagram_tailmult_{str(tm).replace('.','p')}",
            cfg
        )


def run_v13_analysis_only(cfg: Config) -> None:
    """Reuse an existing V12.1/V13 baseline run; no RL retraining."""
    paths = setup_paths(cfg)
    logger = setup_logger(paths["logs"])
    logger.info("V13 analysis-only start | version=%s", SCRIPT_VERSION)
    device = select_device(cfg, logger)

    # Load/reconstruct public-data state using caches when available.
    gdf = load_communes(paths, cfg, logger)
    pop = load_population(paths, logger)
    catnat = load_catnat_events(paths, logger)
    adjacency = build_adjacency(gdf, paths, cfg, logger)
    panel = build_commune_year_panel(gdf, pop, catnat, cfg, paths, logger)
    panel = add_dependence_summary(panel, adjacency, cfg, paths, logger)
    panel = validate_panel_for_training(panel, cfg, logger)
    _, _, test_years = years_by_split(cfg, panel)

    run_granularity_sweep(panel, adjacency, test_years, cfg, paths, logger)
    phase = run_v13_preference_sensitivity(panel, adjacency, cfg, paths, logger, device)
    figure6_granularity_frontier(pd.DataFrame(), pd.DataFrame(), cfg, paths)
    figure_v13_phase_diagram(cfg, paths)
    make_manifest(cfg, paths, logger, extra={
        "v13_analysis_only": True,
        "v13_preference_sensitivity": "post-training frozen candidate family; no claim of full retraining at each alpha/lambda",
        "v13_phase_cells": int(len(phase)),
    })
    logger.info("V13 ANALYSIS DONE | results=%s", paths["results"])


# Wrap the full V13 pipeline so a fresh run also produces V13 analyses.
_v13_baseline_production_pipeline = production_pipeline

def production_pipeline(cfg: Config, smoke: bool=False) -> None:
    _v13_baseline_production_pipeline(cfg, smoke=smoke)
    if smoke:
        return
    paths = setup_paths(cfg)
    logger = setup_logger(paths["logs"])
    device = select_device(cfg, logger)
    gdf = load_communes(paths, cfg, logger)
    pop = load_population(paths, logger)
    catnat = load_catnat_events(paths, logger)
    adjacency = build_adjacency(gdf, paths, cfg, logger)
    panel = build_commune_year_panel(gdf, pop, catnat, cfg, paths, logger)
    panel = add_dependence_summary(panel, adjacency, cfg, paths, logger)
    panel = validate_panel_for_training(panel, cfg, logger)
    # The baseline run_experiment already produced the V13 direct granularity
    # sweep because the global override is active. Do not recompute it here.
    phase = run_v13_preference_sensitivity(panel, adjacency, cfg, paths, logger, device)
    figure6_granularity_frontier(pd.DataFrame(), pd.DataFrame(), cfg, paths)
    figure_v13_phase_diagram(cfg, paths)
    make_manifest(cfg, paths, logger, extra={
        "v13_post_training_analysis": True,
        "v13_preference_sensitivity": "post-training frozen candidate family; no claim of full retraining at each alpha/lambda",
        "v13_phase_cells": int(len(phase)),
    })
    logger.info("V13 post-training analyses DONE")



# -----------------------------------------------------------------------------
# 14. CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description="IME Tail-QR Zoning reproducible pipeline")
    mode=p.add_mutually_exclusive_group(required=False)
    mode.add_argument("--run-all",action="store_true",help="download/process/train/evaluate/figures/tables")
    mode.add_argument("--download-data",action="store_true",help="download public data only")
    mode.add_argument("--smoke-test",action="store_true",help="small offline synthetic software validation")
    mode.add_argument("--rebuild-outputs",action="store_true",help="rebuild selected outputs from saved CSVs")
    mode.add_argument("--v13-analysis-only",action="store_true",help="reuse existing trained V12.1/V13 models and run only V13 granularity + preference sensitivity")
    p.add_argument("--config",type=str,default=None,help="JSON config override")
    p.add_argument("--use-cuda",action="store_true",help="request CUDA")
    p.add_argument("--cpu",action="store_true",help="force CPU")
    p.add_argument("--force-download",action="store_true")
    p.add_argument("--force-compute",action="store_true")
    p.add_argument("--data-dir",type=str,default=None)
    p.add_argument("--results-dir",type=str,default=None)
    return p.parse_args()


def load_config(args: argparse.Namespace) -> Config:
    cfg=Config()
    if args.config:
        raw=json.loads(Path(args.config).read_text(encoding="utf-8"))
        known={f.name for f in cfg.__dataclass_fields__.values()}
        unknown=set(raw)-known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        for k,v in raw.items():
            setattr(cfg,k,v)
    if args.use_cuda:
        cfg.use_cuda=True
    if args.cpu:
        cfg.use_cuda=False
    if args.force_download:
        cfg.force_download=True
    if args.force_compute:
        cfg.force_compute=True
    if args.data_dir:
        cfg.data_dir=args.data_dir
    if args.results_dir:
        cfg.results_dir=args.results_dir
    if args.smoke_test:
        cfg.seeds=cfg.seeds[:1]
        cfg.train_steps_per_stage=min(3, cfg.smoke_train_steps)
        cfg.evaluation_stages=1
        cfg.n_test_trajectories=1
        cfg.n_selection_trajectories=2
        cfg.k_max=12
        cfg.e_min=500
        cfg.hidden_dim=64
        cfg.batch_size=32
        cfg.replay_capacity=2000
        cfg.warmup_steps=64
        cfg.target_refresh=50
        cfg.log_every=100
        cfg.max_action_set=4
        cfg.m_geographic_total=2
        cfg.m_shift_total=2
        cfg.m_merge_total=1
        cfg.m_pricing_per_partition=2
        cfg.horizon=min(2,cfg.smoke_years)
        # Synthetic years are 2015..; use strictly disjoint 4/2/2 blocks.
        smoke_start=2015
        cfg.train_end_year=smoke_start+3
        cfg.selection_end_year=smoke_start+5
        cfg.test_start_year=smoke_start+6
        if not args.data_dir:
            cfg.data_dir="Data_IME_SMOKE"
        if not args.results_dir:
            cfg.results_dir="Results_IME_SMOKE"
    return cfg


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    args=parse_args()
    cfg=load_config(args)
    if args.download_data:
        paths=setup_paths(cfg)
        logger=setup_logger(paths["logs"])
        save_config(cfg,paths)
        download_public_data(cfg,paths,logger)
        make_manifest(cfg,paths,logger,extra={"download_only":True})
        return
    if args.rebuild_outputs:
        rebuild_outputs(cfg)
        return
    if args.v13_analysis_only:
        run_v13_analysis_only(cfg)
        return
    if args.smoke_test:
        production_pipeline(cfg,smoke=True)
        return
    # Default to full pipeline if no explicit mode is given.
    production_pipeline(cfg,smoke=False)


if __name__ == "__main__":
    main()
