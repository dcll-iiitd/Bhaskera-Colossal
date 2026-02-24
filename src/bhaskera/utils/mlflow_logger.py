"""
MLflow logger for Bhaskera training.

Logs all training metrics + per-GPU hardware stats (memory, utilisation,
temperature, power draw) at every step.
"""
from __future__ import annotations

import subprocess
import xml.etree.ElementTree as ET
from typing import Dict, Any

import mlflow


# ---------------------------------------------------------------------------
# GPU stats helpers
# ---------------------------------------------------------------------------

def _query_nvidia_smi() -> list[dict]:
    """
    Call `nvidia-smi -q -x` and return a list of per-GPU stat dicts.
    Returns an empty list if nvidia-smi is unavailable or fails.

    Collected fields (all floats, units noted in the key name):
        gpu_util_pct          – GPU compute utilisation  [0-100]
        mem_util_pct          – Memory-controller utilisation  [0-100]
        mem_used_mib          – Used VRAM in MiB
        mem_free_mib          – Free VRAM in MiB
        mem_total_mib         – Total VRAM in MiB
        mem_used_pct          – mem_used / mem_total * 100
        temp_c                – GPU temperature in °C
        power_draw_w          – Instant power draw in Watts
        power_limit_w         – Enforced power limit in Watts
        power_pct             – power_draw / power_limit * 100
        fan_speed_pct         – Fan speed  [0-100]  (may be N/A → omitted)
        sm_clock_mhz          – Streaming-Multiprocessor clock in MHz
        mem_clock_mhz         – Memory clock in MHz
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "-q", "-x"],
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return []

    try:
        root = ET.fromstring(out)
    except ET.ParseError:
        return []

    results = []
    for gpu in root.findall("gpu"):
        def _float(path: str) -> float | None:
            el = gpu.find(path)
            if el is None or el.text is None:
                return None
            txt = el.text.strip().split()[0]   # strip unit suffix
            try:
                return float(txt)
            except ValueError:
                return None

        util     = gpu.find("utilization")
        fb       = gpu.find("fb_memory_usage")
        temp_el  = gpu.find("temperature")
        power_el = gpu.find("power_readings") or gpu.find("gpu_power_readings")
        clocks   = gpu.find("clocks")

        gpu_util  = _float("utilization/gpu_util")
        mem_util  = _float("utilization/memory_util")
        mem_used  = _float("fb_memory_usage/used")
        mem_free  = _float("fb_memory_usage/free")
        mem_total = _float("fb_memory_usage/total")
        temp_c    = _float("temperature/gpu_temp")
        pwr_draw  = None
        pwr_limit = None
        if power_el is not None:
            pwr_draw  = _float(f"{power_el.tag}/power_draw")
            pwr_limit = _float(f"{power_el.tag}/power_limit")
        fan       = _float("fan_speed")
        sm_clock  = _float("clocks/graphics_clock")
        mem_clock = _float("clocks/mem_clock")

        row: dict[str, float] = {}
        if gpu_util  is not None: row["gpu_util_pct"]    = gpu_util
        if mem_util  is not None: row["mem_util_pct"]    = mem_util
        if mem_used  is not None: row["mem_used_mib"]    = mem_used
        if mem_free  is not None: row["mem_free_mib"]    = mem_free
        if mem_total is not None: row["mem_total_mib"]   = mem_total
        if mem_used is not None and mem_total:
            row["mem_used_pct"] = round(mem_used / mem_total * 100, 2)
        if temp_c    is not None: row["temp_c"]          = temp_c
        if pwr_draw  is not None: row["power_draw_w"]    = pwr_draw
        if pwr_limit is not None: row["power_limit_w"]   = pwr_limit
        if pwr_draw is not None and pwr_limit:
            row["power_pct"] = round(pwr_draw / pwr_limit * 100, 2)
        if fan       is not None: row["fan_speed_pct"]   = fan
        if sm_clock  is not None: row["sm_clock_mhz"]   = sm_clock
        if mem_clock is not None: row["mem_clock_mhz"]  = mem_clock

        results.append(row)

    return results


def collect_gpu_metrics(prefix: str = "gpu") -> Dict[str, float]:
    """
    Return a flat dict of GPU metrics ready to pass to mlflow.log_metrics().

    Keys are formatted as  `<prefix>/<gpu_index>/<stat_name>`, e.g.::

        gpu/0/mem_used_mib
        gpu/0/gpu_util_pct
        gpu/1/temp_c
        ...

    A roll-up `gpu/all/<stat>` (mean across visible GPUs) is also included
    for stats that exist on every GPU.
    """
    per_gpu = _query_nvidia_smi()
    if not per_gpu:
        return {}

    flat: Dict[str, float] = {}
    # Per-device metrics
    for i, stats in enumerate(per_gpu):
        for k, v in stats.items():
            flat[f"{prefix}/{i}/{k}"] = v

    # Mean across all GPUs for each stat that appears on every device
    if len(per_gpu) > 1:
        all_keys = set(per_gpu[0].keys())
        for g in per_gpu[1:]:
            all_keys &= set(g.keys())
        for k in all_keys:
            flat[f"{prefix}/all/{k}"] = round(
                sum(g[k] for g in per_gpu) / len(per_gpu), 4
            )

    return flat


# ---------------------------------------------------------------------------
# MLflowLogger
# ---------------------------------------------------------------------------

class MLflowLogger:
    """
    Thin wrapper around the MLflow Python client.

    Config contract
    ---------------
    The ``cfg`` object must expose (or be duck-typed to provide):

    * ``PROJECT``  / ``cfg.project``   – MLflow experiment name
    * ``RUN_NAME`` / ``cfg.run_name``  – Human-readable run name
    * ``MLFLOW_TRACKING_URI``          – (optional) tracking server URI
    * ``as_dict()``                    – returns {str: any} of all hyper-params

    All three attribute spellings (UPPER, lower, camelCase) are tried so this
    works with both the legacy ``Config`` dataclass and the new YAML-loaded one.
    """

    def __init__(self, cfg, log_gpu: bool = True, gpu_log_every_n_steps: int = 1):
        """
        Args:
            cfg:                      Bhaskera config object.
            log_gpu:                  Whether to collect and log GPU stats.
            gpu_log_every_n_steps:    Log GPU stats every N optimizer steps
                                      (set >1 to reduce nvidia-smi overhead).
        """
        self.log_gpu = log_gpu
        self.gpu_log_every_n_steps = max(1, gpu_log_every_n_steps)
        self._step_count = 0  # internal counter used for GPU cadence

        # ---- resolve tracking URI ----------------------------------------
        tracking_uri = (
            getattr(cfg, "MLFLOW_TRACKING_URI", None)
            or getattr(cfg, "mlflow_tracking_uri", None)
        )
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        # ---- resolve experiment / run names --------------------------------
        experiment = (
            getattr(cfg, "PROJECT", None)
            or getattr(cfg, "project", None)
            or "bhaskera-training"
        )
        run_name = (
            getattr(cfg, "RUN_NAME", None)
            or getattr(cfg, "run_name", None)
            or "run"
        )

        mlflow.set_experiment(experiment)
        mlflow.start_run(run_name=run_name)

        # ---- log hyper-parameters ------------------------------------------
        if hasattr(cfg, "as_dict"):
            params = cfg.as_dict()
        elif hasattr(cfg, "__dict__"):
            params = {
                k: v for k, v in cfg.__dict__.items()
                if not k.startswith("_") and isinstance(v, (str, int, float, bool, type(None)))
            }
        else:
            params = {}

        # MLflow param values must be strings ≤ 500 chars
        for k, v in params.items():
            try:
                mlflow.log_param(k, str(v)[:500])
            except Exception:
                pass  # never crash training because of a logging hiccup

        # ---- log initial GPU hardware info ---------------------------------
        if self.log_gpu:
            self._log_gpu_hardware_info()

    # -----------------------------------------------------------------------
    # Public interface
    # -----------------------------------------------------------------------

    def log(self, metrics: Dict[str, Any], step: int) -> None:
        """
        Log a dict of scalar metrics.  GPU stats are appended automatically
        according to ``gpu_log_every_n_steps``.
        """
        self._step_count += 1

        # Merge GPU stats if it's time
        if self.log_gpu and (self._step_count % self.gpu_log_every_n_steps == 0):
            gpu_metrics = collect_gpu_metrics()
            metrics = {**metrics, **gpu_metrics}

        # Convert everything to float (MLflow only accepts numeric scalars)
        safe: Dict[str, float] = {}
        for k, v in metrics.items():
            try:
                safe[k] = float(v)
            except (TypeError, ValueError):
                pass

        if safe:
            try:
                mlflow.log_metrics(safe, step=step)
            except Exception:
                pass

    def finish(self) -> None:
        """End the MLflow run."""
        try:
            mlflow.end_run()
        except Exception:
            pass

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _log_gpu_hardware_info(self) -> None:
        """
        Log static GPU hardware info (name, driver, CUDA version, total VRAM)
        as MLflow *parameters* once at the start of the run.
        """
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,vbios_version,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                stderr=subprocess.DEVNULL,
                timeout=10,
            ).decode().strip()
        except (FileNotFoundError, subprocess.SubprocessError, OSError):
            return

        for i, line in enumerate(out.splitlines()):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                name, driver, vbios, mem_total = parts[:4]
                mlflow.log_param(f"gpu/{i}/name",         name)
                mlflow.log_param(f"gpu/{i}/driver",       driver)
                mlflow.log_param(f"gpu/{i}/vbios",        vbios)
                mlflow.log_param(f"gpu/{i}/vram_total_mib", mem_total)
