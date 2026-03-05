"""MLflow experiment logger with GPU metric collection."""
from __future__ import annotations

import subprocess
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

import mlflow


# ── GPU metrics via nvidia-smi XML ────────────────────────────────────────────

def _query_nvidia_smi() -> List[Dict[str, float]]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "-q", "-x"], stderr=subprocess.DEVNULL, timeout=10
        ).decode()
    except Exception:
        return []

    try:
        root = ET.fromstring(out)
    except ET.ParseError:
        return []

    results = []
    for gpu in root.findall("gpu"):
        def _float(path: str) -> Optional[float]:
            el = gpu.find(path)
            if el is None or el.text is None:
                return None
            try:
                return float(el.text.split()[0])
            except (ValueError, IndexError):
                return None

        power_el = gpu.find("power_readings") or gpu.find("gpu_power_readings")
        row: Dict[str, float] = {}
        for key, path in [
            ("gpu_util_pct",  "utilization/gpu_util"),
            ("mem_used_mib",  "fb_memory_usage/used"),
            ("mem_total_mib", "fb_memory_usage/total"),
            ("temp_c",        "temperature/gpu_temp"),
            ("fan_speed_pct", "fan_speed"),
            ("sm_clock_mhz",  "clocks/graphics_clock"),
        ]:
            v = _float(path)
            if v is not None:
                row[key] = v
        if power_el is not None:
            pw = _float(f"{power_el.tag}/power_draw")
            pl = _float(f"{power_el.tag}/power_limit")
            if pw is not None: row["power_draw_w"]  = pw
            if pl is not None: row["power_limit_w"] = pl
        if "mem_used_mib" in row and "mem_total_mib" in row and row["mem_total_mib"]:
            row["mem_used_pct"] = round(row["mem_used_mib"] / row["mem_total_mib"] * 100, 2)
        results.append(row)
    return results


def collect_gpu_metrics(prefix: str = "gpu") -> Dict[str, float]:
    per_gpu = _query_nvidia_smi()
    if not per_gpu:
        return {}
    flat: Dict[str, float] = {}
    for i, stats in enumerate(per_gpu):
        for k, v in stats.items():
            flat[f"{prefix}/{i}/{k}"] = v
    if len(per_gpu) > 1:
        common = set(per_gpu[0].keys())
        for g in per_gpu[1:]:
            common &= set(g.keys())
        for k in common:
            flat[f"{prefix}/all/{k}"] = round(sum(g[k] for g in per_gpu) / len(per_gpu), 4)
    return flat


# ── MLflowLogger ───────────────────────────────────────────────────────────────

class MLflowLogger:
    def __init__(self, cfg, log_gpu: bool = True, gpu_log_every_n_steps: int = 10):
        self.log_gpu               = log_gpu
        self.gpu_log_every_n_steps = max(1, gpu_log_every_n_steps)
        self._step_count           = 0

        tracking_uri = getattr(cfg, "MLFLOW_TRACKING_URI", None) or \
                       getattr(cfg, "mlflow_tracking_uri", None)
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        experiment = getattr(cfg, "PROJECT", None) or "bhaskera-training"
        run_name   = getattr(cfg, "RUN_NAME", None) or "run"

        mlflow.set_experiment(experiment)
        mlflow.start_run(run_name=run_name)

        params = cfg.as_dict() if hasattr(cfg, "as_dict") else {}
        for k, v in params.items():
            try:
                mlflow.log_param(k, str(v)[:500])
            except Exception:
                pass

        if self.log_gpu:
            self._log_gpu_hardware()

    def log(self, metrics: Dict[str, Any], step: int) -> None:
        self._step_count += 1
        if self.log_gpu and self._step_count % self.gpu_log_every_n_steps == 0:
            metrics = {**metrics, **collect_gpu_metrics()}
        safe = {}
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
        try:
            mlflow.end_run()
        except Exception:
            pass

    def _log_gpu_hardware(self) -> None:
        try:
            out = subprocess.check_output(
                ["nvidia-smi",
                 "--query-gpu=name,driver_version,vbios_version,memory.total",
                 "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL, timeout=10,
            ).decode().strip()
        except Exception:
            return
        for i, line in enumerate(out.splitlines()):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                name, driver, vbios, mem = parts[:4]
                mlflow.log_param(f"gpu/{i}/name",   name)
                mlflow.log_param(f"gpu/{i}/driver", driver)
                mlflow.log_param(f"gpu/{i}/vram_mib", mem)
