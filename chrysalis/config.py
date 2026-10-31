import dataclasses
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclasses.dataclass(frozen=True)
class Config:
    regions: int
    items_per_region: int
    fit_per_region: int
    alpha: float
    alpha_q: float
    calibration_queries: int
    k: int
    max_degree: int
    ef_construction: int
    ef_search: int
    prune_alpha: float
    bridge_rank: int
    ridge_grid: tuple
    ridge_folds: int
    resolve_limit: int
    resolution_mode: str
    sync_deadline_ms: float
    lease_seconds: float
    epoch_node_fraction: float
    epoch_edge_fraction: float
    tau: float
    segment_rows: int
    heat_half_life_requests: int
    scheduler_batch: int
    auditor_pool: int
    recent_window: int
    encode_batch: int
    build_threads: int
    gpu_price_per_hour: dict
    zipf_exponent: float
    stream_requests: int
    warmup_requests: int
    replay_seeds: tuple
    calibration_seeds: tuple
    seed: int

    @property
    def calibration_per_region(self):
        return self.items_per_region - self.fit_per_region

    def replace(self, **kw):
        return dataclasses.replace(self, **kw)

    def to_dict(self):
        return dataclasses.asdict(self)


def load(path=ROOT / "configs" / "paper.json", **overrides):
    raw = json.loads(Path(path).read_text())
    raw.update(overrides)
    for key in ("ridge_grid", "replay_seeds", "calibration_seeds"):
        raw[key] = tuple(raw[key])
    if raw["resolution_mode"] not in ("async", "sync", "off"):
        raise ValueError("resolution_mode must be async, sync, or off")
    if not 0 < raw["fit_per_region"] < raw["items_per_region"]:
        raise ValueError("fit_per_region must leave calibration pairs")
    return Config(**raw)


def models(path=ROOT / "configs" / "models.json"):
    return json.loads(Path(path).read_text())
