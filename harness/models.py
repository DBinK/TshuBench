"""数据结构定义。"""

from dataclasses import dataclass, field

WEIGHTS = {
    "cvd": 0.18,
    "hed": 0.22,
    "enu": 0.20,
    "ssf": 0.10,
    "af": 0.08,
    "cci": 0.06,
    "pni": 0.04,
    "vtr": 0.12,
}


@dataclass(frozen=True, slots=True)
class Task:
    id: str
    scenario: str
    register: str
    trap_type: str
    max_chars: int
    prompt: str
    verdict_anchors: list[str]
    actionable_entities: list[str]
    fence_sitting_markers: list[str] = field(default_factory=list)
    refutation_anchors: list[str] = field(default_factory=list)

    @property
    def answer_anchors(self) -> list[str]:
        if self.trap_type == "flawed_premise" and self.refutation_anchors:
            return self.refutation_anchors
        return self.verdict_anchors


@dataclass(frozen=True, slots=True)
class Sample:
    task_id: str
    model: str
    response: str
    think_chars: int = 0
    latency_ms: float = 0.0
    thinking: str = ""


@dataclass(frozen=True, slots=True)
class MetricScores:
    cvd: float
    hed: float
    enu: float
    ssf: float
    af: float
    cci: float
    pni: float
    vtr: float

    def as_dict(self) -> dict[str, float]:
        return {
            "cvd": self.cvd,
            "hed": self.hed,
            "enu": self.enu,
            "ssf": self.ssf,
            "af": self.af,
            "cci": self.cci,
            "pni": self.pni,
            "vtr": self.vtr,
        }

    @property
    def tshu_score(self) -> float:
        return round(sum(self.as_dict()[k] * w for k, w in WEIGHTS.items()), 2)


@dataclass(frozen=True, slots=True)
class ScoredSample:
    sample: Sample
    task: Task
    metrics: MetricScores
    raw: dict[str, float]


@dataclass(slots=True)
class ModelReport:
    model: str
    n_samples: int
    tshu_score: float
    metrics: dict[str, float]
    opening_entropy: float
    worst: list[ScoredSample] = field(default_factory=list)
