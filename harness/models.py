"""数据结构定义。"""

from dataclasses import dataclass, field

WEIGHTS = {
    "adr": 0.15,
    "eid": 0.15,
    "ogr": 0.15,
    "ssf": 0.15,
    "cci": 0.10,
    "pni": 0.10,
    "osc": 0.10,
    "vtr": 0.10,
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
    d_ref: float | None = None

    @property
    def answer_anchors(self) -> list[str]:
        if self.trap_type == "flawed_premise" and self.refutation_anchors:
            return self.refutation_anchors
        return self.verdict_anchors

    @property
    def expected_density(self) -> float:
        """专家在字数预算内用掉约六成有效实体时的密度，可按题覆盖。"""
        if self.d_ref is not None:
            return self.d_ref
        return (len(self.actionable_entities) * 0.6 + 2.0) * 100.0 / max(1, self.max_chars)


@dataclass(frozen=True, slots=True)
class Sample:
    task_id: str
    model: str
    response: str
    think_chars: int = 0
    latency_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class MetricScores:
    adr: float
    eid: float
    ogr: float
    ssf: float
    cci: float
    pni: float
    osc: float
    vtr: float

    def as_dict(self) -> dict[str, float]:
        return {
            "adr": self.adr,
            "eid": self.eid,
            "ogr": self.ogr,
            "ssf": self.ssf,
            "cci": self.cci,
            "pni": self.pni,
            "osc": self.osc,
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
