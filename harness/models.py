"""数据结构与 Pydantic v2 模型定义。"""

import warnings

from pydantic import BaseModel, ConfigDict, Field

WEIGHTS = {
    "cvd": 0.12,
    "hed": 0.14,
    "enu": 0.22,
    "cci": 0.22,
    "pni": 0.14,
    "vtr": 0.16,
}


def classify_tier(score: float) -> str:
    """按综合天书指数划分三档梯队。"""
    if score < 11.0:
        return "T1 · 人话级"
    if score < 15.0:
        return "T2 · 黑话级"
    return "T3 · 天书级"


with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message='Field name "register".*')

    class Task(BaseModel):
        """单条测试任务定义。"""

        model_config = ConfigDict(frozen=True)

        id: str
        scenario: str
        register: str
        trap_type: str
        max_chars: int
        prompt: str
        verdict_anchors: list[str]
        actionable_entities: list[str]
        fence_sitting_markers: list[str] = Field(default_factory=list)
        refutation_anchors: list[str] = Field(default_factory=list)

        @property
        def answer_anchors(self) -> list[str]:
            if self.trap_type == "flawed_premise" and self.refutation_anchors:
                return self.refutation_anchors
            return self.verdict_anchors


class Sample(BaseModel):
    """单条模型输出样本。"""

    model_config = ConfigDict(frozen=True)

    task_id: str
    model: str
    response: str
    think_chars: int = 0
    latency_ms: float = 0.0
    thinking: str = ""
    reasoning_effort: str = ""
    temperature: float = 0.3
    max_tokens: int = 8192
    suite_hash: str = ""
    sampled_at: str = ""
    response_model: str = ""


class MetricScores(BaseModel):
    """六维指标得分。"""

    model_config = ConfigDict(frozen=True)

    cvd: float
    hed: float
    enu: float
    cci: float
    pni: float
    vtr: float

    def as_dict(self) -> dict[str, float]:
        return {
            "cvd": self.cvd,
            "hed": self.hed,
            "enu": self.enu,
            "cci": self.cci,
            "pni": self.pni,
            "vtr": self.vtr,
        }

    @property
    def open_score(self) -> float:
        return round(self.cvd * WEIGHTS["cvd"] + self.hed * WEIGHTS["hed"], 2)

    @property
    def body_score(self) -> float:
        return round(
            self.enu * WEIGHTS["enu"]
            + self.cci * WEIGHTS["cci"]
            + self.pni * WEIGHTS["pni"]
            + self.vtr * WEIGHTS["vtr"],
            2,
        )

    @property
    def tshu_score(self) -> float:
        return round(sum(self.as_dict()[k] * w for k, w in WEIGHTS.items()), 2)


class ScoredSample(BaseModel):
    """单条样本及其评测结果。"""

    model_config = ConfigDict(frozen=True)

    sample: Sample
    task: Task
    metrics: MetricScores
    raw: dict[str, float]


class ModelReport(BaseModel):
    """单模型聚合评测报告。"""

    model: str
    n_samples: int
    tshu_score: float
    metrics: dict[str, float]
    opening_entropy: float
    worst: list[ScoredSample] = Field(default_factory=list)
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    open_score: float = 0.0
    body_score: float = 0.0
    tier: str = ""
    reasoning_effort: str = ""
