from enum import StrEnum
from typing import List, Optional

from pydantic import BaseModel, Field

from config.dataset_config import DataCategories, DatasetConfig, DatasetLanguage


class EAPContinuationMode(StrEnum):
    normal = "normal"  # start_with, ack_invert, bold_html, language, single_word
    teacher_forced = "teacher_forced"  # banned_word, include_word, word_count


class EAPMetrics(StrEnum):
    logit_diff = "logit_diff"


class EAPMethods(StrEnum):
    EAP = "EAP"
    EAP_IG_inputs = "EAP-IG-inputs"
    EAP_IG_activations = "EAP-IG-activations"
    clean_corrupted = "clean-corrupted"


class EAPDatasetConfig(DatasetConfig):
    category: List[DataCategories] = Field(
        ...,
        min_length=1,
        max_length=1,
        description="exactly one category to attribute over",
    )

    language: List[DatasetLanguage] = Field(
        ...,
        min_length=1,
        max_length=1,
        description="language of the rows this config attributes over",
    )


class EAPConfig(BaseModel):
    mode: EAPContinuationMode = Field(
        default=EAPContinuationMode.normal,
        description=(
            "how the metric reads model output: 'normal' scores the first "
            "response token (start_with, ack_invert, bold_html, language, "
            "single_word); 'teacher_forced' scores a forced multi-token "
            "response (banned_word, include_word, word_count)"
        ),
    )
    model_id: str = Field(
        ..., description="HuggingFace model id of the model to attribute"
    )
    dataset_config: EAPDatasetConfig = Field(
        ..., description="dataset config, restricted to a single category per run"
    )
    batch_size: int = Field(
        default=10, description="number of examples per dataloader batch"
    )
    metrics: EAPMetrics = Field(
        default=EAPMetrics.logit_diff,
        description="which metric to attribute/evaluate with respect to",
    )
    method: EAPMethods = Field(
        default=EAPMethods.EAP_IG_activations,
        description="attribution method to run: EAP, EAP-IG-inputs, EAP-IG-activations, or clean-corrupted",
    )
    ig_steps: int = Field(
        default=10, description="number of integrated-gradients interpolation steps"
    )
    n_edge_steps: int = Field(
        default=20, description="number of points to sample in the n_edges sweep"
    )
    n_edge_start: int = Field(
        default=50, description="smallest n_edges tested in the sweep"
    )
    n_edge_end_proportion: float = Field(
        default=0.05,
        description=(
            "largest n_edges tested in the sweep, expressed as a proportion "
            "of the graph's total real edges"
        ),
    )

    def image_file_name(self, operation: str) -> str:
        """Build a filename for a run artifact, e.g. image_file_name('circuit')."""
        model_id = self.model_id.split("/")[1]
        return (
            f"{model_id}_{self.dataset_config.language[0].value}_"
            f"{self.method.value}_{self.dataset_config.category[0].value}_"
            f"{operation}.png"
        )


class EAPNodes(BaseModel):
    name: str = Field(..., description="node name, e.g. 'input', 'm11', 'a0.h11'")
    score: Optional[float] = Field(
        default=None,
        description="node's attribution score, if node-level scoring was used",
    )


class EAPEdges(BaseModel):
    name: str = Field(..., description="edge name, formatted as '[parent]->[child]'")
    score: Optional[float] = Field(
        default=None, description="edge's attribution score from the attribution method"
    )


class EAPResult(BaseModel):
    n_nodes: int = Field(..., description="number of nodes included in this circuit")
    n_edges: int = Field(
        ...,
        description="number of edges included in this circuit (this sweep point's n_edges budget)",
    )
    nodes: List[EAPNodes] = Field(
        default_factory=list, description="nodes included in the pruned circuit"
    )
    edges: List[EAPEdges] = Field(
        default_factory=list, description="edges included in the pruned circuit"
    )
    circuit_performance: float = Field(
        default=0,
        description="raw metric value with only this circuit kept clean, everything else corrupted",
    )
    circuit_faithfulness: float = Field(
        default=0,
        description=(
            "normalized faithfulness: "
            "(circuit_performance - corrupted_baseline) / (baseline - corrupted_baseline)"
        ),
    )


class EAPResults(BaseModel):
    results: List[EAPResult] = Field(
        description="one entry per n_edges value tested in the sweep"
    )
    metadata: Optional[dict] = Field(
        default_factory=dict,
        description="run metadata: the EAPConfig used for this run, serialized as JSON",
    )
    baseline: float = Field(
        default=0, description="metric on the full, unablated model"
    )
    corrupted_baseline: float = Field(
        default=0,
        description="metric on the model with everything corrupted (the floor reference)",
    )
    best: EAPResult = Field(
        description="the sweep point selected as the knee/elbow of the n_edges-vs-faithfulness curve"
    )
    circuit_image_path: Optional[str] = Field(
        default=None,
        description="local path to the best circuit's rendered image, if pygraphviz was available",
    )
    knee_image_path: Optional[str] = Field(
        default=None,
        description="local path to the n_edges-vs-faithfulness knee plot, if it saved successfully",
    )
