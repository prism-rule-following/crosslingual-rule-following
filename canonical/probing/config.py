from pydantic import BaseModel, ConfigDict, model_validator
from datetime import datetime as dt
from typing import Optional
import os


def now_date() -> str:
    """Util to get now time in a certain format."""
    return dt.now().strftime("%Y-%m-%d_%H-%M-%S")


class RunConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    language: str
    n_layers: int
    dataset_name: str
    trained_probes_folder: str = "trained_probes"
    eval_folder: str = "eval_probes"
    vis_folder: str = "vis_folder"
    hf_repo_id: str = "crosslingual-rule-following/rule-following-eval"
    date: Optional[str] = None
    results_folder: Optional[str] = None
    trained_probes_path: Optional[str] = None
    eval_path: Optional[str] = None
    vis_path: Optional[str] = None

    @model_validator(mode="after")
    def _fill_computed_paths(self) -> "RunConfig":
        date = self.date or now_date()
        results_folder = self.results_folder or f"probing_results_{self.language}_{date}"
        object.__setattr__(self, "date", date)
        object.__setattr__(self, "results_folder", results_folder)
        object.__setattr__(
            self,
            "trained_probes_path",
            self.trained_probes_path or os.path.join(results_folder, self.trained_probes_folder),
        )
        object.__setattr__(
            self, "eval_path", self.eval_path or os.path.join(results_folder, self.eval_folder)
        )
        object.__setattr__(
            self, "vis_path", self.vis_path or os.path.join(results_folder, self.vis_folder)
        )
        return self
