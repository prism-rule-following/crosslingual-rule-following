from pydantic import BaseModel, ConfigDict
from datetime import datetime as dt
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
    date: str = now_date()
    results_folder: str = f"probing_results{date}"
    eval_folder: str = "eval_probes"
    vis_folder: str = "vis_folder"
    hf_repo_id: str = "crosslingual-rule-following/rule-following-eval"
    trained_probes_path: str = os.path.join(results_folder, trained_probes_folder)
    eval_path: str = os.path.join(results_folder, eval_folder)
    vis_path: str = os.path.join(results_folder, vis_folder)
