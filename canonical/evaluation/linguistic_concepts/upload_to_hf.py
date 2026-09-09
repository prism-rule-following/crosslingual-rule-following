from canonical.model.dataset import HFDataHelper
from dotenv import load_dotenv

load_dotenv()

LANGUAGES = ["en", "de", "hi", "ig", "it", "ko", "ru", "tr", "ur", "yo"]
SPLITS = ["train", "test", "train_scenario", "test_scenario"]


def main():
    hf_repo_id = "nunaa/canonical_obligation_dataset"
    helper = HFDataHelper(hf_repo_id)

    for lang in LANGUAGES:
        for split in SPLITS:
            local_file = f"canonical/data/v2/obligation_{lang}_{split}.json"
            path_in_repo = f"data/{lang}_{split}.json"
            print(f"Uploading {local_file} -> {path_in_repo}")
            helper.upload_file(local_file, path_in_repo)


if __name__ == "__main__":
    main()
