from canonical.model.dataset import HFDataHelper
from dotenv import load_dotenv

load_dotenv()


def main():
    local_file = "canonical/data/v2/obligation_en_test.json"
    hf_repo_id = "nunaa/canonical_obligation_dataset"

    helper = HFDataHelper(hf_repo_id)
    helper.upload_file(local_file, "data/en_test.json")


if __name__ == "__main__":
    main()
