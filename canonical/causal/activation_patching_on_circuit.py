"""Main script to run activation patching on a circuit."""

def sufficiency_verification(circuit: dict) -> dict:
    """
    Sufficiency: keep circuit, corrupt rest.
    The function returns a dict with responses and adherence evaluation.
    """
    pass


def necessity_verification(circuit: dict) -> dict:
    """
    Necessity: keep rest, corrupt circuit.
    The function returns a dict with responses and adherence evaluation.
    """
    pass


def completeness_verification(circuit: dict) -> dict:
    """
    Completeness: co-ablate circuit and its backups.
    The function returns a dict with responses and adherence evaluation.
    """
    pass


def minimality_verification(circuit: dict) -> dict:
    """
    Minimality: ablate every node, evaluate adherence.
    The function returns a dict with responses and adherence evaluation.
    """
    pass


def run_statistical_comparison(act_patching_results) -> dict:
    """Function to answer the following (and many more) questions:
    
    1. How similar were the experiments on original vs held-out data?
    2. How similar were the experiments across different languages?"""
    pass


def run_activation_patching(circuit: dict):
    """Function to co-ordinate all of the above."""
    # Run sufficiency verification
    # Run on held-out
    # Run on different languages
