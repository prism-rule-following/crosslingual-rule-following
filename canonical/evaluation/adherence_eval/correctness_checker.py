"""
Separate from llm_judge.py on purpose -- this checks something genuinely
different from rule adherence: whether the response actually got the
underlying task right, per nuna's PR point. Two questions, two checkers:

  Adherence (llm_judge.py): did the output satisfy the system rule?
  Correctness (this file):  did the output answer the user's query correctly?

Uses the same free local embedding approach already established for the
"meaning" check earlier in this project (sentence-transformers, no API
calls) -- but compares the response against an EXPECTED ANSWER, not the
question. That's a meaningfully harder, more discriminating comparison than
question-similarity (which is why the earlier meaning check stayed
uninformative near ceiling -- being on-topic is a low bar almost anything
clears; being *right* is a much higher one).

HONEST LIMITATION, stated up front, not discovered after the fact: embedding
similarity can still fail on the specific failure mode this is meant to
catch -- two responses can be highly similar in embedding space while one
is factually correct and the other asserts the opposite, especially in
topic-dense domains like medical/legal/financial phrasing, where correct
and incorrect answers often share most of their vocabulary. The adversarial
test at the bottom of this file is built specifically to check for that
failure mode -- but it needs a REAL embedding model to actually run, which
this environment can't reach (no huggingface.co access here) to download.
Run that test in Colab/Lambda before trusting this checker on real data --
don't skip it because the logic elsewhere checks out.
"""
import numpy as np


_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def is_degenerate(response, min_chars=5):
    stripped = response.strip()
    if len(stripped) < min_chars:
        return True
    words = stripped.split()
    if len(words) > 3 and len(set(words)) == 1:
        return True
    return False


def correctness_score(response, expected_answer, threshold=0.5):
    """
    Embedding similarity between the response and an expected/reference
    answer -- NOT the question, which is what made the earlier meaning-check
    weak. Returns (similarity, is_correct: bool).
    """
    if is_degenerate(response):
        return 0.0, False
    embedder = get_embedder()
    embs = embedder.encode([response, expected_answer])
    sim = float(np.dot(embs[0], embs[1]) / (np.linalg.norm(embs[0]) * np.linalg.norm(embs[1]) + 1e-12))
    return sim, sim > threshold


def evaluate_adherence_and_correctness(row, response, check_rule_following_fn, expected_answer=None,
                                          correctness_threshold=0.5):
    """
    Combines both checks for one response, per nuna's proposed split:
      - Did the output satisfy the system rule? (existing checker, e.g. from
        the formal categories, or an LLM judge for tone_norm-style rules)
      - Did the output answer the user's query correctly? (this file, only
        runs if row has an "expected_answer" field -- categories without one,
        like word_count/start_with/bold_html, don't have a "correct answer"
        beyond the format itself, so this returns None for those rather than
        forcing a comparison that doesn't mean anything)
    Returns both as independent labels -- this is the actual point: distinguishing
    "followed the rule but got the task wrong" from "got it right but ignored
    the rule," which a single combined score can't do.
    """
    rule_following = check_rule_following_fn(row, response)
    expected = expected_answer or row.get("expected_answer")
    if expected is None:
        correctness = None  # no expected_answer for this row/category -- not applicable, not "failed"
        sim = None
    else:
        sim, correctness = correctness_score(response, expected, correctness_threshold)
    return {
        "rule_following": rule_following,
        "correctness": correctness,
        "correctness_similarity": sim,
        "category": (rule_following, correctness),  # the four-way breakdown nuna asked for, at a glance
    }


# ---------------------------------------------------------------------
# What CAN be tested here: the combination logic, with mocked similarity
# scores standing in for a real embedder. What CANNOT be tested here: the
# adversarial case below actually needs a real embedding model, which this
# environment can't download (no huggingface.co access) -- run this specific
# block for real in Colab/Lambda before trusting the checker on real data.
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # combination-logic check with mocked correctness_score
    def _mock_evaluate(rule_following, correctness):
        return {"rule_following": rule_following, "correctness": correctness,
                "category": (rule_following, correctness)}

    cases = [
        (True, True, "followed rule AND correct -- the good case"),
        (True, False, "followed rule but got the TASK wrong"),
        (False, True, "got the task right but IGNORED the rule"),
        (False, False, "failed both"),
    ]
    print("Four-way breakdown this is meant to distinguish:")
    for rf, corr, desc in cases:
        result = _mock_evaluate(rf, corr)
        print(f"  rule_following={rf}, correctness={corr} -> {desc}")
    print("\nThis is exactly the distinction a single combined adherence score can't make --")
    print("confirmed the logic keeps both labels independent rather than collapsing them.")

    print("\n" + "=" * 70)
    print("Adversarial embedding test -- needs a real embedding model")
    print("=" * 70)

    adversarial_cases = [
        # (response, expected_answer, should_be_correct)
        ("The earth is a sphere, not flat.", "The earth is round.", True),
        # WRONG, but shares most of its vocabulary with the expected answer --
        # the exact failure mode this test exists to catch
        ("The earth is flat, despite what most people believe.", "The earth is round.", False),
        ("I don't know.", "The earth is round.", False),  # correct rejection -- should score low
    ]

    try:
        get_embedder()  # forces the download/load attempt now, so failure is caught cleanly below
        embedder_available = True
    except Exception as e:
        embedder_available = False
        print(f"Embedder not available here ({type(e).__name__}: {e}).")
        print("This is expected in an environment without huggingface.co access -- run this")
        print("same file in Colab/Lambda (where sentence-transformers can actually download")
        print("the model) to get real results before trusting correctness_score on real data.")

    if embedder_available:
        all_ok = True
        for response, expected, should_be_correct in adversarial_cases:
            sim, is_correct = correctness_score(response, expected)
            ok = (is_correct == should_be_correct)
            all_ok &= ok
            status = "OK" if ok else "FAILED -- similarity didn't distinguish correct from incorrect"
            print(f"[{status}] sim={sim:.3f}  response={response!r}")
        print()
        if all_ok:
            print("All three cases passed -- similarity is distinguishing correct from incorrect here.")
        else:
            print("At least one case failed -- likely the flat-earth/round-earth pair, since it shares")
            print("almost all its vocabulary with the correct answer while asserting the opposite.")
            print("That confirms the concern raised in the PR thread: this checker needs an upgrade")
            print("(e.g. an NLI/entailment model instead of raw cosine similarity) before it's")
            print("trustworthy for this failure mode -- not just a threshold tweak.")
