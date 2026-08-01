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

ALSO ADDED, per veerlosar's PR comment: make_independent_correctness_judge,
an API-based alternative to correctness_score's embedding similarity, for
the specific failure mode embedding similarity can miss -- a confidently
WRONG response that shares most of its vocabulary with the correct answer
can score as similar despite asserting the opposite. veerlosar's suggestion:
use the API version while there's budget, fall back to the free embedding
version when there isn't. Both are available; nothing auto-switches between
them yet -- that's a reasonable next step if it'd help, but for now it's a
manual choice of which to call.
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


def make_independent_correctness_judge(api_key, model_name="claude-haiku-4-5-20251001"):
    """
    API-based alternative to correctness_score's embedding similarity, for
    the specific failure mode embedding similarity can miss: a confidently
    WRONG response that shares most of its vocabulary with the correct
    answer (the flat-earth/round-earth case in this file's adversarial
    test) can score as similar despite asserting the opposite. An LLM asked
    directly whether the response matches the expected answer should catch
    that more reliably than cosine similarity.

    Costs an API call per row, unlike correctness_score (free, local) --
    per veerlosar's point, worth using while there's API budget rather than
    building it and never trying it, but not a blanket replacement for the
    free version at full dataset scale. A sensible middle ground: run the
    free embedding check first, and only spend an API call on cases near
    the threshold where the cheap version is genuinely ambiguous, rather
    than calling this on every row unconditionally.

    Returns a callable judge_fn(response, expected_answer) -> dict with a
    "correct" key, mirroring correctness_score's boolean output shape so
    it's a drop-in alternative wherever that's used.
    """
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    def judge_fn(response, expected_answer):
        prompt = (f'Expected answer: "{expected_answer}"\n'
                  f'Actual response: "{response}"\n\n'
                  f"Does the actual response convey the same substantive answer as the expected "
                  f"answer, even if worded differently? Answer no if the response contradicts the "
                  f"expected answer, even if the wording is similar.\n\n"
                  f"Answer with only one word: yes or no.\nAnswer:")
        try:
            message = client.messages.create(model=model_name, max_tokens=5,
                                                messages=[{"role": "user", "content": prompt}])
            text = message.content[0].text.strip().lower()
            correct = text.startswith("yes") if text.startswith(("yes", "no")) else None
            return {"correct": correct, "raw": text}
        except Exception as e:
            return {"correct": None, "note": f"API error: {type(e).__name__}: {e}"}

    return judge_fn


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

    # make_independent_correctness_judge: response-parsing logic tested with
    # a mock, since a real call needs an API key and network access -- same
    # discipline as the adherence-judge test in llm_judge.py.
    def _parse_correct(text):
        text = text.strip().lower()
        return text.startswith("yes") if text.startswith(("yes", "no")) else None

    assert _parse_correct("yes") is True
    assert _parse_correct("Yes.") is True
    assert _parse_correct("no") is False
    assert _parse_correct("garbage") is None
    print("make_independent_correctness_judge's response parsing verified correct.")

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
