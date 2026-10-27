"""Sampling-stability and judge-confidence analysis for the camera-ready revision.

data.py    -- loads the 5-sample responses and the three logprob judge exports
              into one wide table (ack_invert excluded: not judged in this run).
metrics.py -- the pre-registered checks and analyses (one function per item).
run.py     -- CLI that writes every table, REPORT.md and summary.json, and can
              upload the output folder to Hugging Face.
"""
