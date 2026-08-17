"""
check_credits.py — print remaining OpenRouter credits via the OpenRouter SDK.

Env:
  OPENROUTER_API_KEY   required. Note: OpenRouter's /credits endpoint needs a
                        Management API key — a plain inference key may get a
                        403 here even though it works fine for chat calls.
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

load_dotenv()

from openrouter import OpenRouter, errors


def main() -> None:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        print("OPENROUTER_API_KEY not set", file=sys.stderr)
        raise SystemExit(1)

    client = OpenRouter(api_key=key)
    try:
        resp = client.credits.get_credits()
    except errors.OpenRouterError as e:
        print(
            f"OpenRouter API error: HTTP {e.status_code}: {e.body[:300]}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    total = resp.data.total_credits
    used = resp.data.total_usage
    remaining = total - used

    print(f"Total credits purchased: {total:,.4f}")
    print(f"Total credits used:      {used:,.4f}")
    print(f"Remaining credits:       {remaining:,.4f}")


if __name__ == "__main__":
    main()
