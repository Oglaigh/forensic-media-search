"""Legacy smoke-test entrypoint, now routed through the ensemble pipeline.

Prefer invoking ``forensic_search.py`` directly. This file remains so existing
development workflows do not execute the former softmax/probability example.
"""

from forensic_media_search.cli import main


if __name__ == "__main__":
    raise SystemExit(
        main(
            [
                "--directory", "/evidence",
                "--query", "a house with a red roof",
                "--query", "a house with a black door",
                "--top-k", "1",
                "--batch-size", "1",
                "--max-images", "1",
                "--output", "/output/test_clip_report.csv",
            ]
        )
    )
