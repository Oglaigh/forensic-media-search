"""Backward-compatible script entrypoint for the modular application."""

from forensic_media_search.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
