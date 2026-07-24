"""Enables `python -m tokenwatchdog`."""

import sys

from tokenwatchdog.cli import main

if __name__ == "__main__":
    sys.exit(main())
