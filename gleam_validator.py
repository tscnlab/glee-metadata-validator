"""Backward-compatible import and CLI alias for the GLC validator."""

from glc_validator import *  # noqa: F401,F403
from glc_validator import main


if __name__ == "__main__":
    main()
