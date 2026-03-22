"""
Active-round training wrapper.

Runs the same optimization as train_v6, but under a different script name so
the supervisor can decide independently whether to allow it during active rounds.
"""

import runpy

runpy.run_path("/workspace/train_v6.py", run_name="__main__")
