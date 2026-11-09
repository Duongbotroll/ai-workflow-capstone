#!/usr/bin/env python
"""
run every unit test (API, model, logger, monitoring) with a single command:

    python run-tests.py
"""

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

if __name__ == "__main__":
    suite = unittest.TestLoader().discover(os.path.join(HERE, "unittests"),
                                           pattern="*Tests.py", top_level_dir=HERE)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
