"""VLA Tasks Extension for Isaac Lab."""

import os
import toml

# Get extension path
EXT_VLA_TASKS_EXT_DIR = os.path.dirname(os.path.realpath(__file__))

# Register Gym environments
from .config.franka import *
