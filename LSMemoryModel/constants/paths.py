"""
Contains paths to various files and directories all given with respect
to the root directory of the project.
"""
import os
from pathlib import Path

# Base paths for the project.
ROOT_PATH = Path(__file__).parent.parent.parent
DATA_PATH = os.path.join(ROOT_PATH, "data")
ETC_PATH = os.path.join(ROOT_PATH, "etc")

# object dump path
OBJECT_DUMP_PATH = os.path.join(ETC_PATH, "object_dump")
