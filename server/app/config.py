import os
from pathlib import Path

import platformdirs

DATA_DIR = Path(os.environ.get("AUDAPOLIS_DATA_DIR", platformdirs.user_data_dir("audapolis")))
DATA_DIR.mkdir(exist_ok=True, parents=True)

CACHE_DIR = Path(os.environ.get("AUDAPOLIS_CACHE_DIR", platformdirs.user_cache_dir("audapolis")))
CACHE_DIR.mkdir(exist_ok=True, parents=True)
