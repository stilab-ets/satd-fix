import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
FIXING_SATD_ROOT = Path(os.getenv("SATD_REPLICATION_ROOT", PACKAGE_ROOT))
ENV_PATH = FIXING_SATD_ROOT / ".env"
LEGACY_ENV_PATH = FIXING_SATD_ROOT / "conf" / ".env"


def load_project_env():
    if load_dotenv is not None:
        load_dotenv(ENV_PATH, override=False)
        load_dotenv(LEGACY_ENV_PATH, override=False)
    else:
        for path in (ENV_PATH, LEGACY_ENV_PATH):
            if not path.exists():
                continue
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    return ENV_PATH if ENV_PATH.exists() else LEGACY_ENV_PATH
