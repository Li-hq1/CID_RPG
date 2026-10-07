"""Locations shared by the visualization and analysis scripts."""
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
RESULTS_ROOT = PACKAGE_DIR / "vis_results"


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a path against the repository root when it is not absolute."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()
