"""To get version and git hash of built package."""
from importlib import metadata

try:
    __version__ = metadata.version("pyscipopt_ml")
except metadata.PackageNotFoundError:
    __version__ = "dev"

GIT_HASH = "$Format:%H$"


def get_versions():
    """Get package version."""
    # Downloaded package with inserted git hash.
    if "Format" not in GIT_HASH:
        git_hash = f"-{GIT_HASH}"
    # No inserted git hash, the repo is probably cloned.
    else:
        git_hash = ""

    return {"short": __version__, "long": f"{__version__}{git_hash}"}
