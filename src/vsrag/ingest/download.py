import shutil
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

COMPLAINTS_URL = "https://static.nhtsa.gov/odi/ffdd/cmpl/COMPLAINTS_RECEIVED_2020-2024.zip"
RECALLS_URL = "https://static.nhtsa.gov/odi/ffdd/rcl/FLAT_RCL_POST_2010.zip"


def download(url: str, dest_dir: Path) -> Path:
    """Save `url` into `dest_dir`, unless it's already there. Returns the file's path.

    Downloads to a .part file first, so an interrupted download never looks finished.
    """
    dest = dest_dir / Path(urlparse(url).path).name
    if dest.exists():
        return dest

    dest_dir.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "vehicle-safety-rag"})
    with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as out:
        shutil.copyfileobj(response, out)
    partial.replace(dest)
    return dest
