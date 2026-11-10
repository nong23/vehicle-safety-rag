from pathlib import Path

from vsrag.ingest.download import download


def test_download_saves_the_file(tmp_path: Path) -> None:
    source = tmp_path / "source.zip"
    source.write_bytes(b"data")

    saved = download(source.as_uri(), tmp_path / "raw")

    assert saved == tmp_path / "raw" / "source.zip"
    assert saved.read_bytes() == b"data"


def test_download_skips_a_file_already_there(tmp_path: Path) -> None:
    existing = tmp_path / "raw" / "source.zip"
    existing.parent.mkdir()
    existing.write_bytes(b"old")

    saved = download((tmp_path / "source.zip").as_uri(), tmp_path / "raw")

    assert saved.read_bytes() == b"old"
