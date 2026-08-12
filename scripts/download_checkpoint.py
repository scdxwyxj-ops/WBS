#!/usr/bin/env python3
"""Download and verify the exact SAM2.1 Hiera-L checkpoint used by this release."""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path

DEFAULT_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
EXPECTED_SHA256 = "2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("checkpoints/sam2.1_hiera_large.pt"))
    parser.add_argument("--url", default=DEFAULT_URL)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file() and sha256(output) == EXPECTED_SHA256:
        print(f"Already verified: {output}")
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        print(f"Downloading {args.url} -> {output}")
        urllib.request.urlretrieve(args.url, temporary)
        actual = sha256(temporary)
        if actual != EXPECTED_SHA256:
            raise RuntimeError(f"Checkpoint SHA256 mismatch: expected {EXPECTED_SHA256}, got {actual}")
        temporary.replace(output)
        print(f"Verified SHA256: {actual}")
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
