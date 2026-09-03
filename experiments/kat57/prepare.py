# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "datasets>=4.0.0",
#     "huggingface-hub>=0.30",
#     "pillow",
#     "tqdm",
# ]
# ///
"""Convert Lund University Library's Kat57 PAGE XML release to a Hub dataset.

The source is a 36 GB ZIP, so this script is designed for a cheap Hugging Face
CPU Job with 50 GB of ephemeral storage. It keeps the source archive plus only
one Parquet shard on disk at a time and uploads completed shards immediately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import urllib.request
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

from datasets import Dataset, Features, Image, Sequence, Value
from huggingface_hub import HfApi, get_token, hf_hub_download
from tqdm import tqdm

ZENODO_RECORD = "https://zenodo.org/records/14679534"
ARCHIVE_URL = "https://zenodo.org/api/records/14679534/files/kat57-gt-dataset.zip/content"
ARCHIVE_FILENAME = "kat57-gt-dataset.zip"
ARCHIVE_SIZE = 36_301_215_824
ARCHIVE_MD5 = "0895d5785cd0ddf9f9c6488484ac447e"
LICENSE = "cc-by-4.0"
MANIFEST_PATH = "conversion-manifest.json"

FEATURES = Features(
    {
        "id": Value("string"),
        "drawer_id": Value("string"),
        "card_id": Value("string"),
        "image": Image(),
        "reference": Value("string"),
        "lines": Sequence(Value("string")),
        "line_count": Value("int32"),
        "page_xml": Value("string"),
        "source_image": Value("string"),
    }
)


@dataclass(frozen=True)
class CardPair:
    """Matched image and PAGE XML members for one catalogue card."""

    card_id: str
    image_member: str
    xml_member: str


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_page_xml(xml_bytes: bytes) -> tuple[list[str], str]:
    """Extract line transcriptions in PAGE document order.

    eScriptorium's Kat57 files store one ``TextEquiv/Unicode`` transcription
    under each ``TextLine``. XML order reflects the human transcription order;
    joining lines with newlines preserves it while CER/WER normalized mode can
    intentionally ignore line wrapping.
    """
    root = ElementTree.fromstring(xml_bytes)
    lines: list[str] = []
    for element in root.iter():
        if _local_name(element.tag) != "TextLine":
            continue
        transcription = next(
            (child.text or "" for child in element.iter() if _local_name(child.tag) == "Unicode"),
            "",
        )
        if transcription:
            lines.append(transcription)
    return lines, "\n".join(lines)


def discover_pairs(archive: zipfile.ZipFile) -> list[CardPair]:
    """Match PNG and PAGE XML members by stem and fail on incomplete data."""
    images: dict[str, str] = {}
    xml_files: dict[str, str] = {}
    for member in archive.namelist():
        if member.endswith("/") or member.startswith("__MACOSX/"):
            continue
        suffix = Path(member).suffix.lower()
        if suffix not in {".png", ".xml"}:
            continue
        stem = Path(member).stem
        target = images if suffix == ".png" else xml_files
        if stem in target:
            raise ValueError(f"Duplicate {suffix} member for '{stem}'")
        target[stem] = member

    missing_images = sorted(set(xml_files) - set(images))
    missing_xml = sorted(set(images) - set(xml_files))
    if missing_images or missing_xml:
        raise ValueError(
            "Kat57 archive contains unmatched files: "
            f"{len(missing_images)} missing PNG, {len(missing_xml)} missing XML"
        )
    if not images:
        raise ValueError("Kat57 archive contains no PNG/XML card pairs")

    return [CardPair(stem, images[stem], xml_files[stem]) for stem in sorted(images)]


def card_row(archive: zipfile.ZipFile, pair: CardPair) -> dict[str, object]:
    """Read one matched pair into the stable Hugging Face row schema."""
    xml_bytes = archive.read(pair.xml_member)
    lines, reference = parse_page_xml(xml_bytes)
    image_bytes = archive.read(pair.image_member)
    parts = pair.card_id.split("-")
    drawer_id = parts[-2] if len(parts) >= 3 else ""
    card_id = parts[-1] if len(parts) >= 2 else pair.card_id
    return {
        "id": pair.card_id,
        "drawer_id": drawer_id,
        "card_id": card_id,
        "image": {"bytes": image_bytes, "path": Path(pair.image_member).name},
        "reference": reference,
        "lines": lines,
        "line_count": len(lines),
        "page_xml": xml_bytes.decode("utf-8"),
        "source_image": Path(pair.image_member).name,
    }


def iter_shards(
    archive: zipfile.ZipFile,
    pairs: list[CardPair],
    shard_size: int,
) -> Iterator[tuple[int, Dataset]]:
    """Yield bounded-memory Dataset shards."""
    for start in range(0, len(pairs), shard_size):
        rows = [card_row(archive, pair) for pair in pairs[start : start + shard_size]]
        yield start // shard_size, Dataset.from_list(rows, features=FEATURES)


def deterministic_sample(pairs: list[CardPair], sample_size: int, seed: int) -> list[CardPair]:
    """Select a reproducible pseudo-random subset, returned in source order.

    Ranking stable card identifiers by a seeded SHA-256 digest avoids depending
    on Python's random/shuffle implementation. Sorting the selected cards back
    into identifier order keeps the published dataset easy to inspect while the
    membership remains an unbiased deterministic sample.
    """
    if sample_size < 1:
        raise ValueError("Sample size must be at least 1")
    if sample_size > len(pairs):
        raise ValueError(f"Sample size {sample_size} exceeds the {len(pairs)} available cards")

    def sample_key(pair: CardPair) -> bytes:
        return hashlib.sha256(f"{seed}\0{pair.card_id}".encode()).digest()

    selected = sorted(pairs, key=sample_key)[:sample_size]
    return sorted(selected, key=lambda pair: pair.card_id)


def conversion_manifest(
    pairs: list[CardPair],
    *,
    source_count: int,
    shard_size: int,
    max_samples: int | None,
    sample_seed: int | None,
) -> dict[str, object]:
    """Describe the exact conversion layout so resume cannot mix runs."""
    membership = hashlib.sha256(
        "\n".join(pair.card_id for pair in pairs).encode()
    ).hexdigest()
    return {
        "version": 1,
        "source_archive_md5": ARCHIVE_MD5,
        "source_count": source_count,
        "example_count": len(pairs),
        "shard_size": shard_size,
        "shard_count": math.ceil(len(pairs) / shard_size),
        "max_samples": max_samples,
        "sample_seed": sample_seed,
        "selection": "seeded-sha256" if sample_seed is not None else "prefix",
        "selected_ids_sha256": membership,
    }


def md5sum(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - verifying Zenodo's published checksum
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_archive(destination: Path) -> Path:
    """Download the fixed Zenodo archive with safe in-job resume support."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing = destination.stat().st_size if destination.exists() else 0
    if existing > ARCHIVE_SIZE:
        raise ValueError(f"Partial archive is larger than expected ({existing} > {ARCHIVE_SIZE})")
    if existing == ARCHIVE_SIZE:
        print(f"Using complete archive at {destination}")
    else:
        request = urllib.request.Request(ARCHIVE_URL)
        if existing:
            request.add_header("Range", f"bytes={existing}-")
        with urllib.request.urlopen(request) as response:
            resumed = existing > 0 and response.status == 206
            if existing and not resumed:
                existing = 0
            mode = "ab" if resumed else "wb"
            with (
                destination.open(mode) as output,
                tqdm(
                    total=ARCHIVE_SIZE,
                    initial=existing,
                    unit="B",
                    unit_scale=True,
                    desc="Downloading Kat57",
                ) as progress,
            ):
                while chunk := response.read(8 * 1024 * 1024):
                    output.write(chunk)
                    progress.update(len(chunk))

    actual_size = destination.stat().st_size
    if actual_size != ARCHIVE_SIZE:
        raise ValueError(
            f"Archive size mismatch: expected {ARCHIVE_SIZE}, downloaded {actual_size}"
        )
    actual_md5 = md5sum(destination)
    if actual_md5 != ARCHIVE_MD5:
        raise ValueError(f"Archive checksum mismatch: expected {ARCHIVE_MD5}, got {actual_md5}")
    return destination


def dataset_card(
    example_count: int,
    shard_count: int,
    *,
    source_count: int,
    sample_seed: int | None,
) -> str:
    """Build the attributed data-only Hub card."""
    if sample_seed is None:
        scope = f"{example_count:,} scanned catalogue"
        conversion = f"The source archive contains {source_count:,} matched PNG/XML pairs."
    else:
        scope = f"a deterministic {example_count:,}-card sample of scanned catalogue"
        conversion = (
            f"The source archive contains {source_count:,} matched PNG/XML pairs. "
            f"This copy selects {example_count:,} cards by ranking stable card identifiers "
            f"with a seeded SHA-256 digest (seed `{sample_seed}`)."
        )
    return f"""---
license: {LICENSE}
language:
- sv
- de
- en
- fr
- da
- 'no'
tags:
- ocr
- htr
- page-xml
- libraries
- cultural-heritage
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*.parquet
---

# Kat57 ground truth

Hugging Face conversion of Lund University Library's
[Kat57 ground-truth release]({ZENODO_RECORD}): {scope}
cards with manually corrected PAGE XML transcriptions.

The cards come from Catalogue -1957, Lund University Library's alphabetical
catalogue of holdings published through 1957. They contain a mixture of
typewritten and handwritten text in several languages.

## Fields

- `image`: original PNG card scan
- `reference`: line transcriptions joined in PAGE XML document order
- `lines`: individual PAGE XML line transcriptions
- `page_xml`: original PAGE XML, including line geometry
- `id`, `drawer_id`, `card_id`, `source_image`: source identifiers

## Conversion

{conversion}
This copy is stored in {shard_count} Parquet shards and was produced by the reproducible
Kat57 converter in [`tadad/ocr-bench`](https://github.com/tadad/ocr-bench/tree/feat/cer-wer-kat57/experiments/kat57).
No OCR or synthetic labels were introduced: `reference`, `lines`, and
`page_xml` are derived directly from Lund's human-transcribed PAGE XML.

Source ZIP MD5: `{ARCHIVE_MD5}`.

## License and attribution

The source dataset is CC BY 4.0. Credit Maria Hedberg and Lund University
Library, and cite the [Zenodo record]({ZENODO_RECORD}) when using this
conversion. This repository is a convenience conversion and is not an
official Lund University Library publication.
"""


def publish(
    archive_path: Path,
    repo_id: str,
    *,
    shard_size: int,
    max_samples: int | None,
    private: bool,
    resume: bool,
    sample_seed: int | None,
    work_dir: Path,
) -> None:
    token = get_token() or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("No Hugging Face token found; use `hf auth login` or HF_TOKEN")
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)

    existing_files = set(api.list_repo_files(repo_id, repo_type="dataset"))
    existing_shards = {path for path in existing_files if path.startswith("data/train-")}
    if existing_shards and not resume:
        raise RuntimeError(
            f"{repo_id} already has {len(existing_shards)} data shards; "
            "pass --resume or choose an empty repository"
        )

    work_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        pairs = discover_pairs(archive)
        source_count = len(pairs)
        if max_samples is not None:
            pairs = (
                deterministic_sample(pairs, max_samples, sample_seed)
                if sample_seed is not None
                else pairs[:max_samples]
            )
        shard_count = math.ceil(len(pairs) / shard_size)
        manifest = conversion_manifest(
            pairs,
            source_count=source_count,
            shard_size=shard_size,
            max_samples=max_samples,
            sample_seed=sample_seed,
        )
        expected_shards = {
            f"data/train-{index:05d}-of-{shard_count:05d}.parquet" for index in range(shard_count)
        }
        stale = existing_shards - expected_shards
        if stale:
            raise RuntimeError(
                "Existing shard layout does not match this run; use a new empty repo. "
                f"Unexpected examples: {sorted(stale)[:3]}"
            )
        if MANIFEST_PATH in existing_files:
            manifest_file = hf_hub_download(
                repo_id,
                MANIFEST_PATH,
                repo_type="dataset",
                token=token,
            )
            with Path(manifest_file).open() as file:
                existing_manifest = json.load(file)
            if existing_manifest != manifest:
                raise RuntimeError(
                    "Existing conversion manifest does not match this run; "
                    "use the original arguments or a new empty repo"
                )
        elif existing_shards:
            raise RuntimeError(
                "Existing shards have no conversion manifest, so their parameters "
                "and membership cannot be verified; use a new empty repo"
            )
        else:
            api.upload_file(
                path_or_fileobj=(json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
                path_in_repo=MANIFEST_PATH,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message="Record Kat57 conversion manifest",
            )

        print(f"Converting {len(pairs):,} cards into {shard_count} shards")
        for shard_index, dataset in iter_shards(archive, pairs, shard_size):
            path_in_repo = f"data/train-{shard_index:05d}-of-{shard_count:05d}.parquet"
            if path_in_repo in existing_shards:
                print(f"Skipping uploaded shard {shard_index + 1}/{shard_count}")
                continue
            local_path = work_dir / Path(path_in_repo).name
            dataset.to_parquet(local_path)
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"Add Kat57 shard {shard_index + 1}/{shard_count}",
            )
            local_path.unlink()
            print(f"Uploaded shard {shard_index + 1}/{shard_count}")

    api.upload_file(
        path_or_fileobj=dataset_card(
            len(pairs),
            shard_count,
            source_count=source_count,
            sample_seed=sample_seed,
        ).encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Document Kat57 conversion",
    )
    print(f"Published https://huggingface.co/datasets/{repo_id}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_id", help="Destination Hugging Face dataset repo")
    parser.add_argument(
        "--archive",
        type=Path,
        help="Existing source ZIP; otherwise download the verified Zenodo archive",
    )
    parser.add_argument("--work-dir", type=Path, default=Path("kat57-work"))
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--sample-seed",
        type=int,
        help=(
            "Select --max-samples cards reproducibly across the full archive "
            "instead of taking the first cards"
        ),
    )
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.shard_size < 1:
        raise SystemExit("--shard-size must be at least 1")
    if args.max_samples is not None and args.max_samples < 1:
        raise SystemExit("--max-samples must be at least 1")
    if args.sample_seed is not None and args.max_samples is None:
        raise SystemExit("--sample-seed requires --max-samples")

    work_dir = args.work_dir.resolve()
    archive = args.archive.resolve() if args.archive else work_dir / ARCHIVE_FILENAME
    if args.archive:
        if not archive.is_file():
            raise SystemExit(f"Archive not found: {archive}")
    else:
        archive = download_archive(archive)
    try:
        publish(
            archive,
            args.repo_id,
            shard_size=args.shard_size,
            max_samples=args.max_samples,
            private=args.private,
            resume=args.resume,
            sample_seed=args.sample_seed,
            work_dir=work_dir,
        )
    finally:
        # Locally supplied archives belong to the caller. A downloaded copy is
        # job scratch data and can be removed after a successful/failed publish.
        if not args.archive and archive.exists():
            archive.unlink()
        if work_dir.exists() and not any(work_dir.iterdir()):
            work_dir.rmdir()


if __name__ == "__main__":
    main()
