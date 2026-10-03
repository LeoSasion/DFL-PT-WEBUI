"""Content-addressed aligned-image inventory shared by probe and evaluation.

The inventory follows ME's dataset selection rules: an explicit PAK/ZIP, an
implicit faceset.pak/zip in a directory, or recursively discovered loose JPGs.
Its logical member names are stable across scans and do not depend on process
working directories or archive offsets.
"""

from dataclasses import dataclass
import hashlib
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg"}


@dataclass(frozen=True)
class ProbeImage:
    member: str
    name: str
    sha256: str
    path: Path | None = None
    packed_sample: object = None


def sha256_file(target):
    digest = hashlib.sha256()
    with Path(target).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _loose_images(directory):
    return sorted(
        (target for target in directory.rglob("*")
         if target.is_file() and not target.is_symlink()
         and target.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda target: (target.relative_to(directory).as_posix().casefold(),
                            target.relative_to(directory).as_posix()),
    )


def iter_probe_images(directory):
    """Retain the loose-image iterator for callers that need plain paths."""
    return _loose_images(Path(directory))


def probe_dataset_inventory(dataset):
    """Return (kind, fingerprint, logical images) for any ME faceset input."""
    from samplelib.PackedFaceset import PackedFaceset, _member

    selected = Path(dataset).resolve()
    explicit = selected if selected.is_file() and selected.suffix.lower() in (".pak", ".zip") else None
    directory = selected.parent if explicit else selected
    if not directory.is_dir():
        raise ValueError(f"Aligned dataset does not exist: {dataset}")
    packed = PackedFaceset.load(directory, archive_path=explicit)
    if packed is not None:
        if not packed:
            raise ValueError(f"Packed aligned faceset is empty: {dataset}")
        archive = Path(packed[0]._filename_offset_size[0]).resolve()
        kind = archive.suffix.lower().lstrip(".")
        members = sorted(((_member(sample), sample) for sample in packed),
                         key=lambda item: (item[0].casefold(), item[0]))
        images = [ProbeImage(member, Path(member).name,
                             hashlib.sha256(sample.read_raw_file()).hexdigest(),
                             packed_sample=sample)
                  for member, sample in members]
        # Packed metadata affects landmarks and masks even when JPEG bytes do not.
        fingerprint = hashlib.sha256(
            f"{kind}\0{sha256_file(archive)}".encode("ascii")
        ).hexdigest()
        return kind, fingerprint, images

    if explicit is not None:
        raise ValueError(f"Packed aligned faceset was not found: {selected}")
    digest = hashlib.sha256()
    images = []
    for target in _loose_images(directory):
        member = target.relative_to(directory).as_posix()
        file_digest = sha256_file(target)
        images.append(ProbeImage(member, target.name, file_digest, path=target))
        digest.update(member.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_digest.encode("ascii"))
        digest.update(b"\n")
    return "directory", digest.hexdigest(), images


def fingerprint_probe_directory(directory):
    """Compatibility API returning digests keyed by logical member name."""
    _, fingerprint, images = probe_dataset_inventory(directory)
    return fingerprint, {image.member: image.sha256 for image in images}
