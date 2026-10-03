"""DFL version-1 PAK and ZIP facesets, shared by the CLI and data tools."""

import hashlib
import os
import pickle
import struct
import zipfile
from pathlib import Path, PurePosixPath

from core import pathex
from core.interact import interact as io
from samplelib import Sample

packed_faceset_filename = "faceset.pak"
packed_faceset_filename_zip = "faceset.zip"
packed_faceset_filename_config = "config.pak"
PACK_EXTENSION = "pak"
MAX_CONFIG_BYTES = 128 * 1024 * 1024


def _member(sample):
    parts = ([str(sample.person_name)] if sample.person_name is not None else []) + [str(sample.filename)]
    if any(not item or item in (".", "..") or "/" in item or "\\" in item or ":" in item for item in parts):
        raise ValueError("Unsafe packed faceset member name")
    return "/".join(parts)


def _target(directory, sample):
    candidate = directory.joinpath(*PurePosixPath(_member(sample)).parts)
    if directory.resolve() not in candidate.resolve().parents:
        raise ValueError("Packed faceset member escaped the output directory")
    return candidate


class PackedFaceset:
    VERSION = 1

    @staticmethod
    def pack(samples_path, ext="pak", pak_name=None, delete_original=None):
        from samplelib import SampleLoader
        directory = Path(samples_path)
        if ext is None:
            ext = io.input_str("Faceset archive type", "pak", ["pak", "zip"])
        ext = ext.lstrip(".").lower()
        if ext not in ("pak", "zip"):
            raise ValueError("Faceset archive type must be pak or zip")
        base = pak_name or "faceset"
        if not base or Path(base).name != base or any(char in base for char in '<>:"/\\|?*'):
            raise ValueError("Invalid faceset archive name")
        destination = directory / f"{base}.{ext}"
        if destination.exists():
            io.input_str(f"{destination} already exists. Press Enter to overwrite.")
        subdirectories = pathex.get_all_dir_names(directory)
        as_person = bool(subdirectories) and io.input_bool(
            f"Found {len(subdirectories)} subdirectories. Pack as a person faceset?", True)
        image_paths = []
        if as_person:
            for name in subdirectories:
                image_paths.extend(pathex.get_image_paths(directory / name))
        else:
            image_paths = pathex.get_image_paths(directory)
        samples = SampleLoader.load_face_samples(image_paths)
        if not samples or len(samples) != len(image_paths):
            raise ValueError("Faceset packing requires readable DFL images with metadata")
        configs, members = [], set()
        for sample in samples:
            source = Path(sample.filename)
            sample.filename = source.name
            if as_person:
                sample.person_name = source.parent.name
            member = _member(sample)
            if member in members:
                raise ValueError("Duplicate faceset archive member")
            members.add(member)
            configs.append(sample.get_config())
        metadata = pickle.dumps(configs, protocol=4)
        if len(metadata) > MAX_CONFIG_BYTES:
            raise ValueError("Faceset metadata exceeds the supported limit")
        pending = destination.with_suffix(destination.suffix + ".tmp")
        try:
            if ext == "pak":
                with pending.open("wb") as stream:
                    stream.write(struct.pack("<QQ", PackedFaceset.VERSION, len(metadata)))
                    stream.write(metadata)
                    table_start = stream.tell()
                    stream.write(bytes(8 * (len(samples) + 1)))
                    data_start = stream.tell()
                    offsets = []
                    for sample in io.progress_bar_generator(samples, "Packing"):
                        offsets.append(stream.tell() - data_start)
                        stream.write(_target(directory, sample).read_bytes())
                    offsets.append(stream.tell() - data_start)
                    stream.seek(table_start)
                    for offset in offsets:
                        stream.write(struct.pack("<Q", offset))
            else:
                with zipfile.ZipFile(pending, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                    archive.writestr(packed_faceset_filename_config, metadata)
                    for sample in io.progress_bar_generator(samples, "Packing"):
                        archive.write(_target(directory, sample), _member(sample))
                    archive.comment = hashlib.md5(str(archive.namelist()).encode()).digest()
            os.replace(pending, destination)
        finally:
            pending.unlink(missing_ok=True)
        if delete_original is None:
            delete_original = io.input_bool("Delete original images?", True)
        if delete_original:
            for filename in image_paths:
                Path(filename).unlink()
            if as_person:
                for name in subdirectories:
                    try:
                        (directory / name).rmdir()
                    except OSError:
                        pass
        return destination

    @staticmethod
    def unpack(samples_path, pak_name=None):
        directory = Path(samples_path)
        samples = PackedFaceset.load(directory, pak_name=pak_name)
        if samples is None:
            raise FileNotFoundError(f"No packed faceset found in {directory}")
        archive_path = Path(samples[0]._filename_offset_size[0]) if samples else (
            directory / f"{pak_name or 'faceset'}.pak")
        restored = []
        try:
            for sample in io.progress_bar_generator(samples, "Unpacking"):
                target = _target(directory, sample)
                payload = sample.read_raw_file()
                if target.exists():
                    if target.read_bytes() != payload:
                        raise FileExistsError(f"Refusing to overwrite a different image: {target}")
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                pending = target.with_name(target.name + ".unpack.tmp")
                pending.write_bytes(payload)
                restored.append((pending, target))
            for pending, target in restored:
                os.replace(pending, target)
            archive_path.unlink()
        finally:
            for pending, _ in restored:
                pending.unlink(missing_ok=True)

    @staticmethod
    def path_contains(samples_path):
        directory = Path(samples_path)
        return any((directory / name).is_file() for name in (packed_faceset_filename, packed_faceset_filename_zip))

    @staticmethod
    def load(samples_path, pak_name=None, archive_path=None):
        directory = Path(samples_path)
        base = pak_name or "faceset"
        if Path(base).name != base or any(char in base for char in '<>:"/\\|?*'):
            raise ValueError("Invalid faceset archive name")
        if archive_path is not None:
            path = Path(archive_path).resolve()
            if path.parent != directory.resolve() or path.suffix.lower() not in (".pak", ".zip"):
                raise ValueError("Explicit faceset archive must be a PAK/ZIP file inside the faceset directory")
            if not path.is_file():
                raise FileNotFoundError(f"Packed faceset archive not found: {path}")
        else:
            path = directory / f"{base}.pak"
            if not path.exists():
                path = directory / f"{base}.zip"
                if not path.exists():
                    return None
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as archive:
                info = archive.getinfo(packed_faceset_filename_config)
                if info.file_size > MAX_CONFIG_BYTES:
                    raise ValueError("Faceset metadata is too large")
                configs = pickle.loads(archive.read(packed_faceset_filename_config))
                if not isinstance(configs, list):
                    raise ValueError("Faceset metadata must contain a sample list")
                samples = [Sample(**config) for config in configs]
                members = set()
                for sample in samples:
                    member = _member(sample)
                    if member in members or member not in archive.namelist():
                        raise ValueError("Duplicate or missing ZIP faceset image")
                    members.add(member)
                    sample.set_filename_offset_size(str(path), -1, -1)
                return samples
        with path.open("rb") as stream:
            header = stream.read(16)
            if len(header) != 16:
                raise ValueError("Truncated faceset header")
            version, metadata_size = struct.unpack("<QQ", header)
            if version != PackedFaceset.VERSION or metadata_size > MAX_CONFIG_BYTES:
                raise ValueError("Unsupported faceset version or metadata size")
            metadata = stream.read(metadata_size)
            if len(metadata) != metadata_size:
                raise ValueError("Truncated faceset metadata")
            configs = pickle.loads(metadata)
            if not isinstance(configs, list):
                raise ValueError("Faceset metadata must contain a sample list")
            samples = [Sample(**config) for config in configs]
            table = stream.read(8 * (len(samples) + 1))
            if len(table) != 8 * (len(samples) + 1):
                raise ValueError("Truncated faceset offset table")
            offsets = list(struct.unpack(f"<{len(samples) + 1}Q", table))
            data_start = stream.tell()
        data_size = path.stat().st_size - data_start
        if offsets[-1] == path.stat().st_size:
            # Original DFL writes an absolute EOF for the final offset.
            offsets[-1] = data_size
        if offsets[0] != 0 or offsets[-1] != data_size or any(a > b for a, b in zip(offsets, offsets[1:])):
            raise ValueError("Invalid faceset image offsets")
        members = set()
        for index, sample in enumerate(samples):
            member = _member(sample)
            if member in members:
                raise ValueError("Duplicate faceset image")
            members.add(member)
            sample.set_filename_offset_size(str(path), data_start + offsets[index], offsets[index + 1] - offsets[index])
        return samples
