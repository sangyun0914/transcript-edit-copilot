"""Reconstruct an exact artifact in a new copy-on-write clone; never edit inputs."""

import ctypes
import hashlib
import os
from pathlib import Path
import re
import shutil


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def identity(path):
    stat = Path(path).stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def clone_file(source, destination):
    library = ctypes.CDLL(None, use_errno=True)
    function = library.clonefile
    function.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int)
    function.restype = ctypes.c_int
    if function(os.fsencode(source), os.fsencode(destination), 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


def rebuild_clone(
    base,
    target,
    destination,
    *,
    base_sha256,
    target_sha256,
    max_changed_bytes,
    minimum_free_bytes,
    chunk_bytes=1024 * 1024,
):
    """Create target's exact bytes using base's shared blocks and bounded writes.

    This operation makes no model transformation or source replacement. The
    caller must separately establish ownership before publishing a new artifact.
    """
    base, target, destination = map(Path, (base, target, destination))
    if any(path.is_symlink() for path in (base, target, destination)):
        raise ValueError("Symlink artifact paths are not supported.")
    base, target = base.resolve(strict=True), target.resolve(strict=True)
    destination = destination.parent.resolve(strict=True) / destination.name
    if destination.exists():
        raise FileExistsError(destination)
    if not base.is_file() or not target.is_file():
        raise ValueError("Both inputs must be regular files.")
    if any(
        not re.fullmatch(r"[0-9a-f]{64}", value)
        for value in (base_sha256, target_sha256)
    ):
        raise ValueError("Bind both complete artifact hashes.")
    if (
        type(chunk_bytes) is not int
        or not 4096 <= chunk_bytes <= 8 * 1024 * 1024
        or chunk_bytes % 4096
        or type(max_changed_bytes) is not int
        or max_changed_bytes < 0
        or type(minimum_free_bytes) is not int
        or minimum_free_bytes < 0
    ):
        raise ValueError("Use bounded, aligned chunks and explicit storage limits.")
    before = {"base": identity(base), "target": identity(target)}
    if before["base"][2] != before["target"][2]:
        raise ValueError("Clone reconstruction requires equal artifact sizes.")
    free_before = shutil.disk_usage(destination.parent).free
    if free_before < minimum_free_bytes + max_changed_bytes:
        raise ValueError("Insufficient free space for the bounded private writes.")
    created = False
    try:
        clone_file(base, destination)
        created = True
        if identity(destination)[:2] in {before["base"][:2], before["target"][:2]}:
            raise ValueError("A clone must have its own inode.")
        free_after_clone = shutil.disk_usage(destination.parent).free
        hashes = {"base": hashlib.sha256(), "target": hashlib.sha256()}
        changed_bytes, changed_chunks, offset = 0, 0, 0
        with (
            base.open("rb") as left,
            target.open("rb") as right,
            destination.open("r+b") as result,
        ):
            while True:
                original, wanted = left.read(chunk_bytes), right.read(chunk_bytes)
                if len(original) != len(wanted):
                    raise ValueError("Input size changed during reconstruction.")
                if not original:
                    break
                hashes["base"].update(original)
                hashes["target"].update(wanted)
                if original != wanted:
                    changed_bytes += len(wanted)
                    changed_chunks += 1
                    if changed_bytes > max_changed_bytes:
                        raise ValueError("Private-write budget exceeded.")
                    if shutil.disk_usage(
                        destination.parent
                    ).free < minimum_free_bytes + len(wanted):
                        raise ValueError("Free-space reserve would be crossed.")
                    result.seek(offset)
                    result.write(wanted)
                offset += len(wanted)
            result.flush()
            os.fsync(result.fileno())
        if {"base": identity(base), "target": identity(target)} != before:
            raise ValueError("Input identity changed during reconstruction.")
        if (
            hashes["base"].hexdigest() != base_sha256
            or hashes["target"].hexdigest() != target_sha256
        ):
            raise ValueError("Input artifact hash differs from its binding.")
        actual = digest(destination)
        if actual != target_sha256:
            raise ValueError("Cloned output differs from the complete target hash.")
        return {
            "version": "passive-export-clone-v1",
            "status": "exact_bytes_verified",
            "base": {"path": str(base), "sha256": base_sha256},
            "target": {"path": str(target), "sha256": target_sha256},
            "destination": {"path": str(destination), "sha256": actual},
            "bytes": offset,
            "chunk_bytes": chunk_bytes,
            "changed_chunks": changed_chunks,
            "changed_chunk_bytes": changed_bytes,
            "max_changed_bytes": max_changed_bytes,
            "minimum_free_bytes": minimum_free_bytes,
            "free_bytes_before": free_before,
            "free_bytes_after_clone": free_after_clone,
            "free_bytes_after_writes": shutil.disk_usage(destination.parent).free,
            "input_identities_unchanged": True,
            "inputs_replaced": False,
            "allocation_scope": "Changed chunk bytes bound explicit writes, not filesystem allocation. Free-space changes include concurrent host activity.",
        }
    except BaseException:
        if created:
            destination.unlink()
        raise


if __name__ == "__main__":
    raise SystemExit("Import rebuild_clone from a reviewed, artifact-bound caller.")
