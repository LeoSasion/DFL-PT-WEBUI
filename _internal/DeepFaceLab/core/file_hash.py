"""Bounded-memory file digests; callers retain their own path safety policy."""
import hashlib


def file_sha256(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()
