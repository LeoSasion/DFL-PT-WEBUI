"""Resolve pinned installed resources before the legacy research cache.

An existing installed group is authoritative: a damaged group must fail its
identity check rather than silently falling back to a different directory.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_ROOT = ROOT / '_internal/vision_models/production'
CACHE_ROOT = ROOT / 'workspace/.vision-models'


def resource_group(model_id, fallback, *, project_root=ROOT):
    if not isinstance(model_id, str) or not model_id or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-.' for c in model_id):
        raise ValueError('Invalid fixed resource group')
    installed = Path(project_root) / '_internal/vision_models/production' / model_id
    if installed.exists():
        if installed.is_symlink() or installed.is_junction() or not installed.is_dir():
            raise ValueError('Installed resource group must be a plain directory')
        return installed
    return Path(fallback)
