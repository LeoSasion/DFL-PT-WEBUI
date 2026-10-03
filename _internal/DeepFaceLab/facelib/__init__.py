from .FaceType import FaceType


def __getattr__(name):
    """Metadata/image tools do not need to import the Torch network modules."""
    if name in ("S3FDExtractor", "FANExtractor", "FaceEnhancer", "XSegNet"):
        from importlib import import_module
        value = getattr(import_module(f".{name}", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(name)
