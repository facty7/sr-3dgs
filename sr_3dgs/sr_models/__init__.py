"""Super-resolution model registry and factory."""

from .base import BaseSRModel

_MODEL_REGISTRY = {}


def register_model(name: str):
    """Decorator to register an SR model class."""
    def wrapper(cls):
        _MODEL_REGISTRY[name] = cls
        return cls
    return wrapper


def get_sr_model(name: str, **kwargs) -> BaseSRModel:
    """Factory: instantiate an SR model by name."""
    if name not in _MODEL_REGISTRY:
        from .real_esrgan import RealESRGANModel
        from .dat import DATModel
        from .supir import SUPIRModel
        from .basicvsr_plus import BasicVSRPlusModel

    if name not in _MODEL_REGISTRY:
        available = list(_MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown SR model '{name}'. Available: {available}"
        )
    return _MODEL_REGISTRY[name](**kwargs)


def list_models():
    """List all registered SR model names."""
    # Ensure models are imported
    if not _MODEL_REGISTRY:
        from .real_esrgan import RealESRGANModel
        from .dat import DATModel
        from .supir import SUPIRModel
        from .basicvsr_plus import BasicVSRPlusModel
    return list(_MODEL_REGISTRY.keys())
