from model import meta_arch


def build_model(cfg, pool='max'):
    model_factory = getattr(meta_arch, cfg.MODEL.META_ARCHITECTURE)
    model = model_factory(cfg, pool)
    return model
