from . import iteris, ssr_merge, sum, ties


methods = {
    "iteris": iteris,
    "ssr_merge": ssr_merge,
    "sum": sum,
    "ties": ties,
}


def requires_pipeline(method_name):
    method = methods.get(method_name)
    return bool(method and getattr(method, "requires_pipeline", False))


def compose(transformer, cfg_composition, **kwargs):
    method_name = cfg_composition.get("method")
    try:
        method = methods[method_name]
    except KeyError as error:
        available = ", ".join(sorted(methods))
        raise ValueError(
            f"Unknown composition method {method_name!r}; "
            f"available methods: {available}"
        ) from error
    return method.compose(transformer, cfg_composition, **kwargs)
