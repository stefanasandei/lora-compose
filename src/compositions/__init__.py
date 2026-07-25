from . import sum


methods = {
    "sum": sum,
}


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
