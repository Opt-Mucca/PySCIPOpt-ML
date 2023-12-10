"""Utility function to find function that add a predictor in dictionary."""


def get_convertor(predictor, convertors):
    """Return the convertor for a given predictor."""
    convertor = None
    try:
        convertor = convertors[type(predictor)]
    except KeyError:
        pass
    if convertor is None:
        for parent in type(predictor).mro():
            try:
                convertor = convertors[parent]
                break
            except KeyError:
                pass
    if convertor is None:
        name = type(predictor).__name__
        try:
            convertor = convertors[name]
        except KeyError:
            pass
    return convertor
