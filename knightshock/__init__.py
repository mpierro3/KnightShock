__version__ = "0.0.1"


from . import absorption
from . import figures
from . import kinetics


def format_mixture(mixture: str | dict) -> dict[str, float]:
    if isinstance(mixture, dict):
        return dict((x.strip().upper(), float(y)) for x, y in mixture.items())
    elif isinstance(mixture, str):
        mixture = mixture.replace("{", "").replace("}", "").upper()
        if ":" not in mixture:
            return {mixture.strip(): 1.0}
        else:
            return dict(
                (x.strip(), float(y))
                for x, y in (element.split(":") for element in (mixture.split(",")))
            )
    else:
        raise TypeError("Mixture argument must be str or dict.")
