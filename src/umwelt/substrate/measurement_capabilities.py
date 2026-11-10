"""Read-only operation declarations; these are not observations or confidence."""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class MeasurementCapabilities:
    """Whether positive-strength operations are implemented, subject to valid input.

    None means undeclared, not unsupported. Support does not imply exact global
    back-action, physical truth or calibrated probabilities; read the description.
    """
    axis: bool | None = None
    pair: bool | None = None
    description: str = "Measurement capabilities are not declared."
    contract: str = field(default="measurement-capabilities/1", init=False)

    def __post_init__(self):
        if any(value is not None and type(value) is not bool for value in (self.axis, self.pair)):
            raise TypeError("measurement capabilities must be bool or None")
        if not isinstance(self.description, str) or not self.description:
            raise TypeError("measurement capability description must be a nonempty string")


_UNKNOWN = MeasurementCapabilities()


def measurement_capabilities(cluster) -> MeasurementCapabilities:
    """Return a backend's declaration without attempting a measurement.

    Unknown implementations are never guessed dense because they have no flags,
    or called to discover support. Older clients may retain exception fallback.
    """
    declared = getattr(cluster, "measurement_capabilities", _UNKNOWN)
    if not isinstance(declared, MeasurementCapabilities):
        raise TypeError("backend has an invalid measurement capability declaration")
    return declared
