from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Tuple

import numpy as np


@dataclass
class Domain1D:
    """
    Simple 1D domain with a structured grid.

    Parameters
    ----------
    x0, x1:
        Left and right boundaries of the spatial domain.
    nx:
        Number of grid points (including boundaries).
    periodic:
        If True, the domain is considered periodic in x.
    """

    x0: float
    x1: float
    nx: int
    periodic: bool = False

    def __post_init__(self) -> None:
        if self.nx < 2:
            raise ValueError("nx must be at least 2.")
        if self.x1 <= self.x0:
            raise ValueError("x1 must be greater than x0.")

        # Precompute grid and spacing
        self._x = np.linspace(self.x0, self.x1, self.nx)
        self._dx = float(self._x[1] - self._x[0])

    # ------------------------------------------------------------------
    # Basic properties
    # ------------------------------------------------------------------
    @property
    def x(self) -> np.ndarray:
        """Return the 1D grid coordinates."""
        return self._x

    @property
    def dx(self) -> float:
        """Return the uniform grid spacing."""
        return self._dx

    # ------------------------------------------------------------------
    # (De-)serialisation helpers
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict:
        """Convert this domain into a JSON-serialisable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "Domain1D":
        """
        Construct a Domain1D from a dictionary.

        The dictionary is expected to contain the keys:
        - x0, x1, nx
        - periodic (optional, defaults to False)
        """
        return cls(
            x0=float(data["x0"]),
            x1=float(data["x1"]),
            nx=int(data["nx"]),
            periodic=bool(data.get("periodic", False)),
        )


