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


@dataclass
class Domain2D:
    """
    Simple 2D rectangular domain with a structured tensor-product grid.

    Parameters
    ----------
    x0, x1, nx:
        Bounds and number of points in the x-direction.
    y0, y1, ny:
        Bounds and number of points in the y-direction.
    periodic_x, periodic_y:
        Flags indicating periodicity along x and y respectively.
    """

    x0: float
    x1: float
    nx: int
    y0: float
    y1: float
    ny: int
    periodic_x: bool = False
    periodic_y: bool = False

    def __post_init__(self) -> None:
        if self.nx < 2 or self.ny < 2:
            raise ValueError("nx and ny must each be at least 2.")
        if self.x1 <= self.x0:
            raise ValueError("x1 must be greater than x0.")
        if self.y1 <= self.y0:
            raise ValueError("y1 must be greater than y0.")

        self._x = np.linspace(self.x0, self.x1, self.nx)
        self._y = np.linspace(self.y0, self.y1, self.ny)
        self._dx = float(self._x[1] - self._x[0])
        self._dy = float(self._y[1] - self._y[0])
        self._X, self._Y = np.meshgrid(self._x, self._y, indexing="xy")

    # ------------------------------------------------------------------
    # Basic properties
    # ------------------------------------------------------------------
    @property
    def x(self) -> np.ndarray:
        """1D grid coordinates along x."""
        return self._x

    @property
    def y(self) -> np.ndarray:
        """1D grid coordinates along y."""
        return self._y

    @property
    def X(self) -> np.ndarray:
        """2D meshgrid of x-coordinates (shape (ny, nx))."""
        return self._X

    @property
    def Y(self) -> np.ndarray:
        """2D meshgrid of y-coordinates (shape (ny, nx))."""
        return self._Y

    @property
    def dx(self) -> float:
        return self._dx

    @property
    def dy(self) -> float:
        return self._dy

    @property
    def shape(self) -> Tuple[int, int]:
        """Grid shape as (ny, nx)."""
        return (self.ny, self.nx)

    @property
    def size(self) -> int:
        """Total number of grid points (ny * nx)."""
        return self.nx * self.ny

    @property
    def periodic(self) -> bool:
        """
        Compatibility flag with 1D domains.

        A 2D domain is considered periodic only if it is periodic along both
        x and y directions.
        """
        return self.periodic_x and self.periodic_y

    # ------------------------------------------------------------------
    # Flatten / unflatten helpers
    # ------------------------------------------------------------------
    def flatten(self, u2d: np.ndarray) -> np.ndarray:
        """
        Flatten a 2D field with shape (ny, nx) into a 1D vector.
        """
        arr = np.asarray(u2d, dtype=float)
        if arr.shape != self.shape:
            raise ValueError(f"Expected array of shape {self.shape}, got {arr.shape}.")
        return arr.ravel()

    def unflatten(self, u_flat: np.ndarray) -> np.ndarray:
        """
        Reshape a 1D vector into a 2D field with shape (ny, nx).
        """
        arr = np.asarray(u_flat, dtype=float)
        if arr.size != self.size:
            raise ValueError(f"Expected flat array of size {self.size}, got {arr.size}.")
        return arr.reshape(self.shape)

    # ------------------------------------------------------------------
    # (De-)serialisation helpers
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict:
        """Convert this domain into a JSON-serialisable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "Domain2D":
        """
        Construct a Domain2D from a dictionary.

        Expected keys:
        - x0, x1, nx
        - y0, y1, ny
        - periodic_x, periodic_y (optional, default False)
        """
        return cls(
            x0=float(data["x0"]),
            x1=float(data["x1"]),
            nx=int(data["nx"]),
            y0=float(data["y0"]),
            y1=float(data["y1"]),
            ny=int(data["ny"]),
            periodic_x=bool(data.get("periodic_x", False)),
            periodic_y=bool(data.get("periodic_y", False)),
        )

