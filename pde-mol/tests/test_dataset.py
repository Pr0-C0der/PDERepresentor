"""
Tests for dataset generation functionality.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from pde.dataset import ParameterRange, ParameterSampler, generate_dataset, _substitute_parameters


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestParameterRange:
    """Tests for ParameterRange class."""

    def test_parameter_range_creation(self):
        pr = ParameterRange("H", 1.0, 3.0)
        assert pr.name == "H"
        assert pr.low == 1.0
        assert pr.high == 3.0

    def test_parameter_range_validation(self):
        with pytest.raises(ValueError, match="low.*must be < high"):
            ParameterRange("H", 3.0, 1.0)

        with pytest.raises(ValueError, match="low.*must be < high"):
            ParameterRange("H", 2.0, 2.0)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestParameterSampler:
    """Tests for ParameterSampler class."""

    def test_sampler_creation(self):
        param_ranges = [
            ParameterRange("H", 1.0, 3.0),
            ParameterRange("A0", 0.0, 2.0),
        ]
        sampler = ParameterSampler(param_ranges, seed=42)
        assert len(sampler.param_ranges) == 2

    def test_sampler_sample(self):
        param_ranges = [
            ParameterRange("H", 1.0, 3.0),
            ParameterRange("A0", 0.0, 2.0),
        ]
        sampler = ParameterSampler(param_ranges, seed=42)
        params = sampler.sample()

        assert "H" in params
        assert "A0" in params
        assert 1.0 <= params["H"] <= 3.0
        assert 0.0 <= params["A0"] <= 2.0

    def test_sampler_deterministic(self):
        param_ranges = [ParameterRange("H", 1.0, 3.0)]
        sampler1 = ParameterSampler(param_ranges, seed=42)
        sampler2 = ParameterSampler(param_ranges, seed=42)

        params1 = sampler1.sample()
        params2 = sampler2.sample()

        assert params1["H"] == params2["H"]

    def test_sampler_sample_n(self):
        param_ranges = [ParameterRange("H", 1.0, 3.0)]
        sampler = ParameterSampler(param_ranges, seed=42)
        samples = sampler.sample_n(10)

        assert len(samples) == 10
        for sample in samples:
            assert "H" in sample
            assert 1.0 <= sample["H"] <= 3.0


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestParameterSubstitution:
    """Tests for parameter substitution function."""

    def test_substitute_simple_dict(self):
        obj = {"H": "H", "value": 5.0}
        params = {"H": 2.0}
        result = _substitute_parameters(obj, params)
        assert result == {"H": 2.0, "value": 5.0}

    def test_substitute_nested_dict(self):
        obj = {
            "operator": {"type": "diffusion", "nu": "H"},
            "ic": {"expr": "A0 * np.sin(x)"},
        }
        params = {"H": 2.0, "A0": 1.5}
        result = _substitute_parameters(obj, params)
        assert result["operator"]["nu"] == 2.0
        # Expression strings are not substituted (they're evaluated later)
        assert result["ic"]["expr"] == "A0 * np.sin(x)"

    def test_substitute_list(self):
        obj = ["H", "A0", 5.0]
        params = {"H": 2.0, "A0": 1.5}
        result = _substitute_parameters(obj, params)
        assert result == [2.0, 1.5, 5.0]

    def test_substitute_numeric_string(self):
        obj = "2.5"
        params = {}
        result = _substitute_parameters(obj, params)
        assert result == 2.5

    def test_substitute_expression_string(self):
        obj = "np.sin(x)"
        params = {"H": 2.0}
        result = _substitute_parameters(obj, params)
        # Expression strings are preserved
        assert result == "np.sin(x)"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestDatasetGeneration:
    """Tests for dataset generation function."""

    def test_single_sample_1d_periodic(self):
        """Test generating a single sample for 1D periodic problem."""
        template = {
            "domain": {
                "type": "1d",
                "x0": 0.0,
                "x1": 6.283185307179586,
                "nx": 51,
                "periodic": True,
            },
            "initial_condition": {
                "type": "expression",
                "expr": "np.sin(x)",
            },
            "operators": [
                {"type": "diffusion", "nu": "H"},
            ],
            "time": {
                "t0": 0.0,
                "t1": 0.1,
                "num_points": 10,
            },
        }

        param_ranges = [ParameterRange("H", 0.1, 1.0)]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            savepath = Path(tmpdir) / "test_dataset.pt"
            dataset = generate_dataset(
                json_template=template,
                param_ranges=param_ranges,
                num_samples=1,
                savepath=savepath,
                seed=42,
            )

            # Check dataset structure
            assert "params" in dataset
            assert "x" in dataset
            assert "t" in dataset
            assert "u" in dataset
            assert "param_names" in dataset

            # Check shapes
            assert dataset["params"].shape == (1, 1)  # (num_samples, num_params)
            assert dataset["x"].shape == (51,)  # (nx,)
            assert dataset["t"].shape == (10,)  # (nt,)
            assert dataset["u"].shape == (1, 10, 51)  # (num_samples, nt, nx)

            # Check parameter value is in range
            h_value = dataset["params"][0, 0].item()
            assert 0.1 <= h_value <= 1.0

            # Verify file was saved
            assert savepath.exists()
            loaded = torch.load(savepath)
            assert "params" in loaded
            assert loaded["params"].shape == (1, 1)

    def test_multi_sample_1d(self):
        """Test generating multiple samples."""
        template = {
            "domain": {
                "type": "1d",
                "x0": 0.0,
                "x1": 6.283185307179586,
                "nx": 51,
                "periodic": True,
            },
            "initial_condition": {
                "type": "expression",
                "expr": "A0 * np.sin(x)",
            },
            "operators": [
                {"type": "diffusion", "nu": "H"},
            ],
            "time": {
                "t0": 0.0,
                "t1": 0.1,
                "num_points": 10,
            },
        }

        param_ranges = [
            ParameterRange("H", 0.1, 1.0),
            ParameterRange("A0", 0.5, 2.0),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            savepath = Path(tmpdir) / "test_dataset.pt"
            dataset = generate_dataset(
                json_template=template,
                param_ranges=param_ranges,
                num_samples=5,
                savepath=savepath,
                seed=42,
            )

            # Check shapes
            assert dataset["params"].shape == (5, 2)  # (num_samples, num_params)
            assert dataset["u"].shape == (5, 10, 51)  # (num_samples, nt, nx)

            # Check all parameter values are in range
            for i in range(5):
                h_value = dataset["params"][i, 0].item()
                a0_value = dataset["params"][i, 1].item()
                assert 0.1 <= h_value <= 1.0
                assert 0.5 <= a0_value <= 2.0

            # Check parameter names
            assert dataset["param_names"] == ["H", "A0"]

    def test_1d_nonperiodic_with_bc(self):
        """Test 1D non-periodic problem with boundary conditions."""
        template = {
            "domain": {
                "type": "1d",
                "x0": 0.0,
                "x1": 1.0,
                "nx": 51,
                "periodic": False,
            },
            "initial_condition": {
                "type": "expression",
                "expr": "np.ones_like(x)",
            },
            "boundary_conditions": {
                "left": {"type": "dirichlet", "value": "u_left"},
                "right": {"type": "dirichlet", "value": "u_right"},
            },
            "operators": [
                {"type": "diffusion", "nu": "D"},
            ],
            "time": {
                "t0": 0.0,
                "t1": 0.1,
                "num_points": 10,
            },
        }

        param_ranges = [
            ParameterRange("D", 0.01, 0.1),
            ParameterRange("u_left", 0.0, 0.5),
            ParameterRange("u_right", 0.5, 1.0),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            savepath = Path(tmpdir) / "test_dataset.pt"
            dataset = generate_dataset(
                json_template=template,
                param_ranges=param_ranges,
                num_samples=3,
                savepath=savepath,
                seed=42,
            )

            # Check shapes
            assert dataset["params"].shape == (3, 3)
            assert dataset["u"].shape == (3, 10, 51)

            # Verify boundary conditions are satisfied (approximately)
            for i in range(3):
                u_solution = dataset["u"][i, -1, :].numpy()  # Final time step
                u_left = dataset["params"][i, 1].item()
                u_right = dataset["params"][i, 2].item()
                # Check boundaries (allowing for numerical error)
                assert abs(u_solution[0] - u_left) < 0.01
                assert abs(u_solution[-1] - u_right) < 0.01

    def test_save_final_only(self):
        """Test saving only final snapshot."""
        template = {
            "domain": {
                "type": "1d",
                "x0": 0.0,
                "x1": 6.283185307179586,
                "nx": 51,
                "periodic": True,
            },
            "initial_condition": {
                "type": "expression",
                "expr": "np.sin(x)",
            },
            "operators": [
                {"type": "diffusion", "nu": "H"},
            ],
            "time": {
                "t0": 0.0,
                "t1": 0.1,
                "num_points": 10,
            },
        }

        param_ranges = [ParameterRange("H", 0.1, 1.0)]

        with tempfile.TemporaryDirectory() as tmpdir:
            savepath = Path(tmpdir) / "test_dataset.pt"
            dataset = generate_dataset(
                json_template=template,
                param_ranges=param_ranges,
                num_samples=2,
                savepath=savepath,
                save_full_time=False,
                seed=42,
            )

            # Check shape: should have only 1 time point
            assert dataset["u"].shape == (2, 1, 51)  # (num_samples, 1, nx)

    def test_deterministic_sampling(self):
        """Test that same seed produces same parameter values."""
        template = {
            "domain": {
                "type": "1d",
                "x0": 0.0,
                "x1": 6.283185307179586,
                "nx": 51,
                "periodic": True,
            },
            "initial_condition": {
                "type": "expression",
                "expr": "np.sin(x)",
            },
            "operators": [
                {"type": "diffusion", "nu": "H"},
            ],
            "time": {
                "t0": 0.0,
                "t1": 0.1,
                "num_points": 5,
            },
        }

        param_ranges = [ParameterRange("H", 0.1, 1.0)]

        with tempfile.TemporaryDirectory() as tmpdir:
            savepath1 = Path(tmpdir) / "dataset1.pt"
            dataset1 = generate_dataset(
                json_template=template,
                param_ranges=param_ranges,
                num_samples=3,
                savepath=savepath1,
                seed=42,
            )

            savepath2 = Path(tmpdir) / "dataset2.pt"
            dataset2 = generate_dataset(
                json_template=template,
                param_ranges=param_ranges,
                num_samples=3,
                savepath=savepath2,
                seed=42,
            )

            # Parameters should be identical
            assert torch.allclose(dataset1["params"], dataset2["params"])

    def test_json_file_template(self):
        """Test using a JSON file as template."""
        template = {
            "domain": {
                "type": "1d",
                "x0": 0.0,
                "x1": 6.283185307179586,
                "nx": 51,
                "periodic": True,
            },
            "initial_condition": {
                "type": "expression",
                "expr": "np.sin(x)",
            },
            "operators": [
                {"type": "diffusion", "nu": "H"},
            ],
            "time": {
                "t0": 0.0,
                "t1": 0.1,
                "num_points": 5,
            },
        }

        param_ranges = [ParameterRange("H", 0.1, 1.0)]

        with tempfile.TemporaryDirectory() as tmpdir:
            # Write template to file
            template_path = Path(tmpdir) / "template.json"
            with open(template_path, "w") as f:
                json.dump(template, f)

            savepath = Path(tmpdir) / "dataset.pt"
            dataset = generate_dataset(
                json_template=template_path,
                param_ranges=param_ranges,
                num_samples=2,
                savepath=savepath,
                seed=42,
            )

            assert dataset["params"].shape == (2, 1)
            assert dataset["u"].shape == (2, 5, 51)

