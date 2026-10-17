"""Pure checks for discrete query-rotation retrieval."""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from continuloc.evaluation.query_rotation_search import (
    absolute_heading_error_rad,
    attach_heading_errors,
    circular_heading_from_responses,
    require_scene_mean_scale,
    resolve_retrieval_mode,
    rotation_search_ranking,
    rotation_steps_rad,
    summarize_heading_errors,
)


class RotationGridTest(unittest.TestCase):
    def test_ten_degree_grid_covers_the_circle_once(self):
        angles = rotation_steps_rad(10)
        self.assertEqual(angles.numel(), 36)
        degrees = torch.rad2deg(angles.double())
        self.assertTrue(torch.allclose(degrees, torch.arange(36, dtype=torch.float64) * 10))

    def test_step_must_divide_the_circle(self):
        with self.assertRaises(ValueError):
            rotation_steps_rad(7)
        with self.assertRaises(ValueError):
            rotation_steps_rad(0)
        with self.assertRaises(ValueError):
            rotation_steps_rad(float("nan"))


class RetrievalModeTest(unittest.TestCase):
    def test_default_and_reserved_modes(self):
        self.assertEqual(resolve_retrieval_mode(None), "north_aligned")
        self.assertEqual(resolve_retrieval_mode("Query_Rotation_Search"), "query_rotation_search")
        with self.assertRaises(NotImplementedError):
            resolve_retrieval_mode("reference_rotation_search")
        with self.assertRaises(ValueError):
            resolve_retrieval_mode("scale_sweep")

    def test_rotation_search_rejects_multiscale(self):
        self.assertEqual(require_scene_mean_scale("scene_mean"), "scene_mean")
        with self.assertRaises(ValueError):
            require_scene_mean_scale("multiscale")


class RotationSearchMathTest(unittest.TestCase):
    def test_summed_response_selects_the_location_and_its_angle(self):
        # Location 0 has the strongest single angle but the smaller sum.
        # Location 1 wins on the sum, and its mass is centered at a warp of 90 degrees.
        angles = rotation_steps_rad(10)
        responses = torch.full((1, 36, 2), -0.2)
        responses[0, :, 0] = -0.2
        responses[0, 0, 0] = 0.99
        center = 9  # 90 degrees
        responses[0, center - 1, 1] = 0.4
        responses[0, center, 1] = 0.8
        responses[0, center + 1, 1] = 0.4
        indices, heading = rotation_search_ranking(responses, angles, top_k=2)
        self.assertEqual(int(indices[0, 0]), 1)
        self.assertTrue(torch.allclose(heading, torch.tensor([-math.pi / 2]), atol=1e-4))

    def test_circular_mean_wraps_across_zero(self):
        angles = rotation_steps_rad(10)
        responses = torch.zeros(1, 36)
        responses[0, 0] = 1.0
        responses[0, -1] = 1.0
        heading = circular_heading_from_responses(responses, angles)
        # Equal weight at warp 0 and 350 degrees is a warp of -5 degrees, so heading is +5.
        self.assertAlmostEqual(float(torch.rad2deg(heading)), 5.0, delta=0.05)
        error = absolute_heading_error_rad(heading, torch.deg2rad(torch.tensor(5.0)))
        self.assertLess(float(error), math.radians(0.05))

    def test_negative_cosine_is_shifted_before_weighting(self):
        angles = torch.deg2rad(torch.tensor([0.0, 10.0, 20.0]))
        responses = torch.tensor([[-1.0, 0.0, -0.5]])
        heading = circular_heading_from_responses(responses, angles)
        weights = torch.tensor([0.0, 2.0 / 3.0, 1.0 / 3.0])
        expected_warp = torch.atan2(
            (weights * torch.sin(angles)).sum(), (weights * torch.cos(angles)).sum(),
        )
        self.assertAlmostEqual(float(heading), float(-expected_warp), places=5)

    def test_heading_summary_follows_val_and_test_order(self):
        errors = torch.deg2rad(torch.tensor([0.0, 10.0, 30.0]))
        summary = summarize_heading_errors(["val", "test", "test"], errors)
        self.assertAlmostEqual(summary["val"]["heading_error_deg_mean"], 0.0, delta=1e-4)
        self.assertAlmostEqual(summary["test"]["heading_error_deg_mean"], 20.0, delta=1e-4)
        self.assertAlmostEqual(summary["test"]["heading_error_deg_median"], 10.0, delta=1e-4)
        nested = attach_heading_errors({"val": {"recall@1": 1.0}, "test": {"recall@1": 0.5}}, summary)
        self.assertAlmostEqual(nested["test"]["heading_error_deg_mean"], 20.0, delta=1e-4)
        flat = attach_heading_errors({"recall@1": 0.5}, {"test": summary["test"]})
        self.assertAlmostEqual(flat["heading_error_deg_median"], 10.0, delta=1e-4)


if __name__ == "__main__":
    unittest.main()
