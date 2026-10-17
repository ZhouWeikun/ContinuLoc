"""Pure checks for discrete query-rotation retrieval."""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from continuloc.evaluation.query_rotation_search import (
    absolute_heading_error_rad,
    attach_heading_errors,
    exponential_heading_from_responses,
    feature_chord_distance,
    normalized_base_grid,
    require_scene_mean_scale,
    resolve_forward_variant_limit,
    resolve_heading_temperature,
    resolve_retrieval_mode,
    rotation_scale_grid,
    rotation_search_ranking,
    rotation_steps_rad,
    summarize_heading_errors,
    variant_coordinates,
)


class SamplingGridTest(unittest.TestCase):
    def test_rotation_scale_grid_matches_affine_grid(self):
        base = normalized_base_grid(5, 7)
        rot = torch.tensor([0.3, -1.2, 0.0])
        scale = torch.tensor([1.0, 0.8, 1.4])
        grid = rotation_scale_grid(base, rot, scale)
        theta = torch.zeros(3, 2, 3)
        cos_v, sin_v = torch.cos(rot), torch.sin(rot)
        theta[:, 0, 0] = cos_v * scale
        theta[:, 0, 1] = sin_v * scale
        theta[:, 1, 0] = -sin_v * scale
        theta[:, 1, 1] = cos_v * scale
        reference = F.affine_grid(theta, (3, 1, 5, 7), align_corners=False)
        self.assertTrue(torch.allclose(grid, reference, atol=1e-6))
        image = torch.arange(3 * 5 * 7, dtype=torch.float32).reshape(3, 1, 5, 7)
        sampled = F.grid_sample(image, grid, align_corners=False, padding_mode="border")
        sampled_ref = F.grid_sample(image, reference, align_corners=False, padding_mode="border")
        self.assertTrue(torch.allclose(sampled, sampled_ref, atol=1e-5))

    def test_variants_are_query_major_and_forward_cap_is_512(self):
        query_index, angle_index = variant_coordinates(torch.tensor([0, 35, 36, 511]), 36)
        self.assertEqual(query_index.tolist(), [0, 0, 1, 14])
        self.assertEqual(angle_index.tolist(), [0, 35, 0, 7])
        self.assertEqual(resolve_forward_variant_limit(512, 32), 512)
        self.assertEqual(resolve_forward_variant_limit(None, 32), 32)
        with self.assertRaises(ValueError):
            resolve_forward_variant_limit(0, 32)


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

    def test_equal_peaks_across_zero_stay_local(self):
        angles = rotation_steps_rad(10)
        responses = torch.zeros(1, 36)
        responses[0, 0] = 1.0
        responses[0, -1] = 1.0
        heading = exponential_heading_from_responses(responses, angles, temperature=0.02)
        # Equal distance at warp 0 and 350 degrees averages to a warp of -5 degrees.
        self.assertAlmostEqual(float(torch.rad2deg(heading)), 5.0, delta=0.05)
        error = absolute_heading_error_rad(heading, torch.deg2rad(torch.tensor(5.0)))
        self.assertLess(float(error), math.radians(0.05))

    def test_sharp_temperature_stays_near_the_peak(self):
        angles = torch.deg2rad(torch.tensor([0.0, 10.0, 180.0]))
        responses = torch.tensor([[0.99, 0.98, 0.50]])
        heading = exponential_heading_from_responses(responses, angles, temperature=0.02)
        distance = feature_chord_distance(responses)
        weights = torch.softmax(-distance / 0.02, dim=-1)
        delta = torch.atan2(torch.sin(angles), torch.cos(angles))
        expected_warp = (weights * delta).sum()
        self.assertAlmostEqual(float(heading), float(-expected_warp), places=5)
        # The opposite angle has a much larger feature distance, so it does not pull the mean.
        self.assertLess(abs(float(torch.rad2deg(heading))), 1.0)
        self.assertGreater(float(weights[0, 0]), 0.9)
        with self.assertRaises(ValueError):
            resolve_heading_temperature(0)
        self.assertEqual(resolve_heading_temperature(None), 0.02)

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
