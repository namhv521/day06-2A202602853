"""CPU checks: python -m unittest src.test_obstacles -v."""
import unittest
import numpy as np
from src.obstacles import pipeline, low_scene, occupancy
from starter.datasets import load_points


class ObstacleChecks(unittest.TestCase):
    def test_seeded_real_frame_repeats_exact_geometry(self):
        points = load_points('data/kitti_mini', '000001')
        baseline = pipeline(points)
        for _ in range(5):
            repeated = pipeline(points)
            np.testing.assert_array_equal(baseline['plane'], repeated['plane'])
            np.testing.assert_array_equal(baseline['labels'], repeated['labels'])

    def test_low_obstacle_is_lost_with_wide_ground_band(self):
        points = low_scene()
        fine = pipeline(points, .05, .03)
        wide = pipeline(points, .05, .30)
        self.assertEqual(len(fine['boxes']), 2)
        self.assertEqual(len(wide['boxes']), 1)
        self.assertLess(fine['nearest_m'], wide['nearest_m'])
        self.assertGreater(fine['plane'][2], .9)

    def test_empty_invalid_and_occupancy(self):
        result = pipeline(np.array([[np.nan, 0, 0], [-1, 0, 0]]), .1, .05)
        self.assertEqual(len(result['boxes']), 0)
        self.assertTrue(np.isnan(result['nearest_m']))
        grid = occupancy(np.array([[0., 0, 1], [29.99, 11.99, 0], [30, 0, 0]]))
        self.assertEqual(grid.sum(), 2)
        with self.assertRaises(ValueError):
            pipeline(np.zeros((5, 3)), 0, .05)


if __name__ == '__main__':
    unittest.main()
