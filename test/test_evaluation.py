import unittest
import numpy as np
import scene_point_etk.evaluation as evaluation


class TestMapUpdating(unittest.TestCase):

    def setUp(self):
        # Define two small point clouds with slight offset
        self.pc1 = np.random.rand(50, 3)
        self.pc2 = self.pc1 + 0.02

    def test_nearest_distance(self):
        dist = evaluation.nearest_distance(self.pc1, self.pc2)
        self.assertEqual(dist.shape, (self.pc1.shape[0],))
        self.assertTrue(np.all(dist >= 0))

    def test_chamfer_distance(self):
        cd = evaluation.chamfer_distance(self.pc1, self.pc2)
        self.assertIsInstance(cd, float)
        self.assertGreaterEqual(cd, 0)

    def test_hausdorff_distance(self):
        hd = evaluation.hausdorff_distance(self.pc1, self.pc2)
        self.assertIsInstance(hd, float)
        self.assertGreaterEqual(hd, 0)

    def test_modified_hausdorff_distance(self):
        mhd = evaluation.modified_hausdorff_distance(self.pc1, self.pc2)
        self.assertIsInstance(mhd, float)
        self.assertGreaterEqual(mhd, 0)

    def test_median_point_distance(self):
        mpd = evaluation.median_point_distance(self.pc1, self.pc2)
        self.assertIsInstance(mpd, float)
        self.assertGreaterEqual(mpd, 0)

    def test_all_point_cloud_metrics(self):
        results = evaluation.all_point_cloud_metrics(self.pc1, self.pc2)
        self.assertIsInstance(results, dict)
        expected_keys = {
            "chamfer_dist",
            "hausdorff_dist",
            "modified_hausdorff_dist",
            "median_point_dist",
        }
        self.assertEqual(set(results.keys()), expected_keys)
        for value in results.values():
            self.assertIsInstance(value, float)
            self.assertGreaterEqual(value, 0)


if __name__ == "__main__":
    unittest.main()
