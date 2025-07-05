import os
import unittest

import numpy as np
import pandas as pd
import scene_point_etk.argoverse2 as argoverse2


class TestArgoverse2(unittest.TestCase):

    test_log_id = "00a6ffc16ce93bc3a0606006e9893a1a"

    def test_check_log_id(self):

        argoverse2.check_log_id(self.test_log_id)

    def test_get_root_by_log_id(self):

        root = argoverse2.get_root_from_log_id(self.test_log_id)
        self.assertTrue(os.path.exists(root))

    def test_get_poses_by_log_id(self):

        poses = argoverse2.get_poses_by_log_id(self.test_log_id)

        self.assertTrue("timestamp" in poses)
        self.assertTrue("xyz" in poses)
        self.assertTrue("q" in poses)

        self.assertEqual(poses["timestamp"].shape, (2689,))
        self.assertEqual(poses["xyz"].shape, (2689, 3))
        self.assertEqual(poses["q"].shape, (2689, 4))

    def test_get_extrinsic_by_log_id(self):

        extrinsic = argoverse2.get_extrinsic_by_log_id(self.test_log_id)

        self.assertIsInstance(extrinsic, pd.DataFrame)

        extrinsic_keys = [
            "sensor_name",
            "qw",
            "qx",
            "qy",
            "qz",
            "tx_m",
            "ty_m",
            "tz_m",
        ]

        self.assertEqual(extrinsic.columns.tolist(), extrinsic_keys)

        sensor_name = [
            "ring_front_center",
            "ring_front_left",
            "ring_front_right",
            "ring_rear_left",
            "ring_rear_right",
            "ring_side_left",
            "ring_side_right",
            "stereo_front_left",
            "stereo_front_right",
            "up_lidar",
            "down_lidar",
        ]

        self.assertEqual(extrinsic["sensor_name"].tolist(), sensor_name)

    def test_get_intrinsic_by_log_id(self):

        intrinsic = argoverse2.get_intrinsic_by_log_id(self.test_log_id)

        self.assertIsInstance(intrinsic, pd.DataFrame)

        intrinsic_keys = [
            "sensor_name",
            "fx_px",
            "fy_px",
            "cx_px",
            "cy_px",
            "k1",
            "k2",
            "k3",
            "height_px",
            "width_px",
        ]

        self.assertListEqual(intrinsic.columns.tolist(), intrinsic_keys)

        sensor_name = [
            "ring_front_center",
            "ring_front_left",
            "ring_front_right",
            "ring_rear_left",
            "ring_rear_right",
            "ring_side_left",
            "ring_side_right",
            "stereo_front_left",
            "stereo_front_right",
        ]

        self.assertEqual(intrinsic["sensor_name"].tolist(), sensor_name)
