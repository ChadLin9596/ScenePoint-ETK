Argoverse2
==========

.. automodule:: scene_point_etk.argoverse2
    :no-members:

Functions
---------

Module Level
^^^^^^^^^^^^

.. autofunction:: scene_point_etk.argoverse2.set_sensor_root

Log IDs
^^^^^^^

.. autofunction:: scene_point_etk.argoverse2.check_log_id

.. autofunction:: scene_point_etk.argoverse2.list_log_ids

.. autofunction:: scene_point_etk.argoverse2.list_log_ids_by_mode

.. autofunction:: scene_point_etk.argoverse2.get_city_from_log_id

Cities
^^^^^^

.. autofunction:: scene_point_etk.argoverse2.check_city

.. autofunction:: scene_point_etk.argoverse2.list_cities

.. autofunction:: scene_point_etk.argoverse2.get_log_ids_from_city

Images and Cameras
^^^^^^^^^^^^^^^^^^

.. autofunction:: scene_point_etk.argoverse2.list_cameras_by_log_id

Lidar Sweeps
^^^^^^^^^^^^

.. autofunction:: scene_point_etk.argoverse2.list_sweep_files_by_log_id

Classes
-------

Images and Cameras
^^^^^^^^^^^^^^^^^^

.. autoclass:: scene_point_etk.argoverse2.ImageSequence
    :members: figsize, intrinsic, extrinsic, get_an_image, resize, get_a_depth_map, align_timestamps

.. autoclass:: scene_point_etk.argoverse2.CameraSequence
    :members: get_a_camera, set_a_camera, resize, align_timestamps

Lidar Sweeps
^^^^^^^^^^^^

.. autoclass:: scene_point_etk.argoverse2.Sweep

.. autoclass:: scene_point_etk.argoverse2.SweepSequence


3D Annotations
--------------

.. autoclass:: scene_point_etk.argoverse2.Annotations
