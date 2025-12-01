Scene Data Base
===============

Functions
---------

Module Level
^^^^^^^^^^^^

.. autofunction:: scene_point_etk.scene_db.set_scene_root

Scene IDs
^^^^^^^^^

.. autofunction:: scene_point_etk.scene_db.list_scene_ids

.. autofunction:: scene_point_etk.scene_db.list_versions_by_scene_id

.. autofunction:: scene_point_etk.scene_db.list_scene_version_pairs

Classes
-------

Basic Scene structure::

    ├── <Scene ID 00>
    │   │
    │   └── <version name>
    │       ├── details.pkl
    │       ├── scene.pcd
    │       └── cameras
    │           ├── cam_sequence.pkl
    │           ├── <camera name 1>
    │           │   └── sparse_point_indices
    │           │       ├── <point indices 1>.npy
    │           │       └── ...
    │           │
    │           ├── <camera name 2>
    │           └── ...
    │
    ├── <Scene ID 01>
    └── ...

.. autoclass:: scene_point_etk.scene_db.scene.OriginalScene

.. autoclass:: scene_point_etk.scene_db.scene.EditedScene