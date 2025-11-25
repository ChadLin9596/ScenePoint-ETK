Download SceneEdited Dataset
============================

Download `Argoverse dataset`_ (900 GB)
--------------------------------------

.. code-block:: bash

    # dry run (fast testing, no file will be downloaded)
    (your env)$ python ScenePoint-ETK/scripts/00.download_sensor_dataset.py <where you want to store> --dry

    # actual download
    (your env)$ python ScenePoint-ETK/scripts/00.download_sensor_dataset.py <where you want to store>

file structure:

.. code-block:: bash

    <where you want to store>
    ├── 00a6ffc1-6ce9-3bc3-a060-6006e9893a1a
    │   ├── annotations.feather
    │   ├── calibration
    │   │   ├── egovehicle_SE3_sensor.feather
    │   │   └── intrinsics.feather
    │   ├── city_SE3_egovehicle.feather
    │   ├── map
    │   │   ├── 00a6ffc1-6ce9-3bc3-a060-6006e9893a1a_ground_height_surface____PIT.npy
    │   │   ├── 00a6ffc1-6ce9-3bc3-a060-6006e9893a1a___img_Sim2_city.json
    │   │   └── log_map_archive_00a6ffc1-6ce9-3bc3-a060-6006e9893a1a____PIT_city_31785.json
    │   └── sensors
    │       ├── cameras
    │       │   ├── ring_front_center
    │       │   │   ├── 315967376899927209.jpg
    │       │   │   ├── 315967376949927221.jpg
    │       │   │   └── ... (317 more files)
    │       │   ├── ring_front_left
    │       │   ├── ring_front_right
    │       │   ├── ring_rear_left
    │       │   ├── ring_rear_right
    │       │   ├── ring_side_left
    │       │   ├── ring_side_right
    │       │   ├── stereo_front_left
    │       │   └── stereo_front_right
    │       └── lidar
    │           ├── 315967376859506000.feather
    │           ├── 315967376959702000.feather
    │           └── ... (155 more files)
    ├── 01bb304d-7bd8-35f8-bbef-7086b688e35e
    ├── 022af476-9937-3e70-be52-f65420d52703
    └── ... (847 more files)

Download `SceneEdited-Patches`_ (26MB)
--------------------------------------

.. code-block:: bash

    $ wget --content-disposition https://ndownloader.figshare.com/files/59816492
    $ unzip SceneEdited-patches.zip

file structure:

.. code-block:: bash

    patches
    ├── BOLLARD
    │   ├── 0339f59e88.pcd
    │   └── ... (67 more files)
    ├── CONSTRUCTION_BARREL
    │   ├── 003304f27c.pcd
    │   └── ... (13 more files)
    ├── CONSTRUCTION_CONE
    │   └── 5a8f1bcf62.pcd
    ├── MESSAGE_BOARD_TRAILER
    │   ├── 039f2a05e2.pcd
    │   └── ... (6 more files)
    ├── MOBILE_PEDESTRIAN_CROSSING_SIGN
    │   ├── 1b7656f426.pcd
    │   └── ... (3 more files)
    ├── OFFICIAL_SIGNALER
    │   ├── 0f65618f69.pcd
    │   └── ... (4 more files)
    ├── SIGN
    │   ├── 006d09a99c.pcd
    │   └── ... (441 more files)
    ├── STOP_SIGN
    │   ├── 0040351bcf.pcd
    │   └── ... (705 more files)
    ├── TRAFFIC_LIGHT_TRAILER
    │   └── 5ee54ec11d.pcd
    ├── TRUCK_CAB
    │   ├── 026743ebb8.pcd
    │   └── ... (47 more files)
    └── VEHICULAR_TRAILER
        ├── 00b24fff49.pcd
        └── ... (171 more files)


Download `SceneEdited-Scenes`_ (46GB)
--------------------------------------

.. code-block:: bash

    $ wget --content-disposition https://ndownloader.figshare.com/files/59818052
    $ unzip SceneEdited-scenes.zip

file structure:

.. code-block:: bash

    Scenes
    ├── 00a6ffc16ce93bc3a0606006e9893a1a
    │   ├── GT
    │   │   └── details.pkl
    │   ├── v0.add
    │   │   └── details.pkl
    │   ├── v0.delete
    │   │   └── details.pkl
    │   ├── v0.manually_removed
    │   │   └── details.pkl
    │   └── v0.overall
    │       └── details.pkl
    ├── 01bb304d7bd835f8bbef7086b688e35e
    │   ├── GT
    │   │   └── details.pkl
    │   ├── v0.add
    │   │   └── details.pkl
    │   ├── v0.delete
    │   │   └── details.pkl
    │   └── v0.overall
    │       └── details.pkl
    ├── 022af47699373e70be52f65420d52703
    ├── 02678d04cc9f31489f951ba66347dff9
    └── ... (843 more files)

.. _Argoverse dataset: https://www.argoverse.org/
.. _SceneEdited-Patches: https://figshare.com/articles/dataset/SceneEdited_-_Patch_Data_Base/30702329?file=59816492
.. _SceneEdited-Scenes: https://figshare.com/articles/dataset/SceneEdited_-_Scene_Data_Base/30702929?file=59818052