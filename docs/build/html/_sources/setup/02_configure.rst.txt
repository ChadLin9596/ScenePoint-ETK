02 Configure scene_point_etk
============================

After installing and downloading the dataset, you need to configure
*scene_point_etk* to point to the dataset location on your machine. It
only need to be done once.

Argoverse Dataset
-----------------

.. code-block:: python

    >>> import scene_point_etk.argoverse2 as argoverse2
    >>> argoverse2.list_log_ids()
    []
    >>> argoverse2.set_data_root("<where you stored the dataset>")
    >>> argoverse2.list_log_ids()
    [
        '00a6ffc16ce93bc3a0606006e9893a1a',
        '01bb304d7bd835f8bbef7086b688e35e',
        '022af47699373e70be52f65420d52703',
        ...
    ]

SceneEdited - Patches
---------------------

.. code-block:: python

    >>> import scene_point_etk.patch_db as patch_db
    >>> patch_db.list_valid_patch_keys()
    []
    >>> patch_db.set_patch_root("<where you stored the dataset>")
    >>> patch_db.list_valid_patch_keys()
    [
        ('BOLLARD', '0339f59e88'),
        ('BOLLARD', '0554976c9c'),
        ('BOLLARD', '05a24ecb3d'),
        ...
    ]

SceneEdited - Scenes
--------------------

.. code-block:: python

    >>> import scene_point_etk.scene_db as scene_db
    >>> scene_db.list_scene_ids()
    []
    >>> scene_db.set_scene_root("<where you stored the dataset>")
    >>> scene_db.list_scene_ids()
    [
        '00a6ffc16ce93bc3a0606006e9893a1a',
        '01bb304d7bd835f8bbef7086b688e35e',
        '022af47699373e70be52f65420d52703',
        ...
    ]
    >>> scene_db.list_scene_version_pairs()
    [
        ('00a6ffc16ce93bc3a0606006e9893a1a', 'v0.add'),
        ('00a6ffc16ce93bc3a0606006e9893a1a', 'v0.delete'),
        ('00a6ffc16ce93bc3a0606006e9893a1a', 'v0.manually_removed'),
        ('00a6ffc16ce93bc3a0606006e9893a1a', 'v0.overall'),
        ('01bb304d7bd835f8bbef7086b688e35e', 'v0.add'),
        ...
    ]

For more details about configuration, please refer to `examples/01.setup_env_path.ipynb`_


.. _examples/01.setup_env_path.ipynb: https://github.com/ChadLin9596/ScenePoint-ETK/blob/main/examples/01.setup_env_path.ipynb
