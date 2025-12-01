00 Installing scene_point_etk
=============================

Prepare Python Environment
--------------------------

.. code-block:: bash

    # create a python environment from (3.7 - 3.12) virtual environment
    $ source <directory of virtual environment>/bin/activate
    # or
    $ conda create -n <your env name> python=3.9
    $ conda activate <your env name>
    (your env)$


(Optional) Install **py_utils** in your workspace
-------------------------------------------------

*scene_point_etk* will automatically install **py_utils** from GitHub.
However, if you want to modify **py_utils** code for your own purpose,
you can manually install it in your workspace:

.. code-block:: bash

    (your env)$ git clone https://github.com/ChadLin9596/python_utils
    (your env)$ pip install -e ./python_utils

Install **scene_point_etk** in your workspace
---------------------------------------------

.. code-block:: bash

    # clone main repo
    (your env)$ git clone https://github.com/ChadLin9596/ScenePoint-ETK
    (your env)$ pip install -e ./ScenePoint-ETK

(Optional) Install **pptk** from my modified wheel release
----------------------------------------------------------

*scene_point_etk* uses **pptk** for point cloud visualization by default.
However, the official **pptk** PyPI wheel release only supports up to Python 3.7.
If you are using Python 3.8 or above, you can install **pptk**
from my modified wheel release `ChadLin9596/pptk/wheels`_.

.. code-block:: bash

    # An example for Python 3.9
    (your env)$ pip install https://github.com/ChadLin9596/pptk/releases/download/v0.1.1/pptk-0.1.1-cp39-none-manylinux_2_35_x86_64.whl

+--------+----------+-------------------------------------------------+
| python | platform | wheel file                                      |
+========+==========+=================================================+
| 3.7    | linux    | pptk-0.1.1-cp37-none-manylinux_2_35_x86_64.whl  |
+--------+----------+-------------------------------------------------+
| 3.8    | linux    | pptk-0.1.1-cp38-none-manylinux_2_35_x86_64.whl  |
+--------+----------+-------------------------------------------------+
| 3.9    | linux    | pptk-0.1.1-cp39-none-manylinux_2_35_x86_64.whl  |
+--------+----------+-------------------------------------------------+
| 3.10   | linux    | pptk-0.1.1-cp310-none-manylinux_2_35_x86_64.whl |
+--------+----------+-------------------------------------------------+
| 3.11   | linux    | pptk-0.1.1-cp311-none-manylinux_2_35_x86_64.whl |
+--------+----------+-------------------------------------------------+
| 3.12   | linux    | pptk-0.1.1-cp312-none-manylinux_2_35_x86_64.whl |
+--------+----------+-------------------------------------------------+
| 3.9    | mac      | pptk-0.1.1-cp39-none-macosx_15_0_x86_64.whl     |
+--------+----------+-------------------------------------------------+

References
----------

py_utils (python utils)
^^^^^^^^^^^^^^^^^^^^^^^

It is a utility library for python projects. It contains various helper functions for file I/O, data processing, visualization, etc.

GitHub Repository: `py_utils`_

Documentation Link: `py_utils documentation`_

pptk (point cloud toolkit)
^^^^^^^^^^^^^^^^^^^^^^^^^^

The Point Processing Toolkit (pptk) is a Python package for visualizing and processing 2-d/3-d point clouds.

GitHub Repository: `heremaps/pptk`_

Documentation Link: `heremaps/pptk documentation`_

Official PyPI Wheel Release (supports py27 - py37): `heremaps/pptk/wheels`_

My Modified Wheel Release (supports py37 - py312): `ChadLin9596/pptk/wheels`_

.. _py_utils: https://github.com/ChadLin9596/python_utils/
.. _py_utils documentation: https://chadlin9596.github.io/python_utils/
.. _heremaps/pptk: https://github.com/heremaps/pptk
.. _heremaps/pptk documentation: https://heremaps.github.io/pptk/
.. _heremaps/pptk/wheels: https://pypi.org/project/pptk/#files
.. _ChadLin9596/pptk/wheels: https://github.com/ChadLin9596/pptk/releases/tag/v0.1.1
