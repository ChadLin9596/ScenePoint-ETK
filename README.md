# Scene Point Editing Toolkit (ScenePoint-ETK)

---
**NOTE**

This repository is under active update.

---


Scene Point Editing Toolkit (ScenePoint-ETK) is a python package to build various city scenes using Argoverse2 dataset

![figure1](media/main.figure.1.png)

| ![figure2](media/main.figure.3.create_pipeline.1.png) | ![figure2](media/main.figure.3.create_pipeline.2.png) | ![figure2](media/main.figure.3.create_pipeline.3.png) | ![figure2](media/main.figure.3.create_pipeline.4.png) | ![figure2](media/main.figure.3.create_pipeline.5.png) | ![figure2](media/main.figure.3.create_pipeline.6.png) |
| ----------------------------------------------------- | ----------------------------------------------------- | ----------------------------------------------------- | ----------------------------------------------------- | ----------------------------------------------------- | ----------------------------------------------------- |

### Paper

* [arXiv link](https://arxiv.org/abs/2511.15153)


### Installation

```bash
# clone main repo
$ git clone https://github.com/ChadLin9596/ScenePoint-ETK

# create a python environment from (3.7 - 3.12) virtual environment
$ source <directory of virtual environment>/bin/activate
(your env)$ cd <this repository>
(your env)$ pip install .

# Note:
# it will automatically install py_utils from https://github.com/ChadLin9596/python_utils

# install `pptk` from https://github.com/ChadLin9596/pptk/releases
# for example (py3.9)
(your env)$ pip install https://github.com/ChadLin9596/pptk/releases/download/v0.1.1/pptk-0.1.1-cp39-none-manylinux_2_35_x86_64.whl
```

### Example Usage

* unittest
    ``` bash
    (your env)$ cd <this repository>
    (your env)$ python -m unittest
    ```
* tutorial jupyter-notebook under `examples`

### Dataset Preparation

Please refer to [Download SceneEdited Dataset](https://chadlin9596.github.io/ScenePoint-ETK/docs/build/html/download.html) page in the [Documentation](https://chadlin9596.github.io/ScenePoint-ETK/docs/build/html/index.html)

Run `ScenePoint-ETK/examples/01.setup_env_path.ipynb` to assign their directories into this toolkit.

### TODO

* [ ] complete tutorials at examples
    * [x] basic usage
    * [ ] scalibitiy
    * [ ] trackability
    * [x] portability
* [ ] website deployment
    * [ ] project page
    * [ ] documentation
* [x] release v0 dataset
* [ ] more unittests

### Citation

Please use the following citation for any work relies on the SceneEdited dataset or the ScenePoint-ETK toolkit.


```
@InProceedings{SceneEdited,
    author    = {Lin, Chun-Jung and Chin, Tat-Jun and Garg, Sourav and Dayoub, Feras},
    title     = {SceneEdited: A City-Scale Benchmark for 3D HD Map Updating via Image-Guided Change Detection},
    booktitle = {Proceedings of the Winter Conference on Applications of Computer Vision (WACV)},
    month     = {March},
    year      = {2026}
}
```
