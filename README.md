# Scene Point Editing Toolkit (ScenePoint-ETK)

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
$ cd <this repository>
$ pip install .

# Note:
# it will automatically install py_utils from https://github.com/ChadLin9596/python_utils
```

### Example Usage

* unittest
    ``` bash
    $ cd <this repository>
    $ python -m unittest
    ```
* tutorial jupyter-notebook under `examples`

### TODO

* [ ] complete tutorials at examples
* [ ] website deployment
    * [ ] project page
    * [ ] documentation
* [ ] release v0 dataset
* [ ] more unittests
