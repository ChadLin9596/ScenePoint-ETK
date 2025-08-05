from setuptools import setup, find_packages

setup(
    name="scene_point_etk",
    version="0.1.0",
    description="",
    author="Chad Lin",
    url="https://github.com/ChadLin9596/ScenePoint-ETK",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "pyarrow",
        "scikit-learn",
    ],
)
