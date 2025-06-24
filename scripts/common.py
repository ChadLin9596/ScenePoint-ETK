import os
import pickle
import time

VERBOSE = False
PREFIX = " " * 4


def xprint(*args):
    if VERBOSE:
        print(PREFIX, *args)


def pickle_dump(obj, path):
    with open(path, "wb") as fd:
        pickle.dump(obj, fd)


def pickle_load(path):
    with open(path, "rb") as fd:
        obj = pickle.load(fd)
    return obj


def report_time(func):

    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        xprint(f"Elapsed time: {elapsed:.2f} sec")
        return result

    return wrapper


def ensure_output_dir_exists(func):

    def wrapper(log_id, output_dir, **kwargs):
        os.makedirs(output_dir, exist_ok=True)
        return func(log_id, output_dir, **kwargs)

    return wrapper
