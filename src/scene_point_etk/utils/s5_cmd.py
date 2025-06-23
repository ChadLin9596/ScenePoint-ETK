import subprocess


def s5_cmd(*args):
    default = ["s5cmd", "--no-sign-request"]
    return subprocess.run(default + list(args), stdout=subprocess.PIPE)


def s5_ls(path="s3://argoverse/datasets/av2/"):
    x = s5_cmd("ls", path)
    x = x.stdout.decode("utf-8")
    x = [i.strip() for i in x.splitlines()]
    return x


def s5_cp(src, dst):
    return s5_cmd("cp", src, dst)
