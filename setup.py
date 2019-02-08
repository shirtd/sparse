from setuptools import setup

if __name__ == '__main__':
    setup(name = "sparse",
        version = "0.0.1",
        author = "Kirk Gardner",
        author_email = "kirk.gardner@uconn.edu",
        description = "Sparse Filtrations in Python.",
        long_description = "An implementation of sparse (rips) filtrations "\
                            "from A Geometric Perspective on Sparse Filtrations "\
                            "(https://arxiv.org/abs/1506.03797).",
        install_requires = ["cython", "numpy", "scipy",
                            "sklearn", "argparse","tqdm",
                            "matplotlib", "dionysus"],
        url = "https://github.com/shirtd/sparse")
