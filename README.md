# ROBP #

A Python implementation of " ROBP: A robust border-peeling clustering using Cauchy kernel"

## introduction ##

A Python implementation of the clustering algorithm presented in:

   <b><i> Mingjing Du, Ru Wang, Ru Ji, Xia Wang, Yongquan Dong. ROBP: A robust border-peeling clustering using Cauchy kernel. Information Sciences, 2021, 571: 375-400.</i></b>

The paper is available online at: <a href="https://dumingjing.github.io/files/paper-8_2021-05-04/paper.pdf" target="_blank">pdf</a>. 


If you use this implementation in your work, please add a reference/citation to the paper. You can use the following bibtex entry:

```
@article{du2021robp,
  title={ROBP: a robust border-peeling clustering using Cauchy kernel},
  author={Du, Mingjing and Wang, Ru and Ji, Ru and Wang, Xia and Dong, Yongquan},
  journal={Information Sciences},
  volume={571},
  pages={375--400},
  year={2021},
  doi={10.1016/j.ins.2021.04.089}
}
```

ROBP is an improved version of Border-Peeling Clustering (TPAMI,2020)

The  Python implementation of original paper (BP clustering) can be found  <a href="https://github.com/nadavbar/BorderPeelingClustering" target="_blank"> here </a>.


## Requirement ##

* Python >= 3.6
* numpy
* scipy
* scikit-learn
* python_algorithms


## Installation instructions ##

In our implementation we use the numpy, scipy and scikit-learn packages.
We recommend using the Anaconda python distribution which installs these packages automatically.

**Please note that the current code only supports python 3.6.**

For the UnionFind implementation We also use the python_algorithms package: https://pypi.python.org/pypi/python_algorithms

If you are using the pip package manager, you can install it by running the command:

```
pip install python_algorithms
```

## Usage instructions ##

In order to run  ROBP clustering, run the file run_robp.py using python.

The following command line arguments are available for run_robp.py:

```
usage: run_robp.py [-h] --input <file path> --output <file path> [--no-labels]
                 [--pca <dimension>] [--spectral <dimension>]


optional arguments:
  -h, --help            show this help message and exit
  --input <file path>   Path to comma separated input file
  --output <file path>  Path to output file
  --no-labels           Specify that input file has no ground truth labels
  --pca <dimension>     Perform dimensionality reduction using PCA to the
                        given dimension before running the clustering
  --spectral <dimension>
                        Perform spectral embedding to the given dimension
                        before running the clustering (If combined with PCA,
                        PCA is performed first)

```

