# Dataset Bias Analysis Framework

<p align="center">
  <img src="assets/subpup.png" align="center" width="80%">
</p>

## Overview

**Dataset Bias Analysis (DBA)** is a framework for correcting the bias and discrepancies between the training set and the testing set for a higher test set performance given the prevalent of _subpopulation shift_.
This repo conains the implementation code of the framework, which is published as a the paper [Boosting Test Performance with Importance Sampling--a Subpopulation Perspective](https://arxiv.org/pdf/2412.13003) (Shen et al., AAAI 2025).


## Preparation
Note that the implementation of the DBA method is based on a hacking of the repo published with the paper [Change is Hard: A Closer Look at Subpopulation Shift](https://arxiv.org/abs/2302.12254) (Yang et al., ICML 2023). If you use this repo, please consider citing both the DBA paper and this one. For details, please [go to the Citation section](#Citation).

### Installation
Run the following commands to create a conda environment for running this code:

```bash
git clone git@github.com:skyve2012/DBA.git
cd DBA/
conda env create -f subpopulation_env.yml
```

Alternatively, one can also refer to the [Change is Hard](https://github.com/YyzHarry/SubpopBench/blob/main/README.md#:~:text=Run%20the%20following%20commands%20to%20clone%20this%20repo%20and%20create%20the%20Conda%20environment:) repo to install the package there and refer to [`subpopulation_env.yml`](./subpopulation_env.yml) for other missing packages.

### Download datasets
There are three datasets discussed in the paper:
* CMNIST ([Nam et al. 2020; Tsirigotis et al. 2024](https://proceedings.neurips.cc/paper/2020/file/eddc3427c5d77843c2253f1e799fe933-Paper.pdf))
* Waterbirds ([Wah et al., 2011](https://authors.library.caltech.edu/27452/))
* CivilComments ([Borkan et al., 2019](https://arxiv.org/abs/1903.04561)) from the [WILDS benchmark](https://arxiv.org/abs/2012.07421)

To facilitate the implementation, the datasets can be found at the [Google Drive]().

## Obtain Sample Weights


## Correct the Bias



## Citation

```bib
@inproceedings{shen2025subpopulation,
  title={{Boosting Test Performance with Importance Sampling--a Subpopulation Perspective}},
  author={Shen, Hongyu and Zhao, Zhizhen},
  booktitle={he Association for the Advancement
of Artificial Intelligence},
  year={2025}
}
```

```bib
@inproceedings{yang2023change,
  title={Change is Hard: A Closer Look at Subpopulation Shift},
  author={Yang, Yuzhe and Zhang, Haoran and Katabi, Dina and Ghassemi, Marzyeh},
  booktitle={International Conference on Machine Learning},
  year={2023}
}
```
