# Synthetic Data Generation for Time Series Imputation: Comparing the Foundation Model Chronos with Established Methods

**Abstract:** Accurately imputing missing data is critical in time series analysis. The present work compares Foundation Model Chronos against Linear Interpolation, K-Nearest Neighbor Imputer, and Gaussian Mixture Model Imputer with three types of missing data patterns: random, short sequential chunks, and a long sequential chunk. These results confirm that for random missing values, KNN and interpolation yield the highest performance, while Chronos outperforms these on sequences. Indeed, however, for longer sequences of missing values, Chronos starts suffering from cascading errors which eventually allow the simpler imputation methods to outrank it. Another test with limited quantities of training data showed different trade-offs for the different methods. Unlike KNN and interpolation, which smooth out the gaps, Chronos generates variable synthetic data. This can be beneficial in tasks which require control or simulation. The results highlight the strengths and weaknesses of the imputers and, therefore, offer practical insights into trade-offs between computational complexities, accuracy, and suitability for time series imputation scenarios.

# Setup

The version of the operating system used to develop this project is:
- Ubuntu 24.04.1 LTS

Python Version:
- 3.12

### Hardware used

- CPU: AMD EPYC-Milan processor (14 cores, 2.449 GHz)
- GPU: NVIDIA RTX 6000 Ada Generation
- 112 GB of RAM

### Requirements

To keep everything organized and simple,
we will use [MiniConda](https://docs.conda.io/projects/miniconda/en/latest/) to manage our environments.

To create an environment with the required packages for this project, run the following commands:

```bash
conda create -n chronos_imputer python=3.12
```

Then we need to install the requirements:

```bash
pip3 install -r requirements.txt
```

# Results

You can see the results in the [pdf file](paper.pdf).
 
