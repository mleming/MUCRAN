# MUCRAN
## Multi-Confound Regression Adversarial Network

MUCRAN is a DL model applied to 3D images that is able to classify them while regressing out recorded confounds, thus ensuring that certain confounds are not unduly taken into account during the classification process.

[arXiv paper](https://arxiv.org/abs/2205.02885)

For example, let's say you're attempting to train a DL model to distinguish sex, given only the 3D MRIs of males and females, but you don't want it to take head size or age into account while doing so. This can be particularly important for imbalanced datasets (for example, in a training set in which male subjects have larger heads or a statistically significant age difference when compared to females). MUCRAN will be able to classify based on features unrelated to head size or age.

# Input

MUCRAN requires a dataset of 3D .npy files, along with a Pandas .pkl file with the paths to these .npy files as indices. The Pandas file of the above example would look something like this:

|                     | HEAD_SIZE   | AGE  | SEX |
| ------------------- | ----------- | ---- | --- |
| /path/to/mri1.npy   | 200.0       | 24.0 | M   |
| /path/to/mri2.npy   | 435.0       | 26.0 | F   |
| /path/to/mri3.npy   | 184.5       | 26.0 | F   |
| /path/to/mri4.npy   | 455.0       | 34.0 | F   |
| /path/to/mri5.npy   | 301.4       | 27.0 | M   |
| /path/to/mri6.npy   | 350.9       | 35.0 | F   |

# Run

An example bash file to run the main.py code is offered in run_main.sh

```bash
bash run_main.sh
```

If the Pandas file above were located at /path/to/vars_file.pkl, it would be run as such:

```bash
python main.py --/path/to/vars_file.pkl --label SEX --c HEAD_SIZE AGE
```

# Analysis

MUCRAN outputs its results as a number of JSON files. The ensemble_auroc.py script can be used to combine the results of a number of independent MUCRAN outputs into an ensemble.

```bash
python ensemble_auroc.py path/to/output_test1.json path/to/output_test2.json path/to/output_test3.json
```
