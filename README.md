# Malaria detection

## Table of contents

* [Description](#description)
* [Getting Started](#getting-started)
* [Usage](#usage)

## Description

Repository contains whole training pipeline using own architecture of CNN and Xception transfer learning on [Cell images for detecting malaria dataset](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria).

**Files:**

1. Scripts
    * data_loading.py - split data into train/test, create image dataset from directory
    * model.py - baseline cnn model structure
    * cnn_training.py - training baseline cnn model pipeline
    * xception_transfer_learning.py - training xception model using transfer learning
2. XCEPTION / CNN_BASELINE - folders created after training to save models inside
3. cell_images (dataset structure template)
    * Parasitized
    * Uninfected

## Getting Started

### Quick start

Tested with python 3.8.3

Libraries used:
    * tensorflow

You can install all using pip.

```bash
pip install -r requirements.txt
```

Download dataset from Kaggle
<https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria>

## Usage

Split train test. If you followed data sctructure above, you can now use train_test_split() function from data_loading.py

```python
python
>>> import data_loading
>>> data_loading.train_test_split("cell_images", "dataset")
```

If u want to **TRAIN BASELINE CNN**

```bash
python cnn_training.py
```

If u want to **TRAIN XCEPTION** with transfer learning.

```bash
python xception_transfer_learning.py
```
