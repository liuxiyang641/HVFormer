# Hierarchical Visual Context Fusion Transformer

The source code for **An Empirical Study of Hierarchical Visual Context Learning in Multimodal Relation Extraction**.

## Data preprocessing

### MNRE dataset

Due to the large size of MNRE dataset, please download the dataset from the [original repository](https://github.com/thecharm/MNRE). 

Unzip the data and rename the directory as `mnre`, which should be placed in the directory `data`:

```bash
mkdir data logs ckpt
```

We also use the detected visual objects provided in [previous work](https://github.com/zjunlp/MKGformer), which can be downloaded using the commend:

```bash
cd data/
wget 120.27.214.45/Data/re/multimodal/data.tar.gz
tar -xzvf data.tar.gz
```

## Dependencies

Install all necessary dependencies:

```bash
pip install -r requirements.txt
```

## Training the model

The best hyperparameters we found have been witten in `run_mre.sh` file.

You can simply run the bash script for multimodal relation extraction:

```bash
bash run_mre.sh
```

