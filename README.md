
### Dataset

For a dataset with real-valued/categorical features, we need to first discretize and binarize these features to get a binary dataset.
Take the dataset `adult.csv` as example, this can be done by the following steps:
1. put the target dataset file `adult.csv` into the `datasets` folder;

2. modify the feature names as follows:

- For a categorial feature, add `D#` to the feature name. 
- For a real-valued feature, add `C#`. 
- For the target (discrete) column, add `cD#`. 
- To ignore any feature, add `i#` to the feature name. 

3. run '''python ./utils/binarize_dataset.py''' to get the binarized dataset `*_bin.csv`.

The modified feature names of three additional small datasets and all large datasets used in the paper can be found in the file `modified_feature_names.txt` in `datasets` folder.

**NOTE**:
The preprocessed datasets can be found in the `datasets/large_datasets_part.tar.gz.*` and `datasets/small_datasets.tar.gz` files. You can extract them as follows and then run the experiments.

```bash
cd OBDD-NET
cat ./datasets/large_datasets_part.tar.gz.* | tar -xzvf - -C ./datasets
tar -xzvf ./datasets/small_datasets.tar.gz -C ./datasets
```

### Requirements

The experiment was conducted on `Python 3.8`, and the requirements are summeried in the file `requirements.txt`.

```
pip install -r requirements.txt

```


### Code


Conduct experiment for 5-fold cross-validation on a small dataset `hypothyroid-un.csv` :

```
python ./experiment.py --timeout 900 --epoch 3000 --net_depth 6 --train_file  ./datasets/small_datasets/hypothyroid-un.csv

```

Conduct experiment for 5-fold cross-validation on a large dataset `adult_bin.csv` :

```
python ./experiment.py --timeout 1800 --epoch 800 --net_depth 6 --train_file  ./datasets/large_datasets/adult_bin.csv

```

The experimental results will be stored into the file in the `result` folder.




