import Orange
import os
import pandas as pd

# import myconfig


def get_binarized_dataset_by_orange(csv_file, save_file, force=False):


    data = Orange.data.Table(csv_file)

    imputer = Orange.preprocess.Impute()
    data = imputer(data)

    discretizer = Orange.preprocess.Discretize()
    # discretizer.method = Orange.preprocess.discretize.EntropyMDL(force=False)
    discretizer.method = Orange.preprocess.discretize.EntropyMDL(force=force)        # NOTE: "force=False" may remove some original feature when no suitable cut-off points found 

    discetized_data = discretizer(data)

    continuizer = Orange.preprocess.Continuize()
    binarized_data = continuizer(discetized_data)

    new_col_name_s = []
    for i in range(len(binarized_data.domain)-1):
        col_name = binarized_data.domain[i].name

        if "=<" in col_name:
            new_col_name = col_name.replace("=< ", '<')
        elif "=≥" in col_name:
            new_col_name = col_name.replace("=≥ ", '>=')
        elif " - " in col_name:
            col_name = col_name.replace("=", '=[')
            col_name = col_name.replace(" - ", '-')
            new_col_name = col_name + ')'
        else:
            new_col_name = col_name
        new_col_name = new_col_name.replace(" ", '_')
        
        new_col_name_s.append(new_col_name)
    new_col_name_s.append("label")

    # convert the object of Table to DataFrame
    try:
        df = pd.DataFrame(binarized_data, columns=new_col_name_s, dtype=int)
    except:
        df = pd.DataFrame(binarized_data, columns=new_col_name_s)  
        df = df.dropna() # drop an instance hasing missing value in the label column 
        df = df.applymap(int)

    df = df.drop_duplicates() # NOTE: drop the duplicate rows, keeping only their first occurrences

    print(df.head())

    df.to_csv(save_file, index=False)  




if __name__ == '__main__':


    # datasets = ["adult"]  
    large_datasets = ["magic04", "adult", "sec_mushroom", "bank_marketing", "higgs", "weatherAUS", "BNG_labor", "BNG_credit-g"] 
    small_datasets = ["christine", "musk2", "heloc"] 
    datasets = large_datasets + small_datasets

    # data_path = "./datasets"

    for dataset in datasets:

        data_path = "./datasets/large_datasets" if dataset in large_datasets else "./datasets/small_datasets"

        if dataset in ["christine"]: # NOTE：for dataset "christine", to keep all the original features, we set "force=True"
            force=True
        else:
            force=False
            
        dataset_file = f"{data_path}/{dataset}.csv"
        save_binarized_file = f"{data_path}/{dataset}_bin.csv"

        if not os.path.exists(save_binarized_file):
            print(f"generating {save_binarized_file} ...")
            get_binarized_dataset_by_orange(dataset_file, save_binarized_file, force)
