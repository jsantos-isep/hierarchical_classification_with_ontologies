import pandas as pd

def reading_dataset_from_file_in_chunks(input_dir, dataset_name, experience, chunk_size):
    total_rows = sum(1 for _ in open(f"{input_dir}/{dataset_name}_{experience}.csv")) - 1  # Subtract 1 for the header
    ddf = pd.read_csv(f"{input_dir}/{dataset_name}_{experience}.csv", sep=";", chunksize=chunk_size,
                      usecols=["title", "text", "category", "sub_category"])
    return ddf, total_rows

def reading_dataset_from_file(input_dir, dataset_name, dataset_type):
    total_rows = sum(1 for _ in open(f"{input_dir}/{dataset_name}_{dataset_type}.csv")) - 1  # Subtract 1 for the header

    df = pd.read_csv(f"{input_dir}/{dataset_name}_{dataset_type}.csv", sep=";",
                           usecols=["title", "text", "category", "sub_category"])
    return df, total_rows

def reading_dataset_from_file_without_total(input_dir, dataset_name, dataset_type):
    df = pd.read_csv(f"{input_dir}/{dataset_name}_{dataset_type}.csv", sep=";",
                           usecols=["title", "text", "category", "sub_category"])
    return df

def reading_category_dataset_from_file_without_total(input_dir, dataset_name, dataset_type, category):
    print(f"{input_dir}/{dataset_name}_{dataset_type}_{category}.csv")
    df = pd.read_csv(f"{input_dir}/{dataset_name}_{dataset_type}_{category}.csv", sep=";",
                           usecols=["title", "text", "category", "sub_category"])
    return df

def process_json(data, flag):
    if flag == "top":
        return list(data.keys())  # Return only the keys
    else:
        return [item for sublist in data.values() for item in sublist]  # Flatten values
