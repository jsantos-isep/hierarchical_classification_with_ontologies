import os
import json
import os
import re
import time

import pandas as pd
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from huggingface_hub import login
from scipy.constants import precision
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, \
    pipeline

from hierarchical_classification_with_ontologies.functions import general, args_setup, manage_datasets

load_dotenv()
device = general.device()
login(os.getenv("HUGGING_FACE_TOKEN"))

def classify_partition(args, df_partition, model, tokenizer, label_encoder):
    """Classifies a partition of data and tracks confidence"""
    if args.dataset != "openalex":
        inputs = tokenizer(df_partition["text"].tolist(), return_tensors="pt", padding=True, truncation=True).to(device)
    else:
        inputs = tokenizer(df_partition["title"].tolist(), return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1)

    df_partition["confidence"] = probs.max(dim=-1).values.cpu().numpy()
    df_partition["predicted_label_num"] = probs.argmax(dim=-1).cpu().numpy()
    df_partition["predicted_label"] = label_encoder.inverse_transform(probs.argmax(dim=-1).cpu().numpy())

    # Get the number of available classes
    num_classes = logits.shape[1]

    # Set k (ensure k is at most the number of classes)
    k = min(5, num_classes)

    top_5_probs, top_5_indices = torch.topk(probs, k, dim=1)
    df_partition["top_5_predicted_num"] = top_5_indices.cpu().numpy().tolist()
    # Convert indices to class labels
    top_5_labels = label_encoder.inverse_transform(top_5_indices.cpu().numpy().flatten())  # Convert to labels
    top_5_labels = top_5_labels.reshape(top_5_indices.shape)  # Reshape back to original shape

    # Convert to list of lists for Pandas storage
    df_partition["top_5_predicted"] = top_5_labels.tolist()

    return df_partition

def generative_partition(args, df_partition, model, tokenizer, candidate_labels):
    if args.dataset != "openalex":
        prompts = [f"""Analyze the following text and explain what it is about. In the final, classify this document into one of these categories: {', '.join(candidate_labels)}.

            Text: "{doc.text}"
            
            Respond in this format:
            Reasoning: <your explanation>
            Category: <one of the listed categories>"""  for doc in df_partition.itertuples()]
    else:
        prompts = [f"""
                Classify this title into one of these categories: {', '.join(candidate_labels)}

                Title: {doc.title}

                Label:
                """ for doc in df_partition.itertuples()]
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50)
    labels = []
    for o in outputs:
        decoded_output = tokenizer.decode(o, skip_special_tokens=True)
        #print(decoded_output)

        # Split by "Label:" and extract the part after it
        #label = decoded_output.split("Category:")[-1].strip()
        match = re.search(r"Category:\s*(\w+)", decoded_output, re.IGNORECASE)
        label = match.group(1) if match else ""

        # Split by commas and check which labels are in the list of candidate_labels
        possible_labels = label.split(',')

        # Find the first valid label that is in candidate_labels
        first_valid_label = None
        for possible_label in possible_labels:
            cleaned_label = possible_label.strip()
            if cleaned_label in candidate_labels:
                first_valid_label = cleaned_label
                break

        # Append the first valid label (if any)
        if first_valid_label:
            labels.append(first_valid_label)
        else:
            labels.append("")  # In case no valid label is found

    return labels

def top_5_accuracy(true_labels, top_5_labels):
    correct = 0
    for i in range(len(true_labels)):
        if true_labels[i] in top_5_labels[i]:
            correct += 1
    return correct / len(true_labels)

def process_json(data, flag):
    if flag == "top":
        return list(data.keys())  # Return only the keys
    else:
        return [item for sublist in data.values() for item in sublist]  # Flatten values



if __name__ == '__main__':
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    start_time = time.time()
    args = args_setup.get_arg_parser_baseline().parse_args()
    #logging_config.create_log_file(f"baseline_{args.dataset}_{str(args.model_path).split('/')[-1]}_{args.model_type}_{args.label_level}")

    #logging.info(f"Running into {device}")

    # Read dataset
    input_dir = os.getenv(f"DATASETS_FOLDER_{str(args.dataset).upper()}")
    if args.model_type == "generative":
        size_per_chunk = 1
    else:
        if args.dataset == "the_guardian":
            size_per_chunk = 250
        elif args.dataset == "openalex":
            size_per_chunk = 150
        else:
            size_per_chunk = 50

    ddf, total_rows = manage_datasets.reading_dataset_from_file_in_chunks(input_dir=input_dir, dataset_name=args.dataset, experience="test", chunk_size=size_per_chunk)

    with open(f"{input_dir}/initial_ontology.json", "r") as file:  # Replace 'data.json' with your actual file path
        data = json.load(file)
    candidate_labels = process_json(data, args.label_level)

    label_encoder = LabelEncoder()
    label_encoder.fit(candidate_labels)

    if args.model_type == "generative":
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, torch_dtype=torch.float16,  quantization_config=quantization_config)
        model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch.float16).to(device)

    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.model_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)

        general.freeze_all_layers_but_classifier(model)

    all_true_labels = []
    all_predicted_labels = []

    all_predicted_top_5_labels = []

    if args.label_level == "top":
        category_column = "category"
    else:
        category_column = "sub_category"

    for i, chunk in tqdm(enumerate(ddf), total=total_rows // size_per_chunk + 1, desc="Processing Chunks", mininterval=2):

        df_partition = pd.DataFrame(chunk)

        if args.model_type == "generative":
            df_partition["predicted_label"] = generative_partition(args, df_partition, model, tokenizer, candidate_labels)
            #df_partition["predicted_label"] = df_partition["text"].map(lambda x: generative_partition(x, model, tokenizer, candidate_labels))
        else:
            df_partition = classify_partition(args, df_partition, model, tokenizer, label_encoder)

            all_predicted_top_5_labels.extend(df_partition["top_5_predicted"].tolist())

        all_true_labels.extend(df_partition[category_column].tolist())
        all_predicted_labels.extend(df_partition["predicted_label"].tolist())

    accuracy_score = accuracy_score(all_true_labels, all_predicted_labels)
    #logging.info(f"Accuracy score: {accuracy_score*100:.2f}%")

    if args.model_type != "generative":
        accuracy_top_5 = top_5_accuracy(all_true_labels, all_predicted_top_5_labels)
        #logging.info(f"Top 5 accuracy score: {accuracy_top_5*100:.2f}%")

    f1_score_result = f1_score(all_true_labels, all_predicted_labels, average='weighted', zero_division=0)
    #logging.info(f"F1-score: {f1_score_result*100:.2f}%")
    f1_score_macro = f1_score(all_true_labels, all_predicted_labels, average='macro', zero_division=0)
    #logging.info(f"F1-score-macro: {f1_score_macro*100:.2f}%")

    precision = precision_score(all_true_labels, all_predicted_labels, average='weighted', zero_division=0)
    recall = recall_score(all_true_labels, all_predicted_labels, average='weighted', zero_division=0)

    df = pd.DataFrame({'all_true_labels': all_true_labels, 'all_predicted_labels': all_predicted_labels})

    df.to_csv(f"{os.getenv(f"DATASETS_FOLDER_{str(args.dataset).upper()}_METRICS_FOLDER")}/{args.dataset}_{str(args.model_path).split('/')[-1]}_{args.model_type}_{args.label_level}_metrics_baseline_{total_rows}.csv", index=False, header=True)

    #logging.info("--- %s seconds ---" % (time.time() - start_time))

    elapsed = time.time() - start_time  # elapsed time in seconds

    hours, rem = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(rem, 60)

    if args.model_type != "generative":
        df_results = pd.DataFrame({'accuracy': f"{accuracy_score*100:.2f}%", "f1": f"{f1_score_result*100:.2f}%", "F1 macro": f"{f1_score_macro*100:.2f}%", "precision": f"{precision*100:.2f}%", "recall": f"{recall*100:.2f}%", 'Time': f"{hours}:{minutes:02}:{seconds:02}"}, index=[0])
        df_results.to_csv(
            f"{os.getenv(f"DATASETS_FOLDER_{str(args.dataset).upper()}_RESULTS_FOLDER")}/{args.dataset}_{str(args.model_path).split('/')[-1]}_{args.model_type}_{args.label_level}_results_baseline.csv",
            index=True, header=True)
    else:
        df_results = pd.DataFrame({'accuracy': f"{accuracy_score * 100:.2f}%", 'f1': f"{f1_score_result * 100:.2f}%", "F1 macro": f"{f1_score_macro * 100:.2f}%", "precision": f"{precision*100:.2f}%", "recall": f"{recall*100:.2f}%", 'Time': f"{hours}:{minutes:02}:{seconds:02}"}, index=[0])
        df_results.to_csv(
            f"{os.getenv(f"DATASETS_FOLDER_{str(args.dataset).upper()}_RESULTS_FOLDER")}/{args.dataset}_{str(args.model_path).split('/')[-1]}_{args.model_type}_{args.label_level}_results_baseline.csv",
            index=True, header=True)
