import json
import os
import time

import joblib
import pandas as pd
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from hierarchical_classification_with_ontologies.functions import args_setup, general, manage_datasets

load_dotenv()
device = general.device()
login(os.getenv("HUGGING_FACE_TOKEN"))

def top_5_accuracy(true_labels, top_5_labels):
    correct = 0
    for i in range(len(true_labels)):
        if true_labels[i] in top_5_labels[i]:
            correct += 1
    return correct / len(true_labels)

def classify_partition(args, df_partition, model, tokenizer, label_encoder):
    """Classifies a partition of data and tracks confidence"""
    if args.dataset == "the_guardian":
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


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    start_time = time.time()
    args = args_setup.get_arg_parser_test().parse_args()

    input_dir = os.getenv(f"DATASETS_FOLDER_{str(args.dataset).upper()}")
    size_per_chunk = 250

    total_rows = sum(1 for _ in open(f"{input_dir}/{args.dataset}_test.csv")) - 1
    ddf = pd.read_csv(f"{input_dir}/{args.dataset}_test.csv", sep=";", chunksize=size_per_chunk, usecols=["title", "text", "category", "sub_category"])
    #ddf = pd.read_csv(f"{input_dir}/{args.dataset}_test.csv", sep=";", usecols=["title", "text", "category", "sub_category"], nrows=total_rows)
    #print(len(ddf))
    #ddf = [ddf[i:i + size_per_chunk] for i in range(0, len(ddf), size_per_chunk)]

    with open(f"{input_dir}/initial_ontology.json", "r") as file:  # Replace 'data.json' with your actual file path
        data = json.load(file)
    candidate_labels = manage_datasets.process_json(data, args.label_level)

    label_encoder = joblib.load(f"{os.getenv(f"DATASETS_FOLDER_{str(args.dataset).upper()}_MODELS_FOLDER")}/label_encoder.pkl")
    label_encoder.fit(candidate_labels)

    #quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    #generative_tokenizer = AutoTokenizer.from_pretrained(f"{os.getenv(f"DATASETS_FOLDER_{str(args.dataset).upper()}_MODELS_FOLDER")}/{args.dataset}_{str(args.model_path).split('/')[-1]}_{args.label_level}_fine_tuned_model", torch_dtype=torch.float16,
    #                                          quantization_config=quantization_config)
    #generative = AutoModelForCausalLM.from_pretrained(f"{os.getenv(f"DATASETS_FOLDER_{str(args.dataset).upper()}_MODELS_FOLDER")}/{args.dataset}_{str(args.model_path).split('/')[-1]}_{args.label_level}_fine_tuned_model", trust_remote_code=True, torch_dtype=torch.float16).to(    device)

    classifier = AutoModelForSequenceClassification.from_pretrained(f"{os.getenv(f"DATASETS_FOLDER_{str(args.dataset).upper()}_MODELS_FOLDER")}/{args.dataset}_{str(args.classifier_path).split('/')[-1]}_{args.label_level}_fine_tuned_model").to(device)
    classifier_tokenizer = AutoTokenizer.from_pretrained(f"{os.getenv(f"DATASETS_FOLDER_{str(args.dataset).upper()}_MODELS_FOLDER")}/{args.dataset}_{str(args.classifier_path).split('/')[-1]}_{args.label_level}_fine_tuned_model")

    general.freeze_all_layers_but_classifier(classifier)

    all_true_labels = []
    all_predicted_labels = []

    all_predicted_top_5_labels = []

    if args.label_level == "top":
        category_column = "category"
    else:
        category_column = "sub_category"

    for i, chunk in tqdm(enumerate(ddf), total=total_rows // size_per_chunk + 1, desc="Processing Chunks",
                         mininterval=2):

        df_partition = pd.DataFrame(chunk)

        df_partition = classify_partition(args, df_partition, classifier, classifier_tokenizer, label_encoder)

        all_predicted_top_5_labels.extend(df_partition["top_5_predicted"].tolist())

        all_true_labels.extend(df_partition[category_column].tolist())
        all_predicted_labels.extend(df_partition["predicted_label"].tolist())

    accuracy_score = accuracy_score(all_true_labels, all_predicted_labels)
    # logging.info(f"Accuracy score: {accuracy_score*100:.2f}%")

    accuracy_top_5 = top_5_accuracy(all_true_labels, all_predicted_top_5_labels)
    # logging.info(f"Top 5 accuracy score: {accuracy_top_5*100:.2f}%")

    f1_score_result = f1_score(all_true_labels, all_predicted_labels, average='weighted', zero_division=0)
    # logging.info(f"F1-score: {f1_score_result*100:.2f}%")
    f1_score_macro = f1_score(all_true_labels, all_predicted_labels, average='macro', zero_division=0)
    # logging.info(f"F1-score-macro: {f1_score_macro*100:.2f}%")

    precision = precision_score(all_true_labels, all_predicted_labels, average='weighted', zero_division=0)
    recall = recall_score(all_true_labels, all_predicted_labels, average='weighted', zero_division=0)

    df = pd.DataFrame({'all_true_labels': all_true_labels, 'all_predicted_labels': all_predicted_labels})

    df.to_csv(
        f"{os.getenv(f"DATASETS_FOLDER_{str(args.dataset).upper()}_METRICS_FOLDER")}/{args.dataset}_{str(args.classifier_path).split('/')[-1]}_{args.label_level}_metrics_test.csv",
        index=False, header=True)

    # logging.info("--- %s seconds ---" % (time.time() - start_time))

    elapsed = time.time() - start_time  # elapsed time in seconds

    hours, rem = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(rem, 60)

    df_results = pd.DataFrame(
        {'accuracy': f"{accuracy_score * 100:.2f}%",
         "f1": f"{f1_score_result * 100:.2f}%", "F1 macro": f"{f1_score_macro * 100:.2f}%",
         "precision": f"{precision * 100:.2f}%", "recall": f"{recall * 100:.2f}%",
         'Time': f"{hours}:{minutes:02}:{seconds:02}"}, index=[0])
    df_results.to_csv(
        f"{os.getenv(f"DATASETS_FOLDER_{str(args.dataset).upper()}_RESULTS_FOLDER")}/{args.dataset}_{str(args.classifier_path).split('/')[-1]}_{args.label_level}_results_test.csv",
        index=True, header=True)







