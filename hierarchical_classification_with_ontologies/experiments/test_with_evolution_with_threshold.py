import os
import os
import time

import joblib
import pandas as pd
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn.functional as F

from hierarchical_classification_with_ontologies.functions import general, args_setup, manage_datasets

load_dotenv()
device = general.device()
login(os.getenv("HUGGING_FACE_TOKEN"))

def classify_chunk(texts, main_model, main_tokenizer, main_encoder, category_map, single_subcategory, sub_models, threshold):
    # Tokenize all texts in the chunk
    inputs = main_tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(device)
    results = []
    # Classify main categories
    with torch.no_grad():
        logits = main_model(**inputs).logits
        probs = F.softmax(logits, dim=1)
        confidences, main_pred_indices = torch.max(probs, dim=1)

    for i, text in enumerate(texts):
        main_idx = main_pred_indices[i].item()
        main_category = category_map[main_idx]
        confidence = confidences[i].item()

        result = {
            'text': text,
            'main_category': main_category,
            'main_confidence': confidence,
        }

        if main_category in single_subcategory:
            # Category has only one subcategory
            result['sub_category_label'] = single_subcategory[main_category]
            result['sub_category_index'] = 0

        elif confidence >= threshold:
            # Confident enough → use sub-model
            input_i = {k: v[i].unsqueeze(0) for k, v in inputs.items()}
            sub_model = sub_models[main_category]
            encoder = label_encoders[main_category]

            with torch.no_grad():
                sub_logits = sub_model(**input_i).logits
                sub_idx = torch.argmax(sub_logits, dim=1).item()
                sub_label = encoder.inverse_transform([sub_idx])[0]

            result['sub_category_label'] = sub_label
            result['sub_category_index'] = sub_idx

        else:
            # Not confident → mark subcategory as unknown/empty
            result['sub_category_label'] = "uncertain"
            result['sub_category_index'] = -1

        results.append(result)

    return results

def process_dataset(args, ddf, total_rows, main_model, main_tokenizer, main_label_encoder, single_subcategory, category_map, sub_models, threshold):
    all_results = []

    for i, chunk in tqdm(enumerate(ddf), total=total_rows // size_per_chunk + 1, desc="Processing Chunks", mininterval=2):
        if args.dataset == "the_guardian":
            texts = chunk["text"].astype(str).tolist()
        else:
            texts = chunk["title"].astype(str).tolist()

        true_main = chunk['category'].tolist()
        true_sub = chunk['sub_category'].tolist()
        chunk_results = classify_chunk(texts, main_model, main_tokenizer, main_label_encoder, category_map, single_subcategory, sub_models, threshold)

        for i in range(len(chunk_results)):
            chunk_results[i]['true_main_category'] = true_main[i]
            chunk_results[i]['true_sub_category_label'] = true_sub[i]

        all_results.extend(chunk_results)


    return pd.DataFrame(all_results)



def process_json(data, flag):
    if flag == "top":
        return list(data.keys())  # Return only the keys
    else:
        return [item for sublist in data.values() for item in sublist]  # Flatten values



if __name__ == "__main__":
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    start_time = time.time()
    args = args_setup.get_arg_parser_with_evolution_with_threshold().parse_args()

    if args.dataset == "the_guardian":
        size_per_chunk = 250
    else:
        size_per_chunk = 150

    input_dir = os.getenv(f"DATASETS_FOLDER_{str(args.dataset).upper()}")
    ddf, total_rows = manage_datasets.reading_dataset_from_file_in_chunks(input_dir=input_dir,
                                                                          dataset_name=args.dataset, experience="test",
                                                              chunk_size=size_per_chunk)

    # Categories that only have one subcategory

    # Load main label encoder
    print("Loading main label encoder...")
    main_label_encoder = joblib.load(f"{os.getenv(f"DATASETS_FOLDER_{str(args.dataset).upper()}_MODELS_FOLDER")}/{args.dataset}_{str(args.model_path).split('/')[-1]}_top_label_encoder.pkl")

    classifier = AutoModelForSequenceClassification.from_pretrained(
        f"{os.getenv(f"DATASETS_FOLDER_{str(args.dataset).upper()}_MODELS_FOLDER")}/{args.dataset}_{str(args.model_path).split('/')[-1]}_top_fine_tuned_model").to(
        device)
    classifier_tokenizer = AutoTokenizer.from_pretrained(
        f"{os.getenv(f"DATASETS_FOLDER_{str(args.dataset).upper()}_MODELS_FOLDER")}/{args.dataset}_{str(args.model_path).split('/')[-1]}_top_fine_tuned_model")

    general.freeze_all_layers_but_classifier(classifier)

    if args.dataset == "the_guardian":
        single_subcategory = {
            'childrens-books-site': 'childrens-books-site/others',
            'cities': 'cities/others',
            'crosswords': 'crosswords/others',
            'games': 'games/others',
            'global': 'global/others',
            'law': 'law/others',
            'media-network': 'media-network/others',
            'small-business-network': 'small-business-network/others',
            'sustainable-business': 'sustainable-business/others',
            'teacher-network': 'teacher-network/others'
        }
    else:
        single_subcategory = {}

    # Map from index to readable category name
    category_map = {i: label for i, label in enumerate(main_label_encoder.classes_)}

    # Load sub-models (skip ones in SINGLE_SUBCATEGORY)
    print("Loading sub-models and encoders...")
    sub_models = {}
    label_encoders = {}
    for category in tqdm(category_map.values(), total=len(category_map.values()), desc="Processing categories"):
        if category in single_subcategory:
            continue  # no sub-model for these
        model_path = os.path.join(f"{os.getenv(f"DATASETS_FOLDER_{str(args.dataset).upper()}_MODELS_FOLDER")}/{args.dataset}_{str(args.model_path).split('/')[-1]}_sub_{str(category).lower()}_fine_tuned_model")
        encoder_path = os.path.join(f"{os.getenv(f"DATASETS_FOLDER_{str(args.dataset).upper()}_MODELS_FOLDER")}/{args.dataset}_{str(args.model_path).split('/')[-1]}_sub_{str(category).lower()}_label_encoder.pkl")
        if os.path.exists(model_path):
            sub_models[category] = AutoModelForSequenceClassification.from_pretrained(model_path).to(device).eval()
        if os.path.exists(encoder_path):
            label_encoders[category] = joblib.load(encoder_path)

    all_results = process_dataset(args, ddf, total_rows, classifier, classifier_tokenizer, main_label_encoder, single_subcategory, category_map, sub_models, args.threshold)

    main_accuracy = accuracy_score(all_results['true_main_category'], all_results['main_category'])
    f1_score_result = f1_score(all_results['true_main_category'], all_results['main_category'], average='weighted',
                               zero_division=0)
    f1_score_macro = f1_score(all_results['true_main_category'], all_results['main_category'], average='macro',
                              zero_division=0)
    precision_macro = precision_score(all_results['true_main_category'], all_results['main_category'],
                                      average='weighted', zero_division=0)
    recall_macro = recall_score(all_results['true_main_category'], all_results['main_category'], average='weighted',
                                zero_division=0)

    sub_accuracy = accuracy_score(all_results['true_sub_category_label'], all_results['sub_category_label'])
    sub_f1_score_result = f1_score(all_results['true_sub_category_label'], all_results['sub_category_label'], average='weighted',
                               zero_division=0)
    sub_f1_score_macro = f1_score(all_results['true_sub_category_label'], all_results['sub_category_label'], average='macro',
                              zero_division=0)

    precision_sub = precision_score(all_results['true_sub_category_label'], all_results['sub_category_label'],
                                    average='weighted', zero_division=0)
    recall_sub = recall_score(all_results['true_sub_category_label'], all_results['sub_category_label'],
                              average='weighted',
                              zero_division=0)

    elapsed = time.time() - start_time  # elapsed time in seconds

    hours, rem = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(rem, 60)

    df_results = pd.DataFrame({'Main Accuracy': f"{main_accuracy * 100:.2f}%",
                               "Main F1": f"{f1_score_result * 100:.2f}%", "Main F1 macro": f"{f1_score_macro * 100:.2f}%",
                               "Main precision": f"{precision_macro * 100:.2f}%",
                               "Main Recall": f"{recall_macro * 100:.2f}%",
                               'sub accuracy': f"{sub_accuracy * 100:.2f}%",
                               "Sub f1": f"{sub_f1_score_result * 100:.2f}%", "Sub F1 macro": f"{sub_f1_score_macro * 100:.2f}%",
                               "Sub precision": f"{precision_sub * 100:.2f}%",
                               "Sub Recall": f"{recall_sub * 100:.2f}%",
                               'Time': f"{hours}:{minutes:02}:{seconds:02}"}, index=[0])

    output_path = f"{os.getenv(f"DATASETS_FOLDER_{str(args.dataset).upper()}_RESULTS_FOLDER")}/{args.dataset}_{str(args.model_path).split('/')[-1]}_results_test_with_evolution_with_threshold_{str(args.threshold)}.csv"
    df_results.to_csv(output_path, index=True, header=True)
