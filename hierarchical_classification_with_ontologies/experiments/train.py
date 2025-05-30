import os
import time
import csv
import json
import joblib
import pandas as pd
import torch
from datasets import Dataset
from dotenv import load_dotenv
from huggingface_hub import login
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, \
    DataCollatorWithPadding, BartForSequenceClassification, BartTokenizerFast, BertForSequenceClassification

from hierarchical_classification_with_ontologies.functions import args_setup, general, manage_datasets

load_dotenv()
device = general.device()
login(os.getenv("HUGGING_FACE_TOKEN"))

torch.manual_seed(0)
torch.cuda.manual_seed(0)
start_time = time.time()
args = args_setup.get_arg_parser_train().parse_args()

input_dir = os.getenv(f"DATASETS_FOLDER_{str(args.dataset).upper()}")

if args.category is not None:
    train_df = manage_datasets.reading_category_dataset_from_file_without_total(input_dir=input_dir, dataset_name=args.dataset, dataset_type="train", category=args.category)
    val_df = manage_datasets.reading_category_dataset_from_file_without_total(input_dir=input_dir, dataset_name=args.dataset, dataset_type="val", category=args.category)

    category_column = "sub_category"

else:

    train_df = manage_datasets.reading_dataset_from_file_without_total(input_dir=input_dir, dataset_name=args.dataset, dataset_type="train")
    val_df = manage_datasets.reading_dataset_from_file_without_total(input_dir=input_dir, dataset_name=args.dataset, dataset_type="val")

    if args.label_level == "top":
        category_column = "category"
    else:
        category_column = "sub_category"

if args.dataset != "the_guardian":
    train_df["text"] = train_df["title"]
    val_df["text"] = val_df["title"]

label_encoder = LabelEncoder()
train_df["labels"] = label_encoder.fit_transform(train_df[category_column])
val_df["labels"] = label_encoder.transform(val_df[category_column])

num_labels = len(label_encoder.classes_)

tokenizer = AutoTokenizer.from_pretrained(args.model_path)

train_dataset = Dataset.from_pandas(train_df[["text", "labels"]])
val_dataset = Dataset.from_pandas(val_df[["text", "labels"]])

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True)

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=num_labels, ignore_mismatched_sizes=True).to(device)
general.freeze_all_layers_but_classifier(model)

all_results = []

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
        output_dir="./results",  # Save model checkpoints here
        eval_strategy="epoch",  # Evaluate at the end of each epoch
        save_strategy="epoch",  # Save checkpoints at intervals
        save_steps=10_000,  # Save every 10,000 steps (adjust based on dataset size)
        save_total_limit=2,  # Keep only last 2 checkpoints to save space
        logging_steps=100,  # Log metrics frequently for better monitoring
        per_device_train_batch_size=128,  # High batch size for faster training
        per_device_eval_batch_size=128,  # Keep evaluation batch size the same
        gradient_accumulation_steps=1,  # No need for accumulation with 4090
        learning_rate=2e-5,  # Standard for fine-tuning transformers
        weight_decay=0.01,  # Helps prevent overfitting
        num_train_epochs=2,  # Adjust based on validation loss trends
        warmup_ratio=0.05,  # 5% of steps for learning rate warmup
        fp16=True,  # Mixed precision for better speed & lower VRAM usage
        optim="adamw_bnb_8bit",  # 8-bit optimizer for faster training
        save_on_each_node=False,  # Prevent redundant saving in multi-GPU
        report_to="none",  # Disable logging to external platforms
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        seed=0
    )

def compute_metrics(pred):
    labels = pred.label_ids
    logits = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
    preds = logits.argmax(-1)

    decoded_preds = label_encoder.inverse_transform(preds)
    decoded_labels = label_encoder.inverse_transform(labels)

    # Add to results list
    for i in range(len(decoded_preds)):
        all_results.append({
            "pred_subcategory": decoded_preds[i],
            "true_subcategory": decoded_labels[i]
        })

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "precision": precision_score(labels, preds, average='weighted', zero_division=0),
        "recall": recall_score(labels, preds, average='weighted', zero_division=0)
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    data_collator=data_collator
)

trainer.train()


end_time = time.time()
training_time = end_time - start_time  # in seconds
all_results = []
final_metrics = trainer.evaluate()

hours, rem = divmod(int(training_time), 3600)
minutes, seconds = divmod(rem, 60)

df_results = pd.DataFrame({'Accuracy': f"{final_metrics["eval_accuracy"]*100:.2f}%", "F1": f"{final_metrics["eval_f1"]*100:.2f}%", "F1 macro": f"{final_metrics["eval_f1_macro"]*100:.2f}%","Precision": f"{final_metrics["eval_precision"]*100:.2f}%","Recall": f"{final_metrics["eval_recall"]*100:.2f}%", "Loss": f"{final_metrics["eval_loss"]:.2f}", 'Time': f"{hours}:{minutes:02}:{seconds:02}"}, index=[0])

if args.category != None:
    df_results.to_csv(
            f"{os.getenv(f"DATASETS_FOLDER_{str(args.dataset).upper()}_RESULTS_FOLDER")}/{args.dataset}_{str(args.model_path).split('/')[-1]}_{args.label_level}_results_{str(args.category).lower()}_train.csv",
            index=True, header=True)

    model.save_pretrained(
        f"{os.getenv(f"DATASETS_FOLDER_{str(args.dataset).upper()}_MODELS_FOLDER")}/{args.dataset}_{str(args.model_path).split('/')[-1]}_{args.label_level}_{str(args.category).lower()}_fine_tuned_model")
    tokenizer.save_pretrained(
        f"{os.getenv(f"DATASETS_FOLDER_{str(args.dataset).upper()}_MODELS_FOLDER")}/{args.dataset}_{str(args.model_path).split('/')[-1]}_{args.label_level}_{str(args.category).lower()}_fine_tuned_model")
    joblib.dump(label_encoder,
                f"{os.getenv(f"DATASETS_FOLDER_{str(args.dataset).upper()}_MODELS_FOLDER")}/{args.dataset}_{str(args.model_path).split('/')[-1]}_{args.label_level}_{str(args.category).lower()}_label_encoder.pkl")

    pd.DataFrame(all_results).to_csv(f"{os.getenv(f"DATASETS_FOLDER_{str(args.dataset).upper()}_RESULTS_FOLDER")}/{args.dataset}_{str(args.model_path).split('/')[-1]}_{args.label_level}_results_val_dataset.csv", mode='a', index=False, header=False)

else:
    df_results.to_csv(
        f"{os.getenv(f"DATASETS_FOLDER_{str(args.dataset).upper()}_RESULTS_FOLDER")}/{args.dataset}_{str(args.model_path).split('/')[-1]}_{args.label_level}_results_train.csv",
        index=True, header=True)

    model.save_pretrained(
        f"{os.getenv(f"DATASETS_FOLDER_{str(args.dataset).upper()}_MODELS_FOLDER")}/{args.dataset}_{str(args.model_path).split('/')[-1]}_{args.label_level}_fine_tuned_model")
    tokenizer.save_pretrained(
        f"{os.getenv(f"DATASETS_FOLDER_{str(args.dataset).upper()}_MODELS_FOLDER")}/{args.dataset}_{str(args.model_path).split('/')[-1]}_{args.label_level}_fine_tuned_model")

    joblib.dump(label_encoder, f"{os.getenv(f"DATASETS_FOLDER_{str(args.dataset).upper()}_MODELS_FOLDER")}/{args.dataset}_{str(args.model_path).split('/')[-1]}_{args.label_level}_label_encoder.pkl")

