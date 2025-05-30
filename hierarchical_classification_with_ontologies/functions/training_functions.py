from sklearn.metrics import f1_score, accuracy_score
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding


def compute_metrics(pred):
    labels = pred.label_ids
    logits = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
    preds = logits.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
        "f1_macro": f1_score(labels, preds, average="macro")
    }

def train_without_eval(model, tokenizer, train_dataset):
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
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="weighted"),
            "f1_macro": f1_score(labels, preds, average="macro")
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator
    )

    trainer.train()
