import torch
from transformers import (
    AutoModelForSeq2SeqLM,  # Changed from CausalLM
    Seq2SeqTrainingArguments,  # Changed from TrainingArguments
    Seq2SeqTrainer,  # Changed from Trainer
    DataCollatorForSeq2Seq,  # Standard Seq2Seq Collator
)
import numpy as np
from common import MODEL_NAME, SFT_OUTPUT_DIR
from data import get_tokenizer, get_datasets, tokenize_dataset


def get_compute_metrics_fn(tokenizer):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        # In case the model returns more than just logits (like tuple), handle it
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        # Decode generated predictions
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Print a few examples
        print(f"\n{'=' * 20} EXAMPLES {'=' * 20}")
        for i in range(min(3, len(decoded_preds))):
            print(f"Pred: {decoded_preds[i]}")
            print(f"Gold: {decoded_labels[i]}")
            print("-" * 20)

        # Simple Exact Match Accuracy
        matches = [pred.strip() == label.strip() for pred, label in zip(decoded_preds, decoded_labels)]
        perfect_match_acc = sum(matches) / len(matches)

        return {
            "perfect_match_accuracy": perfect_match_acc
        }

    return compute_metrics


if __name__ == "__main__":
    train_dataset, val_dataset, train_subset_dataset = get_datasets()
    tokenizer = get_tokenizer(train_dataset)

    # Load T5 (or similar Encoder-Decoder)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded with {num_params} parameters.")

    sft_training_args = Seq2SeqTrainingArguments(
        output_dir=SFT_OUTPUT_DIR,
        num_train_epochs=50,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        evaluation_strategy="steps",
        eval_steps=250,
        logging_steps=250,
        warmup_steps=100,
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
        report_to="none",
        learning_rate=5e-5,
        lr_scheduler_type='constant',
        predict_with_generate=True,  # Critical for Seq2Seq metrics
        generation_max_length=100,
    )

    # Use standard Seq2Seq collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100
    )

    tokenized_train_dataset = tokenize_dataset(tokenizer, train_dataset)
    tokenized_eval_dataset = tokenize_dataset(tokenizer, val_dataset)
    tokenized_train_subset_dataset = tokenize_dataset(tokenizer, train_subset_dataset)

    compute_metrics_fn = get_compute_metrics_fn(tokenizer)

    sft_trainer = Seq2SeqTrainer(
        model=model,
        args=sft_training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset={"validation": tokenized_eval_dataset, "train_subset": tokenized_train_subset_dataset},
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
    )

    sft_trainer.train()

    sft_trainer.save_model(SFT_OUTPUT_DIR)
    tokenizer.save_pretrained(SFT_OUTPUT_DIR)