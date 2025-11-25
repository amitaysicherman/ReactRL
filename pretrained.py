import torch
from transformers import (
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
import numpy as np
from common import MODEL_NAME, SFT_OUTPUT_DIR
from data import get_tokenizer, get_datasets, tokenize_dataset


class DataCollatorForReaction(DataCollatorForLanguageModeling):
    """
    Custom collator that masks the loss for the "prompt" part (reactants + separator).
    It sets labels to -100 for all tokens up to and including '>>'.
    """

    def torch_call(self, examples):
        # Let the parent class handle basic batching and tensor conversion
        batch = super().torch_call(examples)
        sep_token_id = self.tokenizer.convert_tokens_to_ids(">>")
        for i in range(len(batch["labels"])):
            sep_indices = (batch["labels"][i] == sep_token_id).nonzero(as_tuple=True)[0]
            if len(sep_indices) > 0:
                sep_idx = sep_indices[0]
                batch["labels"][i, :sep_idx + 1] = -100
        return batch


def get_compute_metrics_fn(tokenizer):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        # --- PRINTING LOGIC ---
        # Only print the first 10 to keep logs clean
        n_print = min(10, len(labels))
        print(f"\n{'=' * 20} PREDICTIONS (Loss-contributing tokens only) {'=' * 20}")

        for i in range(n_print):
            # Create a boolean mask for this specific row where label is NOT -100
            # This identifies exactly which tokens contributed to the loss
            row_mask = labels[i] != -100

            # Apply mask to get only the 'valid' IDs for both Gold and Pred
            valid_label_ids = labels[i][row_mask]
            valid_pred_ids = predictions[i][row_mask]

            # Decode only these parts
            gold_str = tokenizer.decode(valid_label_ids, skip_special_tokens=True)
            pred_str = tokenizer.decode(valid_pred_ids, skip_special_tokens=True)

            print(f"Ex {i}:")
            print(f"  Gold: {gold_str}")
            print(f"  Pred: {pred_str}")

        print(f"{'=' * 60}\n")
        # ----------------------

        # --- METRICS LOGIC ---
        # Flatten masks for accuracy calculation
        total_mask = labels != -100
        filtered_preds = predictions[total_mask]
        filtered_labels = labels[total_mask]
        token_accuracy = (filtered_preds == filtered_labels).mean()

        # Row-level accuracy (Perfect Match)
        correct_predictions = (predictions == labels)
        row_is_correct = (correct_predictions | ~total_mask).all(axis=1)
        perfect_match_acc = row_is_correct.mean()

        return {
            "token_accuracy": token_accuracy,
            "perfect_match_accuracy": perfect_match_acc
        }

    return compute_metrics


if __name__ == "__main__":
    train_dataset, val_dataset, train_subset_dataset = get_datasets()
    tokenizer = get_tokenizer(train_dataset)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded with {num_params} parameters.")

    sft_training_args = TrainingArguments(
        output_dir=SFT_OUTPUT_DIR,
        num_train_epochs=50,
        per_device_train_batch_size=64,
        # --- ADDED FOR METRICS ---
        evaluation_strategy="steps",  # Calculate metrics every X steps
        eval_steps=250,  # Align with logging steps
        logging_steps=250,
        # -------------------------
        warmup_steps=100,
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
        report_to="none",
        learning_rate=5e-5,
        lr_scheduler_type='constant',

    )
    reaction_collator = DataCollatorForReaction(
        tokenizer=tokenizer,
        mlm=False  # We are doing Causal LM, not Masked LM
    )
    tokenized_train_dataset = tokenize_dataset(tokenizer, train_dataset)
    tokenized_eval_dataset = tokenize_dataset(tokenizer, val_dataset)
    tokenized_train_sunset_dataset = tokenize_dataset(tokenizer, train_subset_dataset)
    compute_metrics_fn = get_compute_metrics_fn(tokenizer)
    sft_trainer = Trainer(
        model=model,
        args=sft_training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset={"validation": tokenized_eval_dataset, "train_subset": tokenized_train_sunset_dataset},
        data_collator=reaction_collator,
        compute_metrics=compute_metrics_fn,
    )

    sft_trainer.train()

    sft_trainer.save_model(SFT_OUTPUT_DIR)
    tokenizer.save_pretrained(SFT_OUTPUT_DIR)
