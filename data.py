from datasets import Dataset
from transformers import PreTrainedTokenizerFast
import os
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
import json

from common import (
    SMILES_REGEX,
    train_dataset_file,
    val_dataset_file,
    TOKENIZER_OUTPUT_DIR,
    MAX_SEQ_LENGTH,
    VOCAB_SIZE
)


############## DATASET ##############
def regex_smiles_pretokenize(rxn_smiles: str) -> str:
    tokens = [token for token in SMILES_REGEX.findall(rxn_smiles)]
    tokens = [t for t in tokens if t]
    return " ".join(tokens)


def get_datasets():
    # Load dataset from the provided JSONL file
    with open(train_dataset_file, 'r') as f:
        data = [json.loads(line) for line in f]
    train_dataset = Dataset.from_list(data)

    with open(val_dataset_file, 'r') as f:
        data = [json.loads(line) for line in f]
    val_dataset = Dataset.from_list(data)

    train_dataset = train_dataset.map(format_reaction, remove_columns=train_dataset.column_names)
    val_dataset = val_dataset.map(format_reaction, remove_columns=val_dataset.column_names).select(
        range(500))  # Limit val set for faster eval during demo
    train_subset_size = val_dataset.num_rows
    train_subset_dataset = train_dataset.select(range(train_subset_size))
    return train_dataset, val_dataset, train_subset_dataset


def format_reaction(example):
    example["text"] = regex_smiles_pretokenize(example['reaction_smiles'])
    return example


def get_training_corpus(dataset):
    for i in range(0, len(dataset), 1000):
        yield dataset[i: i + 1000]["text"]


def get_tokenizer(dataset):
    if not os.path.exists(f"{TOKENIZER_OUTPUT_DIR}/smiles_tokenizer.json"):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "[EOS]"]
        trainer = WordLevelTrainer(
            vocab_size=VOCAB_SIZE,
            special_tokens=special_tokens
        )
        tokenizer.train_from_iterator(get_training_corpus(dataset), trainer=trainer)
        tokenizer.save(f"{TOKENIZER_OUTPUT_DIR}/smiles_tokenizer.json")
        print(f"Custom tokenizer trained and saved to {TOKENIZER_OUTPUT_DIR}")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=f"{TOKENIZER_OUTPUT_DIR}/smiles_tokenizer.json",
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        model_max_length=MAX_SEQ_LENGTH,
        eos_token="[EOS]",
        padding_side="left",
    )
    return tokenizer


def tokenize_function(tokenizer, examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding="max_length"
    )


def tokenize_dataset(tokenizer, dataset):
    return dataset.map(lambda x: tokenize_function(tokenizer, x), batched=True, remove_columns=["text"])


def tokenize_rl_prompt(tokenizer,example):
    # The prompt is the space-separated string of reactant tokens + " >> "
    return tokenizer(
        example["prompt"],
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding="max_length"
    )
