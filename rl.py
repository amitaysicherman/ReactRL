import torch
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerFast
)
from trl import PPOTrainer, PPOConfig
from trl.models import AutoModelForCausalLMWithValueHead
from rdkit import Chem
import re
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
import numpy as np
import json
from common import (
    MODEL_NAME,
    SFT_OUTPUT_DIR,
    RL_OUTPUT_DIR,
    MAX_SEQ_LENGTH,
)
from data import get_datasets, get_tokenizer,tokenize_rl_prompt
def get_prompt(example):
    tokenized_text = example["text"]
    parts = tokenized_text.split(" >> ")
    prompt_text = parts[0] + " >> "
    return {"prompt": prompt_text, "text": example["text"]}

def smiles_validity_reward(smiles_list):
    """
    Rewards a generation based on whether the generated SMILES string is chemically valid.
    1.0 for valid SMILES, 0.0 otherwise.
    """

    rewards = []
    for smiles_tokens in smiles_list:
        raw_smiles = smiles_tokens.replace(" ", "")
        mol = Chem.MolFromSmiles(raw_smiles)
        is_valid = mol is not None and len(raw_smiles.strip()) > 0
        rewards.append(torch.tensor(1.0 if is_valid else 0.0))
    return rewards


if __name__ == "__main__":
    train_dataset, val_dataset, train_subset_dataset = get_datasets()

    model = AutoModelForCausalLMWithValueHead.from_pretrained(SFT_OUTPUT_DIR)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(SFT_OUTPUT_DIR)

    rl_dataset = train_dataset.map(get_prompt, remove_columns=train_dataset.column_names)
    tokenizer=get_tokenizer(train_dataset)
    rl_dataset = rl_dataset.map(lambda x:tokenize_rl_prompt(tokenizer=tokenizer,example=x), batched=True, remove_columns=["prompt", "text"])
    rl_dataset.set_format("torch", columns=["input_ids", "attention_mask"])




ppo_config = PPOConfig(
    model_name=SFT_OUTPUT_DIR,
    learning_rate=1.41e-5,
    batch_size=4,
    mini_batch_size=4,
    gradient_accumulation_steps=1,
    ppo_epochs=100,
    init_kl_coef=0.1,
    max_grad_norm=1.0,
    seed=0,
    target_kl=0.05,
    vf_coef=0.1,
)

ppo_trainer = PPOTrainer(
    ppo_config,
    model,
    ref_model,
    tokenizer,
    rl_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

# Train the RL model
print("Starting RL (PPO) training...")
for epoch in range(2):
    for batch in ppo_trainer.dataloader:
        query_tensors = batch["input_ids"]

        prompt_texts = tokenizer.batch_decode(query_tensors, skip_special_tokens=True)
        prompts = [p.split(' >> ')[0] + ' >> ' for p in prompt_texts]
        prompt_tensors = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).input_ids
        prompt_tensors = [x for x in prompt_tensors]
        response_tensors = ppo_trainer.generate(
            prompt_tensors,
            max_new_tokens=100,
            pad_token_id=tokenizer.pad_token_id,  # Use PAD token
        )
        response_texts = [
            tokenizer.decode(r.squeeze()[-len(q):], skip_special_tokens=True).strip()
            for r, q in zip(response_tensors, prompt_tensors)
        ]

        generated_smiles_tokens = []
        for smiles_tokens in response_texts:
            # match = re.search(r'>>\s+(.*)', text)
            # smiles_tokens = match.group(1).strip() if match else ""
            # smiles_tokens = smiles_tokens.split(' ')[0].split('<')[0].split('\n')[0]
            generated_smiles_tokens.append(smiles_tokens)

        rewards = smiles_validity_reward(generated_smiles_tokens)
        stats = ppo_trainer.step(prompt_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

    ppo_trainer.save_pretrained(RL_OUTPUT_DIR)
    tokenizer.save_pretrained(RL_OUTPUT_DIR)
    print(f"RL fine-tuned model and tokenizer saved to {RL_OUTPUT_DIR}")

