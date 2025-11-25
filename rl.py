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
        # Ensure we provide an eos token id to generation so model can stop early
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if eos_id is None:
            # fall back to sep or pad if EOS is not set
            eos_id = getattr(tokenizer, "sep_token_id", None) or tokenizer.pad_token_id

        response_tensors = ppo_trainer.generate(
            prompt_tensors,
            max_new_tokens=100,
            pad_token_id=tokenizer.pad_token_id,  # Use PAD token
            eos_token_id=eos_id,
            do_sample=False,
            early_stopping=True,
        )

        # Decode only up to the EOS token for each generated sequence to avoid identical padded tails
        generated_smiles_tokens = []
        for r, q in zip(response_tensors, prompt_tensors):
            full = r.squeeze()
            gen = full[len(q):]
            # Convert to python list of ids
            gen_ids = gen.tolist() if isinstance(gen, torch.Tensor) else list(gen)
            if eos_id in gen_ids:
                idx = gen_ids.index(eos_id)
                gen_ids = gen_ids[:idx]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            generated_smiles_tokens.append(text)

        rewards = smiles_validity_reward(generated_smiles_tokens)
        stats = ppo_trainer.step(prompt_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

    ppo_trainer.save_pretrained(RL_OUTPUT_DIR)
    tokenizer.save_pretrained(RL_OUTPUT_DIR)
    print(f"RL fine-tuned model and tokenizer saved to {RL_OUTPUT_DIR}")
