import torch
from transformers import DataCollatorWithPadding
from trl import PPOTrainer, PPOConfig
# Use the Seq2Seq version of the ValueHead model
from trl.models import AutoModelForSeq2SeqLMWithValueHead
from rdkit import Chem
from common import (
    SFT_OUTPUT_DIR,
    RL_OUTPUT_DIR,
)
from data import get_datasets, get_tokenizer, tokenize_rl_prompt


def smiles_validity_reward(smiles_list):
    rewards = []
    for smiles in smiles_list:
        # Remove special tokens and whitespace
        clean_smiles = smiles.replace(" ", "").replace("[EOS]", "").replace("[PAD]", "")
        mol = Chem.MolFromSmiles(clean_smiles)
        is_valid = mol is not None and len(clean_smiles.strip()) > 0
        rewards.append(torch.tensor(1.0 if is_valid else 0.0))
    return rewards


if __name__ == "__main__":
    train_dataset, val_dataset, train_subset_dataset = get_datasets()

    # Load Seq2Seq models with Value Head
    model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(SFT_OUTPUT_DIR)
    ref_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(SFT_OUTPUT_DIR)

    tokenizer = get_tokenizer(train_dataset)

    # Prepare dataset: We only need the input_text (Reactants) for the prompt
    rl_dataset = train_dataset.map(
        lambda x: tokenize_rl_prompt(tokenizer, x),
        batched=True,
        remove_columns=train_dataset.column_names
    )
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

    # Use standard collator with padding since we just have encoder inputs
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    ppo_trainer = PPOTrainer(
        ppo_config,
        model,
        ref_model,
        tokenizer,
        rl_dataset,
        data_collator=data_collator,
    )

    print("Starting RL (PPO) training...")
    for epoch in range(2):
        for batch in ppo_trainer.dataloader:
            query_tensors = batch["input_ids"]

            # Generate Responses (Decoder Output)
            response_tensors = ppo_trainer.generate(
                query_tensors,
                max_new_tokens=100,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=True  # Usually we want sampling for RL exploration
            )

            # Decode responses for reward calculation
            batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

            # Calculate Rewards
            rewards = smiles_validity_reward(batch["response"])

            # PPO Step
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(stats, batch, rewards)

        ppo_trainer.save_pretrained(RL_OUTPUT_DIR)
        tokenizer.save_pretrained(RL_OUTPUT_DIR)
        print(f"RL fine-tuned model and tokenizer saved to {RL_OUTPUT_DIR}")