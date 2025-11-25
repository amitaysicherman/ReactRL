# --- Configuration ---
import re
MODEL_NAME = "openai-community/gpt2"  # Using gpt2 base model for compatibility with CausalLM and TRL
SFT_OUTPUT_DIR = "./sft_reaction_model"
RL_OUTPUT_DIR = "./rl_reaction_model"
TOKENIZER_OUTPUT_DIR = "./smiles_tokenizer"
MAX_SEQ_LENGTH = 512
VOCAB_SIZE = 512  # Max vocabulary size for the WordLevel model

SMILES_TOKENIZER_PATTERN = r"(\%\([0-9]{3}\)|\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\||\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"
SMILES_REGEX = re.compile(SMILES_TOKENIZER_PATTERN)

train_dataset_file = "uspto50k/cleaned_train.jsonl"
val_dataset_file = "uspto50k/cleaned_val.jsonl"