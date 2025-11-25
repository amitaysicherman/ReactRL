import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from rdkit import Chem
from common import RL_OUTPUT_DIR
from data import regex_smiles_pretokenize
########## INFERENCE ##########


final_model = AutoModelForCausalLM.from_pretrained(RL_OUTPUT_DIR)
final_tokenizer = PreTrainedTokenizerFast.from_pretrained(RL_OUTPUT_DIR)
final_model.eval()

reactant_smiles_raw = "CC(=O)Oc1ccccc1C(=O)O"
input_prompt_raw = f"{reactant_smiles_raw}>>"
input_prompt_tokenized = regex_smiles_pretokenize(input_prompt_raw)
input_ids = final_tokenizer.encode(input_prompt_tokenized, return_tensors="pt")
output_ids = final_model.generate(
    input_ids,
    max_new_tokens=100,
    do_sample=True,
    top_k=50,
    temperature=0.7,
    pad_token_id=final_tokenizer.pad_token_id,
)

generated_text_tokenized = final_tokenizer.decode(output_ids[0], skip_special_tokens=True)
product_tokens = generated_text_tokenized.split(" >> ")[-1].strip()
product_smiles_raw = product_tokens.replace(" ", "")

print(f"Input Reaction (Raw): {reactant_smiles_raw}")
print(f"Input Prompt (Tokenized): {input_prompt_tokenized}")
print(f"Generated Product SMILES (Raw): {product_smiles_raw}")
print(f"SMILES Validity Check: {Chem.MolFromSmiles(product_smiles_raw) is not None}")
