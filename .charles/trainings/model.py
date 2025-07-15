# /qwen_finetuning_project/model.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

from logger_utils import setup_logger
from logger_utils import format_number

def get_model_and_tokenizer(model_id: str, for_sequence_classification: bool = False):
    """
    Loads the specified model and tokenizer.

    Args:
        model_id (str): The Hugging Face model identifier or path to a local model.
        for_sequence_classification (bool): If True, loads the model with a
                                            sequence classification head for reward modeling.
    
    Returns:
        A tuple containing the loaded model and tokenizer.
    """
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Determine the torch dtype for memory efficiency
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    if for_sequence_classification:
        print("Loading model for Sequence Classification (Reward Modeling)...")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            num_labels=1, # We want a single scalar output for the reward score
            torch_dtype=torch_dtype,
            trust_remote_code=True
        )
    else:
        print("Loading model for Causal LM (SFT/DPO)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            trust_remote_code=True
        )
    
    return model, tokenizer

def main():
    """
    Main function to load the model and tokenizer.
    """
    logger = setup_logger("model_loading")
    model_id = "Qwen/Qwen3-0.6B-Base"
    for_sequence_classification = False  # Change to True if loading for reward modeling

    model, tokenizer = get_model_and_tokenizer(model_id, for_sequence_classification)
    logger.success("Model and tokenizer loaded successfully.\n")
    logger.info("Model architecture:\n\n%s\n", model)
    logger.warning("Model parameter size: %s", format_number(sum(p.numel() for p in model.parameters())))
    logger.info("Tokenizer vocabulary size: %s", format_number(len(tokenizer)))

if __name__ == "__main__":
    main()