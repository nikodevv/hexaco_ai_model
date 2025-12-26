import json
import torch
import psutil
import pandas as pd
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

# This script is based on finetune_custom.py, adapted for RunPod with unsloth.

# --- Data Transformation Logic from finetune_custom.py ---

# Step 7: Define and apply prompt formatting function
alpaca_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

HEXACO_TRAIT_MAPPING = {
    "H": "Honesty-Humility",
    "E": "Emotionality",
    "X": "Extraversion",
    "A": "Agreeableness",
    "C": "Conscientiousness",
    "O": "Openness_to_experience"
}
HEXACO_FACET_MAPPING = {
    # Honesty-Humility (H)
    "Sinc": "Sincerity", "Fair": "Fairness", "Gree": "Greed Avoidance", "Mode": "Modesty",
    # Emotionality (E)
    "Fear": "Fearfulness", "Anxi": "Anxiety", "Depe": "Dependence", "Sent": "Sentimentality",
    # Extraversion (X)
    "Self": "Social Self-Esteem", "SocB": "Social Boldness", "Soci": "Sociability", "Live": "Liveliness",
    # Agreeableness (A)
    "Forg": "Forgivingness", "Gent": "Gentleness", "Flex": "Flexibility", "Pati": "Patience",
    # Conscientiousness (C)
    "Orga": "Organization", "Dili": "Diligence", "Perf": "Perfectionism", "Prud": "Prudence",
    # Openness to Experience (O)
    "AesA": "Aesthetic Appreciation", "Inqu": "Inquisitiveness", "Crea": "Creativity", "Unco": "Unconventionality"
}

def formatting_prompts_func(examples):

    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input_text, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input_text, output) + tokenizer.eos_token
        texts.append(text)
    return {"text": texts}

def create_training_data_for_runpod():
    """
    Reads the codebook and cleaned data to generate a dataset for fine-tuning
    in Alpaca JSON format.
    """
    codebook = {}
    with open('data/codebook.txt', 'r') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                codebook[parts[0]] = parts[1]

    df = pd.read_csv('data/data_cleaned_partial.csv')

    # Corrected logic to get personality score columns
    hexaco_cols = [col for col in df.columns if col.startswith('HEXACO_')]
    facet_cols = [col for col in df.columns if col.startswith('FACET_')]
    personality_cols = hexaco_cols + facet_cols

    training_data = []
    for _, row in df.iterrows():
        instruction = "How would a person with the following answer the following questions (1 = strongly disagree, 5 = strongly agree)?\n"
        
        # Add personality traits to instruction
        for col in hexaco_cols:
            trait_key = col.split('_')[1]
            trait_name = HEXACO_TRAIT_MAPPING.get(trait_key, trait_key)
            instruction += f"{trait_name}: {row[col]}\n"

        # Add personality facets to instruction
        for col in facet_cols:
            facet_key = col.split('_')[1]
            facet_name = HEXACO_FACET_MAPPING.get(facet_key, facet_key)
            instruction += f"{facet_name}: {row[col]}\n"

        # Create a training example for each question-answer pair in the row
        for col_name, value in row.items():
            if col_name in codebook:
                training_data.append({
                    "instruction": instruction.strip(),
                    "input": codebook[col_name],
                    "output": str(value),
                })

    output_path = 'data/training_dataset.json'
    with open(output_path, 'w') as f:
        json.dump(training_data, f, indent=2)
        
    print(f"Generated dataset with {len(training_data)} examples at {output_path}")
    return output_path

# --- Fine-Tuning Script for RunPod ---

# Step 1: Create and load the dataset
DATASET_PATH = create_training_data_for_runpod()

# Step 2: Load the full dataset
raw_dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

# Step 3: Split into training and evaluation sets (80% train, 20% eval)
# Using shuffle=False to ensure the last 20% is used for the evaluation set
train_test_split_dataset = raw_dataset.train_test_split(test_size=0.2, shuffle=False)

# Step 4: Load Model and Tokenizer using Unsloth
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/mistral-7b-v0.3-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Step 5: Format prompts for both splits
# The formatting_prompts_func now takes an `examples` dictionary and returns a dictionary with a "text" field.
# So we apply it to both the 'train' and 'test' (evaluation) splits.

train_dataset = train_test_split_dataset["train"].map(formatting_prompts_func, batched=True)
eval_dataset = train_test_split_dataset["test"].map(formatting_prompts_func, batched=True)

# Step 6: Add LoRA Adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
)

# Step 7: Configure and run the training
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset, # Added evaluation dataset
    dataset_text_field="text",
    max_seq_length=2048,
    dataset_num_proc=19,  ## Training on 15vpcu instance, recommendation is to do + 4
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,  # Adjust based on your dataset size
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        output_dir="outputs",
        optim="adamw_8bit",
        seed=42,
        eval_strategy="steps", # Enable evaluation
        eval_steps=10, # Evaluate every 10 steps
    ),
)

print("Starting training...")
trainer.train()
print("Training complete!")

# To save the final model, you can use:
# model.save_pretrained("lora_model")
# tokenizer.save_pretrained("lora_model")
