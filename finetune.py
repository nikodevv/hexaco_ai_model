import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset

# --- Configuration ---
MODEL_NAME = "mistralai/Mistral-7B-v0.1" # Or "mistralai/Mistral-7B-Instruct-v0.1"
DATASET_NAME = "yahma/alpaca-cleaned" # Example dataset, replace with your own
OUTPUT_DIR = "./mistral_7b_finetuned"
NEW_MODEL_NAME = "mistral-7b-finetuned-custom"

# LoRA configuration
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training arguments
TRAINING_ARGS = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=1, # Adjust based on VRAM, 1 is common for 7B with QLoRA
    gradient_accumulation_steps=4, # Accumulate gradients to simulate larger batch size
    optim="paged_adamw_8bit", # Use 8-bit AdamW for memory efficiency
    save_steps=500,
    logging_steps=50,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False, # Set to True if your GPU supports FP16 and you want to use it
    bf16=torch.cuda.is_bf16_supported(), # Use bfloat16 if supported by your AMD GPU
    max_grad_norm=0.3,
    max_steps=-1, # Set to -1 to run for num_train_epochs
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard", # Or "none" if you don't use TensorBoard
    disable_tqdm=False,
)

# --- 4-bit Quantization Configuration (QLoRA) ---
# Note: bitsandbytes might not fully support ROCm on Windows.
# On Linux with ROCm, it might work if compiled correctly or if a compatible wheel is available.
# If you encounter issues, you might need to skip 4-bit quantization or use 8-bit.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    bnb_4bit_use_double_quant=True,
)

# --- Load Model and Tokenizer ---
print(f"Loading model: {MODEL_NAME}")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto", # Automatically maps model to available devices
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
)
model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Mistral models prefer right padding

# --- Prepare Model for QLoRA Training ---
model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    r=LORA_R,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=LORA_TARGET_MODULES,
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# --- Load Dataset ---
print(f"Loading dataset: {DATASET_NAME}")
dataset = load_dataset(DATASET_NAME, split="train")

# --- Define Formatting Function for SFTTrainer ---
# This function formats your dataset into the chat/instruction format expected by Mistral
def formatting_prompts_func(examples):
    output_texts = []
    for i in range(len(examples['instruction'])):
        instruction = examples['instruction'][i]
        response = examples['output'][i]
        context = examples['input'][i]

        if context:
            # Mistral Instruct format
            text = f"[INST] {instruction}\n{context} [/INST]\n{response}"
        else:
            text = f"[INST] {instruction} [/INST]\n{response}"
        output_texts.append(text)
    return output_texts

# --- Initialize SFTTrainer ---
print("Initializing SFTTrainer...")
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    formatting_func=formatting_prompts_func,
    max_seq_length=2048, # Adjust based on your data and VRAM
    tokenizer=tokenizer,
    args=TRAINING_ARGS,
    packing=False, # Set to True for more efficient packing of short sequences
)

# --- Train the Model ---
print("Starting training...")
trainer.train()

# --- Save the Fine-tuned Model ---
print(f"Saving fine-tuned model to {OUTPUT_DIR}/final_checkpoint")
trainer.save_model(f"{OUTPUT_DIR}/final_checkpoint")

# Save the tokenizer as well
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_checkpoint")

print("Fine-tuning complete!")
