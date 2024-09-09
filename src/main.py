from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

# Load GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# GPT-2 doesn't have a padding token, so we set it
tokenizer.pad_token = tokenizer.eos_token

# Load the model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Also set the model's padding token
model.config.pad_token_id = tokenizer.pad_token_id

# Load the IMDb dataset
dataset = load_dataset("imdb")

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    weight_decay=0.01,
)

# Initialize the Trainer with the original GPT-2 model for evaluation
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"].shuffle().select(range(1000)),  # Smaller dataset for faster testing
    eval_dataset=tokenized_datasets["test"].select(range(500)),  # Smaller dataset for evaluation
)

# Evaluate the original GPT-2 model's baseline performance
print("Evaluating the pre-trained GPT-2 model...")
trainer.evaluate()

# Create the LoRA config for parameter-efficient fine-tuning
lora_config = LoraConfig(
    r=8,  # rank
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none"
)

# Convert the GPT-2 model to a PEFT model using the LoRA config
peft_model = get_peft_model(model, lora_config)

# Check the number of trainable parameters in the PEFT model
peft_model.print_trainable_parameters()

# Initialize the Trainer with the PEFT model for training
trainer = Trainer(
    model=peft_model,  # Switch to PEFT model
    args=training_args,
    train_dataset=tokenized_datasets["train"].shuffle().select(range(1000)),  # Smaller dataset for faster testing
    eval_dataset=tokenized_datasets["test"].select(range(500)),  # Smaller dataset for evaluation
)

# Train the PEFT model
print("Training the PEFT model with LoRA...")
trainer.train()

# Save the fine-tuned PEFT model
peft_model.save_pretrained("./gpt2-lora")

# Final evaluation after training
print("Evaluating the fine-tuned PEFT model...")
trainer.evaluate()