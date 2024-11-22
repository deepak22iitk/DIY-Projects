from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling

# Load dataset
dataset = load_dataset("json", data_files={"train": "train.jsonl", "test": "test.jsonl"})

# Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Set padding token

# Tokenize dataset
def preprocess_data(examples):
    #inputs = [example["prompt"] for example in examples["train"]]
    inputs = examples["prompt"]
    outputs = examples["response"]
    model_inputs = tokenizer(inputs, outputs, truncation=True, padding=True, max_length=512)
    model_inputs["labels"] = model_inputs["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_data, batched=True)

# Load pre-trained model
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Causal LM (GPT2) does not use masked language modeling
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./fine_tuned_gpt2",
    eval_strategy="epoch",
    save_strategy="epoch",  # Save the model after each epoch
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    save_steps=500,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=100,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=500,
    load_best_model_at_end=True,
    fp16=True,
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_gpt2")
tokenizer.save_pretrained("./fine_tuned_gpt2")
