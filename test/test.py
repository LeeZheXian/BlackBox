import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    # Note: We are NOT importing IntervalStrategy here to avoid API conflict
)
import os # Added os for good practice when creating directories

def main():

    # Using a common, small pre-trained model for quick testing
    model_name = "distilbert-base-uncased" 
    output_dir = "./checkpoints"

    # Ensure output directory exists before Trainer tries to use it
    os.makedirs(output_dir, exist_ok=True) 

    # 1. Device Check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2. Dataset Loading (The script will download this if not already present)
    dataset = load_dataset("imdb")

    # 3. Tokenizer and Data Processing
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=256,
        )
    
    dataset = dataset.map(tokenize, batched=True)
    dataset = dataset.rename_column("label", "labels")
    
    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    # Use the splits directly after processing
    # Using the full train/test splits (25k samples each)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    
    # 4. Model Initialization
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
    )

    model.to(device)

    # 5. Training Arguments (FIXED API keywords)
    training_args = TrainingArguments(
        output_dir = output_dir,
        
        # FIXED: Using string "steps" and providing eval_steps
        evaluation_strategy="steps", 
        eval_steps=500, # Evaluate every 500 steps
        
        logging_strategy="steps", 
        logging_steps=100,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        learning_rate=2e-5,
        weight_decay=0.01,
        fp16=True,
        report_to="none",
        save_total_limit=2,
    )

    # 6. Trainer Setup and Execution
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    # 7. Final Save
    trainer.save_model(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")
    print(f"\nModel and tokenizer successfully saved to: {output_dir}/final")

if __name__ == "__main__":
    main()