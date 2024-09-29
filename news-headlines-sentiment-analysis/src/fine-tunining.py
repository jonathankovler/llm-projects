from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, \
    DataCollatorWithPadding
import evaluate
import numpy as np

# Import the preprocessed datasets from preprocessing.py
from preprocessing import new_balanced_train_dataset, new_balanced_val_dataset, new_balanced_test_dataset, \
    data_collator, combined_dataset

# Model and tokenizer setup
checkpoint = "avichr/heBERT_sentiment_analysis"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint,
    num_labels=3,  # Assuming 3 classes: Neutral, Positive, Negative
    id2label={0: "Neutral", 1: "Positive", 2: "Negative"},
    label2id={"Neutral": 0, "Positive": 1, "Negative": 2}
)

# Load the accuracy and f1 metrics from Hugging Face's evaluate library
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")


def compute_metrics(pred):
    logits, labels = pred
    predictions = np.argmax(logits, axis=-1)

    # Compute accuracy and F1-score
    accuracy_result = accuracy.compute(predictions=predictions, references=labels)
    f1_result = f1.compute(predictions=predictions, references=labels, average="weighted")

    # Return both accuracy and F1-score
    return {
        "accuracy": accuracy_result["accuracy"],
        "f1": f1_result["f1"]
    }


# Define training arguments
training_args = TrainingArguments(
    output_dir="./heBERT-finetuned-news-sc2",
    eval_strategy="epoch",
    save_strategy="no",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    fp16=True,
    gradient_accumulation_steps=2
)

# Initialize Trainer with the model, dataset, tokenizer, and metrics
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=combined_dataset,
    eval_dataset=new_balanced_val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Save the model after training
trainer.save_model("./heBERT-finetuned-news-sc2")  # Explicitly save model

# Evaluate the model on the test set
results = trainer.evaluate(eval_dataset=new_balanced_test_dataset)
print("Test results:", results)
