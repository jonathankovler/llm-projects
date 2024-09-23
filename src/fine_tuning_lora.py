from datasets import load_dataset, DatasetDict, Dataset

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer)

from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig
import evaluate
import torch
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# Dataset
raw_dataset = load_dataset("HebArabNlpProject/HebrewSentiment",
                           data_files={'train': 'HebSentiment_train.jsonl', 'val': 'HebSentiment_val.jsonl',
                                       'test': 'HebSentiment_test.jsonl'})

# Model
checkpoint = "avichr/heBERT_sentiment_analysis"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# define label maps
id2label = {0: "Neutral", 1: "Positive", 2: 'Negative'}
label2id = {'Neutral': 0, 'Positive': 1, 'Negative': 2}

model = AutoModelForSequenceClassification.from_pretrained(checkpoint,
                                                           num_labels=3, id2label=id2label, label2id=label2id)
# Preprocess data

# add pad token if none exists
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))


def tokenize_function(examples):
    # extract text
    text = examples["text"]

    # tokenize and truncate text
    tokenizer.truncation_side = "left"
    tokenized_inputs = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        max_length=512
    )

    return tokenized_inputs


tokenized_dataset = raw_dataset.map(tokenize_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

print(tokenized_dataset)

# Organizing the dataset

columns_to_remove = ['id', 'task_name', 'campaign_id', 'annotator_agreement_strength', 'survey_name', 'industry',
                     'type', 'text']

tokenized_dataset = tokenized_dataset.remove_columns(columns_to_remove)
tokenized_dataset = tokenized_dataset.rename_column("tag_ids", "labels")

label_map = {'Neutral': 0, 'Positive': 1, 'Negative': 2}


def transform_labels(example):
    example['labels'] = label_map[example['labels'].strip()]
    return example


tokenized_dataset = tokenized_dataset.map(transform_labels)

# Evaluation

# import accuracy evaluation metric
accuracy = evaluate.load("accuracy")

# define an evaluation function to pass into trainer later
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    # Calculate accuracy
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)

    # Calculate F1 score (you may need to specify 'average' for multi-class tasks)
    f1 = f1_metric.compute(predictions=predictions, references=labels,
                           average="weighted")  # Use "weighted" for multi-class

    return {
        "accuracy": accuracy["accuracy"],
        "f1": f1["f1"],
    }


def get_highest_label(prediction):
    return max(prediction, key=lambda x: x['score'])['label']


# define list of examples
text_list = ["היה לי יום מעולה.", "לא אהבתי כל כך, לא ממליץ.", "ישראל נצחה במלחמה",
             "היה לי יום סביר", "צהל חיסל מחבל"]

print("Untrained model predictions:")
print("----------------------------")
for text in text_list:
    # tokenize text
    inputs = tokenizer.encode(text, return_tensors="pt")
    # compute logits
    logits = model(inputs).logits
    # convert logits to label
    sentiment_prediction = torch.argmax(logits)

    print(text + " - " + id2label[sentiment_prediction.tolist()])

# Train Model

peft_config = LoraConfig(task_type="SEQ_CLS",
                         r=4,
                         lora_alpha=32,
                         lora_dropout=0.01,
                         target_modules=['query'])

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

small_train_dataset = tokenized_dataset["train"].shuffle(seed=42).select(range(5000))
small_test_dataset = tokenized_dataset["test"].shuffle(seed=42).select(range(1500))
small_val_dataset = tokenized_dataset["val"].shuffle(seed=42).select(range(200))
# hyperparameters
lr = 1e-5
batch_size = 6
num_epochs = 10

# define training arguments
training_args = TrainingArguments(
    output_dir=r'results/lora-sc',
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# create trainer object
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,  # this will dynamically pad examples in each batch to be equal length
    compute_metrics=compute_metrics,
)

if __name__ == '__main__':
    trainer.train()
    predictions = trainer.predict(small_val_dataset)
    print(predictions.predictions.shape, predictions.label_ids.shape)

    # Get predicted class labels
    predicted_labels = np.argmax(predictions.predictions, axis=1)

    # Compare with true labels
    true_labels = predictions.label_ids

    # Calculate accuracy
    accuracy = np.mean(predicted_labels == true_labels)
    print(f"Accuracy: {accuracy:.4f}")
