import datasets
import pandas as pd
from datasets import load_dataset, concatenate_datasets, Dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from new_data import new_data

# Dataset
raw_dataset = load_dataset("HebArabNlpProject/HebrewSentiment",
                           data_files={'train': 'HebSentiment_train.jsonl', 'val': 'HebSentiment_val.jsonl',
                                       'test': 'HebSentiment_test.jsonl'})

# Model
checkpoint = "avichr/heBERT_sentiment_analysis"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Define label maps
id2label = {0: "Neutral", 1: "Positive", 2: 'Negative'}
label2id = {'Neutral': 0, 'Positive': 1, 'Negative': 2}

model = AutoModelForSequenceClassification.from_pretrained(checkpoint,
                                                           num_labels=3, id2label=id2label, label2id=label2id)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# Function to tokenize text
def tokenize_text(examples):
    # extract text
    text = examples["text"]

    # Tokenize and truncate text
    tokenizer.truncation_side = "left"
    tokenized_inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512
    )

    return tokenized_inputs


# Function to transform labels
def map_labels(example):
    example['labels'] = label2id[example['labels'].strip()]
    return example


# Dataset preparation function
def prepare_dataset(dataset, columns_to_remove):
    """
    Function to preprocess the dataset, including tokenization and column removal.
    """

    # Apply tokenization
    processed_dataset = dataset.map(tokenize_text, batched=True)

    # Remove unnecessary columns
    processed_dataset = processed_dataset.remove_columns(columns_to_remove)

    # Rename 'tag_ids' to 'labels' to align with model input
    processed_dataset = processed_dataset.rename_column("tag_ids", "labels")

    # Transform labels (e.g., 'Neutral', 'Positive', 'Negative' to numeric)
    processed_dataset = processed_dataset.map(map_labels)

    return processed_dataset


# Function to filter by specific label
def filter_by_label(dataset, label):
    """
    Filters dataset based on label value.
    """
    return dataset.filter(lambda x: x['labels'] == label)


# Balancing the dataset by class samples
def balance_dataset_by_samples(processed_dataset, sample_size):
    """
    Function to balance the dataset by ensuring each classification (neutral, positive, negative)
    has an equal number of samples.
    """

    # Filter datasets by label
    neutral_dataset = filter_by_label(processed_dataset, 0)
    positive_dataset = filter_by_label(processed_dataset, 1)
    negative_dataset = filter_by_label(processed_dataset, 2)

    # Shuffle and select the specified number of samples for each class
    neutral_sample = neutral_dataset.shuffle(seed=42).select(range(sample_size))
    positive_sample = positive_dataset.shuffle(seed=42).select(range(sample_size))
    negative_sample = negative_dataset.shuffle(seed=42).select(range(sample_size))

    # Concatenate the balanced samples into a single dataset
    balanced_dataset = concatenate_datasets([neutral_sample, negative_sample, positive_sample])

    return balanced_dataset


# Columns to remove from dataset
columns_to_drop = ['id', 'task_name', 'campaign_id', 'annotator_agreement_strength', 'survey_name', 'industry', 'type',
                   'text']

# Prepare the dataset by tokenizing and cleaning
processed_datasets = prepare_dataset(raw_dataset, columns_to_drop)

# Define the sample sizes for train, validation, and test sets
SAMPLE_SIZE = {"train": 6500, "val": 370, "test": 400}

# Balance the train, validation, and test datasets
balanced_train_dataset = balance_dataset_by_samples(processed_datasets["train"], SAMPLE_SIZE["train"])
balanced_val_dataset = balance_dataset_by_samples(processed_datasets["val"], SAMPLE_SIZE["val"])
balanced_test_dataset = balance_dataset_by_samples(processed_datasets["test"], SAMPLE_SIZE["test"])


def join_and_split_datasets(balanced_train_dataset, balanced_val_dataset, balanced_test_dataset, train_ratio=0.8,
                            val_test_ratio=0.1, seed=42):
    """
    Function to join train, validation, and test datasets, shuffle them, and split them into
    new training, validation, and test datasets based on specified ratios.

    Args:
        balanced_train_dataset: The original training dataset.
        balanced_val_dataset: The original validation dataset.
        balanced_test_dataset: The original test dataset.
        train_ratio: The ratio of the dataset to allocate for training (default 0.8 or 80%).
        val_test_ratio: The ratio of the dataset to allocate for validation and test (default 0.1 or 10% each).
        seed: Seed for random shuffling (default 42).

    Returns:
        new_balanced_train_dataset: The new training dataset (80%).
        new_balanced_val_dataset: The new validation dataset (10%).
        new_balanced_test_dataset: The new test dataset (10%).
    """

    # Step 1: Concatenate the datasets (train + validation + test) into one dataset
    combined_dataset = concatenate_datasets([balanced_train_dataset, balanced_val_dataset, balanced_test_dataset])

    # Step 2: Shuffle the combined dataset to ensure randomness
    shuffled_dataset = combined_dataset.shuffle(seed=seed)

    # Step 3: Calculate sizes based on the ratios
    total_size = len(shuffled_dataset)
    train_size = int(train_ratio * total_size)  # 80% for training
    val_test_size = total_size - train_size  # 20% for validation + test

    # Split the remaining 20% equally into validation and test sets
    val_size = test_size = val_test_size // 2  # Split the remaining 20% into validation and test

    # Step 4: Select the new training, validation, and test datasets
    new_balanced_train_dataset = shuffled_dataset.select(range(train_size))  # 80% for training
    new_balanced_val_dataset = shuffled_dataset.select(
        range(train_size, train_size + val_size))  # First half of 20% for validation
    new_balanced_test_dataset = shuffled_dataset.select(
        range(train_size + val_size, total_size))  # Second half of 20% for test

    # Return the newly split datasets
    return new_balanced_train_dataset, new_balanced_val_dataset, new_balanced_test_dataset


# Example of how to use the function
new_balanced_train_dataset, new_balanced_val_dataset, new_balanced_test_dataset = join_and_split_datasets(
    balanced_train_dataset, balanced_val_dataset, balanced_test_dataset
)

# Output the sizes of the new datasets
print(f"New train size: {len(new_balanced_train_dataset)}")
print(f"New validation size: {len(new_balanced_val_dataset)}")
print(f"New test size: {len(new_balanced_test_dataset)}")


def add_new_data_to_dataset(data):
    """
        This function takes in new data, tokenizes the text, maps the labels to numerical values,
        and prepares it to be concatenated with an existing dataset.
    """

    df = datasets.Dataset.from_pandas(pd.DataFrame(data=data))

    new_dataset_tokenized = df.map(tokenize_text, batched=True)

    new_dataset_tokenized = new_dataset_tokenized.remove_columns(['text'])

    new_dataset_tokenized = new_dataset_tokenized.map(map_labels)

    return new_dataset_tokenized

# df = datasets.Dataset.from_pandas(pd.DataFrame(data=new_data))
#
# print(df)
# print(new_balanced_train_dataset[0])
#
# df_dataset = df.map(tokenize_text, batched=True)
#
# # Apply the tokenizer to the new dataset
# new_dataset_tokenized = df.map(tokenize_text, batched=True)
#
# # Step 5: Remove the 'text' column
# new_dataset_tokenized = new_dataset_tokenized.remove_columns(['text'])
#
# new_dataset_tokenized = new_dataset_tokenized.map(map_labels)
# # Step 6: Ensure the column names match between datasets
# print("Existing dataset columns:", new_balanced_train_dataset.column_names)
# print("Tokenized new dataset columns:", new_dataset_tokenized.column_names)


new_data_processed = add_new_data_to_dataset(new_data)
combined_dataset = concatenate_datasets([new_balanced_train_dataset, new_data_processed])

print(combined_dataset[0])
