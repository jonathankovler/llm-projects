from transformers import AutoTokenizer, AutoModel, pipeline, AutoModelForSequenceClassification
import pandas as pd
import configuration
import torch
import transformers

device_id = 0 if torch.cuda.is_available() else -1

tokenizer = transformers.BertTokenizerFast.from_pretrained(configuration.PATH_TO_TOKENIZER)
model = transformers.BertForSequenceClassification.from_pretrained(configuration.PATH_TO_MY_MODEL).cuda()

sentiment_analysis = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer,
    return_all_scores=False,
    device=device_id

)

df = pd.read_csv(
    configuration.PATH_TO_TABLE,
    encoding='utf-8-sig')


# Function to add sentiment to the csv file
def analyze_sentiment_and_append(row):
    headline = row['Headline'][0]
    sent_analysis = sentiment_analysis(headline)[0]
    # max_result = max(sent_analysis[0], key=lambda x: x['score'])
    row['Sentiment'] = sent_analysis['label']
    row['Score'] = sent_analysis['score']
    return row


if __name__ == '__main__':
    try:
        df = df.apply(analyze_sentiment_and_append, axis=1)
        df.to_csv(configuration.PATH_TO_PROCESSED_TABLE, index=False, encoding='utf-8-sig')
        print("created")

    except Exception as ex:
        print(ex)
