from transformers import AutoTokenizer, AutoModel, pipeline
import pandas as pd
import configuration

sentiment_analysis = pipeline(
    "sentiment-analysis",
    model=configuration.PATH_TO_MY_MODEL,
    tokenizer=configuration.PATH_TO_TOKENIZER,
    return_all_scores=False
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
