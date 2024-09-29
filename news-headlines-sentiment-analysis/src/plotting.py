import re
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import configuration
from wordcloud import WordCloud
from bidi.algorithm import get_display

df = pd.read_csv(
    configuration.PATH_TO_PROCESSED_TABLE,
    encoding='utf-8-sig')

# Convert the 'Datetime' column to a datetime object
df['Datetime'] = pd.to_datetime(df['Datetime'])

# Extract the date part only (ignore the time)
df['DateOnly'] = df['Datetime'].dt.date

# Drop rows without a date
df = df.dropna(subset=['DateOnly'])


def sentiment_percentage_for_date(input_date, sentiment):
    # Convert input date to datetime.date type
    input_date = pd.to_datetime(input_date).date()

    # Filter the DataFrame for the specific date (ignore time)
    single_day = df[df['DateOnly'] == input_date]

    if single_day.empty:
        return f"No data available for {input_date}"

    # Calculate how many times the specified sentiment appears on that day
    count_sentiment_occurrences = (single_day['Sentiment'] == sentiment).sum()

    # Calculate the percentage of that sentiment
    percentage = (count_sentiment_occurrences / single_day.shape[0]) * 100

    return percentage


def find_sentiment_extreme_day():
    unique_dates = df.DateOnly.unique()

    most_positive_day = None
    most_negative_day = None
    highest_positive_percentage = -1
    highest_negative_percentage = -1

    for date in unique_dates:

        positive_sentiment_percentage = sentiment_percentage_for_date(date, 'positive')
        negative_sentiment_percentage = sentiment_percentage_for_date(date, 'negative')

        if positive_sentiment_percentage > highest_positive_percentage:
            highest_positive_percentage = positive_sentiment_percentage
            most_positive_day = date

        if negative_sentiment_percentage > highest_negative_percentage:
            highest_negative_percentage = negative_sentiment_percentage
            most_negative_day = date

    highest_positive_percentage = int(highest_positive_percentage)
    highest_negative_percentage = int(highest_negative_percentage)
    print(f"most positive day: {most_positive_day}, with {highest_positive_percentage}% of positive headlines")
    print(f"most negative day: {most_negative_day}, with {highest_negative_percentage}% of negative headlines")


def sentiment_mapping():
    # Calculate the percentage of each sentiment type
    sentiment_type_percentages = df['Sentiment'].value_counts(normalize=True) * 100

    # Define the colors for each sentiment
    color_mapping = {
        'positive': 'green',
        'negative': 'red',
        'neutral': 'orange'
    }

    # Create the pie chart
    fig, ax = plt.subplots()
    ax.pie(
        sentiment_type_percentages,
        labels=sentiment_type_percentages.index,
        colors=[color_mapping.get(sentiment, 'grey') for sentiment in sentiment_type_percentages.index],
        # Use the color mapping
        explode=[0, 0, 0],  # No explosion effect
        autopct='%1.1f%%'  # Display percentages
    )

    # Show the chart
    plt.show()


# Create DataFrame


# Function to plot sentiment for a specific date
def plot_sentiment_for_date(input_date):
    # Convert input date to datetime.date type
    input_date = pd.to_datetime(input_date).date()

    # Filter the DataFrame for the specific date (ignore time)
    single_day = df[df['DateOnly'] == input_date]

    if single_day.empty:
        print(f"No data available for the date {input_date}")
        return

    # Group the sentiments and count them
    sentiment_counts = single_day['Sentiment'].value_counts().reindex(['negative', 'neutral', 'positive'], fill_value=0)

    # Plot the data
    COLORS = ["red", "orange", "green"]
    TITLE = f"Negative, neutral, and positive sentiment for {input_date}"

    # Convert the counts into a DataFrame
    plot_day = pd.DataFrame([sentiment_counts])
    plot_day.columns = ['negative', 'neutral', 'positive']

    # Plot a stacked bar chart
    plot_day.plot.bar(stacked=True, color=COLORS, title=TITLE, figsize=(10, 6))

    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.show()


def load_stopwords(file_path):
    with open(file_path, 'r', encoding="utf-8") as file:
        stopwords = file.read().splitlines()

    return set(stopwords)


# Function to remove a list of words from a sentence
def remove_stopwords(sentence, stopwords):
    # Create a regular expression pattern that matches any word in the list
    pattern = r'\b(' + '|'.join(re.escape(word) for word in stopwords) + r')\b'

    # Use re.sub() to replace the matched words with an empty string
    cleaned_sentence = re.sub(pattern, '', sentence)

    # Remove any extra spaces that may have been introduced
    cleaned_sentence = re.sub(r'\s+', ' ', cleaned_sentence).strip()

    return cleaned_sentence


def plot_most_common_words():
    # Combine all the headlines into a single string
    text = ' '.join(df["Headline"].tolist())

    stop_words = load_stopwords(configuration.HEBREW_STOP_WORDS)

    text_processed = remove_stopwords(sentence=text, stopwords=stop_words)

    bidi_text = get_display(text_processed)
    # Create a WordCloud object
    meta_mask = np.array(Image.open(r'C:\Users\kovle\PycharmProjects\news-headlines-sentiment-analysis\Ynet.png'))

    wc = WordCloud(font_path=r'C:\Windows\Fonts\courbd.ttf', mask=meta_mask, background_color='white',
                   colormap='Reds',
                   max_words=100).generate(bidi_text)

    # Plot the wordcloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.show()


plot_most_common_words()
find_sentiment_extreme_day()
sentiment_mapping()
