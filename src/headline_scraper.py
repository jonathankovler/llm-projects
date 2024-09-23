import requests
from bs4 import BeautifulSoup
import pandas as pd
import time, os
from datetime import datetime
import configuration


def fetch_and_update_headlines():
    url = configuration.WEB_URL
    file_path = configuration.PATH_TO_TABLE
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        headlines = soup.find_all(name='div', class_="AccordionSection")[0:100]

        new_headlines = []

        for row in headlines:
            headline = row.get_text(strip=True)
            date = None
            time_tag = row.find('time')
            if time_tag:
                date = time_tag.get('datetime', '')[:-5]
                # date_object = datetime.strptime(date, "%Y-%m-%d").date()
            new_headlines.append([headline, date])

        # Check if the CSV exists, and read it or create a new DataFrame
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, encoding='utf-8-sig')
        else:
            df = pd.DataFrame(columns=['Headline', 'Datetime'])  # Create empty DataFrame if file does not exist

        latest_headlines = df.head(100)
        latest_dates = set(latest_headlines['Datetime'])

        # Find new headlines by comparing dates
        new_entries = [entry for entry in new_headlines if entry[1] not in latest_dates]

        if new_entries:
            df_new_entries = pd.DataFrame(new_entries, columns=['Headline', 'Datetime'])
            df_combined = pd.concat([df_new_entries, df]).drop_duplicates(subset=['Headline', 'Datetime'],
                                                                          keep='first')
            df_combined.to_csv(file_path, index=False, encoding='utf-8-sig')
            print("Updated CSV with new headlines.", df_new_entries)
        else:
            print("No new headlines to update.")

    else:
        print(f'Failed to retrieve the website. Status code: {response.status_code}')


if __name__ == '__main__':

    while True:
        fetch_and_update_headlines()
        time.sleep(120)

