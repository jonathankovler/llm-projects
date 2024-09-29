from numpy.ma.setup import configuration
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import configuration
from src.sentiment_analysis import sentiment_analysis

# Load the fine-tuned model and tokenizer
model_path = "./heBERT-finetuned-news-sc2"  # The directory where you saved your model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

sentiment_analysis2= pipeline(
    "sentiment-analysis",
    model=configuration.PATH_TO_MY_MODEL,
    tokenizer=configuration.PATH_TO_TOKENIZER,
    return_all_scores=False,
    device=0

)
# List of sentences to classify
texts = ["חיסלו מחבלים שבאו להרוג יהודים", "סנקציות נגד ישראל: לא תוכל להשתתף בשנה הבאה באולימפיאדת מדעי המחשב",
         "ישראל הפסידה", "חמאס הפסידו במלחמה", "20 ישראלים נהרגו בעזה"]

for text in texts:

    print(f"fine-tuned model-->> text{text}, prediction{sentiment_analysis(text)}")
    print()
