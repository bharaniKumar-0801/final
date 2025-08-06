

import re
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_and_tokenize(text: str):
    text = text.lower()
    doc = nlp(text)
    tokens = []
    for token in doc:
        if not token.is_punct and not token.is_space and token.text not in stop_words:
            tokens.append(lemmatizer.lemmatize(token.text.strip()))
    return tokens

def extract_named_entities(text):
    doc = nlp(text)
    return [ent.text.lower() for ent in doc.ents]

def extract_phone_numbers(text):
    return re.findall(r'\+?\d[\d\- ]{7,}\d', text)

def extract_emails(text):
    return re.findall(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', text)

def extract_urls(text):
    return re.findall(r'https?://\S+', text)

def extract_dates(text):
    return re.findall(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}', text)

def enrich_tokens(text: str):
    base = clean_and_tokenize(text)
    enriched = (
        extract_named_entities(text) +
        extract_phone_numbers(text) +
        extract_emails(text) +
        extract_urls(text) +
        extract_dates(text)
    )
    enriched_tokens = clean_and_tokenize(" ".join(enriched))
    return base + enriched_tokens
