import pandas as pd
from nltk.tokenize import word_tokenize

def preprocess_data(df):
    df['sms'] = df['sms'].apply(lambda x: x.lower())
    df['sms'] = df['sms'].apply(word_tokenize)
    return df

def get_text_corpus(texts):
    corpus = [word for text in texts for word in text]
    return list(set(corpus))