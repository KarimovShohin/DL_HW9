import pandas as pd
import datasets
from nltk.tokenize import word_tokenize
import nltk
import os


nltk.download('punkt_tab')

def load_data():
    file_path = os.path.join(os.path.dirname(__file__), '../../data/sms_data.csv')
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    df['sms'] = df['sms'].apply(lambda x: x.lower())
    df['sms'] = df['sms'].apply(word_tokenize)
    return df

def get_text_corpus(texts):
    corpus = [word for text in texts for word in text]
    return list(set(corpus))

def save_data(df, filename='data/dataset.csv'):
    df.to_csv(filename, index=False)

if __name__ == "__main__":
    df = load_data()
    df = preprocess_data(df)
    
    save_data(df)