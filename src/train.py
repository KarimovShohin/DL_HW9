import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.make_dataset import load_data, preprocess_data, get_text_corpus  
from src.model.LSTMNet import LSTMNet  
from sklearn.model_selection import train_test_split
from gensim.models import FastText  
import logging  

logging.basicConfig(filename='logs/training.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def binary_accuracy(preds, y):
    rounded_preds = torch.round(preds)
    correct = (rounded_preds == y).float() 
    acc = correct.sum() / len(correct)
    return acc

class FastTextDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, max_length=50):
        self.X = [torch.tensor(self.pad_sequence(x, max_length), dtype=torch.float32) for x in X] 
        self.y = torch.tensor(y.tolist(), dtype=torch.float32)
        self.lengths = [len(x) for x in X]  

    def pad_sequence(self, sequence, max_length):
        if len(sequence) < max_length:
            return np.pad(sequence, ((0, max_length - len(sequence)), (0, 0)), mode='constant')
        else:
            return sequence[:max_length]  

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index], self.lengths[index] 

def vectorize_texts(texts, model):
    return [model.wv[text] for text in texts]  

def train(
    model: nn.Module,
    config: dict,
):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    

    model.to(device)

    df = load_data()
    df = preprocess_data(df)

    corpus = get_text_corpus(df['sms'].values)

    X_train, X_test, y_train, y_test = train_test_split(
        df.sms.values, df.label.values, test_size=0.3, random_state=42)

    ft_model = FastText(sentences=X_train.tolist(), vector_size=config['training_params']['embedding_dim'], window=3, min_count=1)
    
    X_train_vectors = vectorize_texts(X_train.tolist(), ft_model)
    X_test_vectors = vectorize_texts(X_test.tolist(), ft_model)

    train_dataset = FastTextDataset(X_train_vectors, y_train)
    test_dataset = FastTextDataset(X_test_vectors, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=config['training_params']['batch_size'], collate_fn=lambda x: tuple(zip(*x)))
    test_dataloader = DataLoader(test_dataset, batch_size=config['training_params']['batch_size'], collate_fn=lambda x: tuple(zip(*x)))

    optimizer = optim.Adam(model.parameters(), lr=config['training_params']['learning_rate'])
    criterion = nn.BCELoss()

    num_epochs = config['training_params']['num_epochs']

    for epoch in range(num_epochs):
        model.train()
        train_losses_epoch = []

        for batch in tqdm(train_dataloader):  
            X_batch, y_batch, text_lengths = batch 
            optimizer.zero_grad()
            X_batch = torch.stack(X_batch).to(device)  
            y_batch = torch.stack(y_batch).to(device)  
            text_lengths = torch.tensor(text_lengths).to(device)  
            output_train = model(X_batch, text_lengths).squeeze()  
            loss_train = criterion(output_train.float(), y_batch.float())
            loss_train.backward()
            optimizer.step()
            train_losses_epoch.append(loss_train.item())
            
        logging.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {np.mean(train_losses_epoch):.4f}")
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {np.mean(train_losses_epoch):.4f}")

if __name__ == "__main__":
    train(model)