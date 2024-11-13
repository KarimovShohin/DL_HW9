import yaml
from src.train import train 
from src.model.LSTMNet import LSTMNet  
import torch


with open("cfg/config.yaml", "r") as f:
    config = yaml.safe_load(f)

with open("cfg/model_params.yaml", "r") as f:
    model_params = yaml.safe_load(f)

model = LSTMNet(**model_params)

device = 'cpu'
model.to(device)

train(model, config)