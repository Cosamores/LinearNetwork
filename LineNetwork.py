#!/usr/bin/env python instead 

#from IPython.display import HTML
import torch
import numpy as np
from torch import nn

class LineNetwork(nn.Module):
    # Inicialização
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 1)
        )
    
    # Como a rede computa
    def forward(self, x):
        return self.layers(x)



#######################################################


from torch.utils.data import Dataset, DataLoader
import torch.distributions.uniform as urand

class AlgebraicDataset(Dataset):
  def __init__(self, f, interval, nsamples):
    X = urand.Uniform(interval[0], interval[1]).sample([nsamples])
    self.data = [(x, f(x)) for x in X]

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    return self.data[idx]
    
line = lambda x: 2*x + 3
interval = (-10, 10)
train_nsamples = 1000
test_nsamples = 100
     
train_dataset = AlgebraicDataset(line, interval, train_nsamples)
test_dataset = AlgebraicDataset(line, interval, test_nsamples)

train_dataloader = DataLoader(train_dataset, batch_size=train_nsamples, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=test_nsamples, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Rodando na {device}")

model = LineNetwork().to(device)

# Erro quadrático médio (Mean Squared Error)
lossfunc = nn.MSELoss()
# Gradiente Descendente Estocástico
# SGD = Stochastic Gradient Descent
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# Taxa de aprendizado = lr ( learning rate )
def train(model, dataloader, lossfunc, optimizer):
    model.train()
    cumloss = 0.0

    for X, y in dataloader:
    # Envia os dados para o dispositivo definnido
        X = X.unsqueeze(1).float().to(device)
        y = y.unsqueeze(1).float().to(device)

        pred = model(X)
        loss = lossfunc(pred, y)

        # Zera os gradientes acumulados
        optimizer.zero_grad()
        # Computa os gradientes
        loss.backward()
        # Anda na direção que reduz o erro local
        optimizer.step()
        # Para obter a soma das perdas - loss é um tensor [[]]
        cumloss += loss.item()
    
    return cumloss / len(dataloader)

def test(model, dataloader, lossfunc):
    model.eval()
    cumloss = 0.0
    
    with torch.no_grad():
        for X, y in dataloader:
            # Envia os dados para o dispositivo definnido
            X = X.unsqueeze(1).float().to(device)
            y = y.unsqueeze(1).float().to(device)

            pred = model(X)
            loss = lossfunc(pred, y)
            cumloss += loss.item()
    
    return cumloss / len(dataloader)
    

#### PARA VISUALIZAR ######################################


import matplotlib.pyplot as plt

def plot_comparinson(f, model, interval=(-10, 10), nsamples=10):
  fig, ax = plt.subplots(figsize=(10, 10))

  ax.grid(True, which='both')
  ax.spines['left'].set_position('zero')
  ax.spines['right'].set_color('none')
  ax.spines['bottom'].set_position('zero')
  ax.spines['top'].set_color('none')

  samples = np.linspace(interval[0], interval[1], nsamples)
  model.eval()
  with torch.no_grad():
    pred = model(torch.tensor(samples).unsqueeze(1).float().to(device))

  ax.plot(samples, list(map(f, samples)), "o", label="ground truth")
  ax.plot(samples, pred.cpu(), label="model")
  plt.legend()
  plt.show()

epochs = 101
for t in range(epochs):
    train_loss = train(model, train_dataloader, lossfunc, optimizer)
        
    if t % 10 == 0:
        print(f"Epoch: {t}; Train Loss: {train_loss}")
        plot_comparinson(line, model)

test_loss = test(model, test_dataloader, lossfunc)
print(f"Test Loss: {test_loss}")



#############################################################################

class MultiLayerNetwork(nn.Module):
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
        nn.Linear(1, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
    )

  def forward(self, x):
    return self.layers(x)
     

multimodel = MultiLayerNetwork().to(device)
     

from math import cos
     

f = lambda x: cos(x/2)
     

train_dataset = AlgebraicDataset(f, interval, train_nsamples)
test_dataset = AlgebraicDataset(f, interval, test_nsamples)

train_dataloader = DataLoader(train_dataset, train_nsamples, shuffle=True)
test_dataloader = DataLoader(test_dataset, test_nsamples, shuffle=True)
     

lossfunc = nn.MSELoss()
optimizer = torch.optim.SGD(multimodel.parameters(), lr=1e-3)
     

epochs = 20001
for t in range(epochs):
  train_loss = train(multimodel, train_dataloader, lossfunc, optimizer)
  if t % 100 == 0:
    print(f"Epoch: {t}; Train Loss: {train_loss}")
    plot_comparinson(f, multimodel, nsamples=40)

test_loss = test(multimodel, test_dataloader, lossfunc)
print(f"Test Loss: {test_loss}")