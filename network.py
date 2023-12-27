import torch
from torch import nn, relu, tanh
from torch.nn import Conv1d, Linear, GRU
from torch.utils.data import TensorDataset
import math

class FxDataset(TensorDataset):
    def __init__(self, x, y, window_len, batch_size):
        super(TensorDataset, self).__init__()

        len_index = len(x) - (len(x) % window_len)
        self.x = x[:len_index].reshape(-1, 1)
        self.y = y[window_len-1 : len_index].reshape(-1, 1)
        self.window_len = window_len
        self.batch_size = batch_size

    def __getitem__(self, index):
        x_out = []
        for i in range(index*self.batch_size, (index+1)*self.batch_size):
            x_out.append(self.x[i : i+self.window_len])
        x_out = torch.stack(x_out)
        y_out = self.y[index*self.batch_size : (index+1)*self.batch_size]
        
        return x_out, y_out
    
    def __len__(self):
        return math.floor((len(self.x) - self.window_len + 1) / self.batch_size)
    
class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, x, y):
        return torch.sqrt(self.mse(x, y))

class ConvRNN(nn.Module):
    def __init__(self):
        super(ConvRNN, self).__init__()
        self.conv1 = Conv1d(1, 16, 12, stride=4)
        self.conv2 = Conv1d(16, 32, 12, stride=3)
        self.rec3 = GRU(32, 64)
        self.fc4 = Linear(64, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.permute(2, 0, 1)
        x, self.hidden = self.rec3(x, self.hidden)
        x = self.fc4(x)
        return x[-1, :, :]
    
    def reset_hidden(self, batch_size, device):
        self.hidden = torch.zeros((1, batch_size, 64)).to(device)

    def train_epoch(self, dataset:FxDataset, loss_function, optimizer:torch.optim.Optimizer, device, shuffle=False):
        if shuffle == True:
            shuffle = torch.randperm(len(dataset))
        else:
            shuffle = range(len(dataset))
        epoch_loss = 0
        for i in range(len(dataset)):
            x, y = dataset[shuffle[i]]
            batch_size = x.shape[0]
            optimizer.zero_grad()
            self.reset_hidden(batch_size, device)
            y_pred = self.forward(x)
            loss = loss_function(y, y_pred)
            loss.backward()
            optimizer.step()
            epoch_loss += loss
            print("Frame {}/{}: {:.2%}".format(i, len(dataset), i/len(dataset)), end='\r')
        epoch_loss /= (i + 1)
        return epoch_loss
    
    def valid_epoch(self, dataset:FxDataset, loss_function, device):
        with torch.no_grad():
            epoch_loss = 0
            for i in range(len(dataset)):
                x, y = dataset[i]
                batch_size = x.shape[0]
                self.reset_hidden(batch_size, device)
                y_pred = self.forward(x)
                loss = loss_function(y, y_pred)
                epoch_loss = epoch_loss + loss
            epoch_loss = epoch_loss / (i + 1)
            return epoch_loss