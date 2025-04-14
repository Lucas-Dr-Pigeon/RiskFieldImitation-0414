import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import argparse
from itertools import product
from model.networks import GGDRiskFieldNetwork
from process.epsilon import get_processed_tensors
from torch.utils.data import DataLoader, TensorDataset

def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default="D:/Productivity/Paper/RSS2024-Journal/data")
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--beta', type=float, default=5)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-epochs', type=int, default=1000)
    parser.add_argument('--model-name', type=str, default='ggd-nobuffer')
    parser.add_argument('--train-test-split-ratio', type=float, default=0.8)
    parser.add_argument('--loglik-eps', type=float, default=1e-4)
    args = parser.parse_args()
    return args

def train(data, model, num_epochs, lr=1e-4, batch_size=64, input_dim=8, split_ratio=0.8, _eps=1e-4):
    train_loss_history = []
    test_loss_history = []
    # split train and tests
    trainlen = int(len(data)*split_ratio)
    trainset, testset = torch.utils.data.random_split(data, [trainlen, len(data)-trainlen])
    # create a dataloader 
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)
    # define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # enumerate epochs
    for ep_ in range(num_epochs):
        # batch_loss 
        train_batch_loss = np.zeros(len(trainloader))
        test_batch_loss = np.zeros(len(testloader))
        # enumerate train batches
        print (f"---- start training epoch #{ep_}/{num_epochs} ----")
        for id_batch, batch in enumerate(tqdm(trainloader)):
            model.train()
            input = batch[:,:input_dim]
            dist = batch[:,-2:]
            gamma = model(input)
            loglik = torch.log( 1 + _eps - torch.exp(-torch.abs(dist[:,0]/gamma[:,0])**5-torch.abs(dist[:,1]/gamma[:,1])**5) )
            u = torch.autograd.grad(loglik, gamma,
                            create_graph=True,
                            allow_unused=True,
                            grad_outputs=torch.ones_like(loglik)
                        )[0]
            ux = u[:,0]
            uy = u[:,1]
            uux = torch.autograd.grad(ux, gamma,
                            create_graph=True,
                            allow_unused=True,
                            grad_outputs=torch.ones_like(ux)
                        )[0][:,0]
            uuy = torch.autograd.grad(uy, gamma,
                            create_graph=True,
                            allow_unused=True,
                            grad_outputs=torch.ones_like(ux)
                        )[0][:,1]
            loss = torch.mean(uux + uuy)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_batch_loss[id_batch] = loss.cpu()
        train_epoch_loss = np.mean(train_batch_loss)
        train_loss_history.append(train_epoch_loss)

        print (f"---- start testing epoch #{ep_}/{num_epochs} ----")
        for id_batch, batch in enumerate(tqdm(testloader)):
            model.eval()
            input = batch[:,:input_dim]
            dist = batch[:,-2:]
            gamma = model(input)
            loglik = torch.log( 1 + _eps - torch.exp(-torch.abs(dist[:,0]/gamma[:,0])**5-torch.abs(dist[:,1]/gamma[:,1])**5) )
            u = torch.autograd.grad(loglik, gamma,
                            create_graph=True,
                            allow_unused=True,
                            grad_outputs=torch.ones_like(loglik)
                        )[0]
            ux = u[:,0]
            uy = u[:,1]
            uux = torch.autograd.grad(ux, gamma,
                            create_graph=True,
                            allow_unused=True,
                            grad_outputs=torch.ones_like(ux)
                        )[0][:,0]
            uuy = torch.autograd.grad(uy, gamma,
                            create_graph=True,
                            allow_unused=True,
                            grad_outputs=torch.ones_like(ux)
                        )[0][:,1]
            loss = torch.mean(uux + uuy)
            test_batch_loss[id_batch] = loss.cpu()
        test_epoch_loss = np.mean(test_batch_loss) 
        test_loss_history.append(test_epoch_loss)
        # save weights to checkpoint
        # model.save_checkpoint()

        print (f"---- epoch #{ep_}/{num_epochs} train loss: {train_epoch_loss} test loss: {test_epoch_loss} ----")


if __name__ == '__main__':
    Args = add_arguments()
    print (f"---- Reading raw data ----")
    data = pd.concat([ pd.read_csv(f"{Args.data_path}/{r:02}_spacings_2.csv",index_col=0) for r in trange(1,61)])
    print (f"---- Generating tensor ----")
    dataset = get_processed_tensors(data).to('cuda')
    print (f"---- Dataset dimension {dataset.shape} ----")
    ggd = GGDRiskFieldNetwork(input_dim=dataset.shape[1]-2,
                              beta=Args.beta,
                              name=Args.model_name,
                              )
    train(
        dataset,
        ggd,
        num_epochs=Args.num_epochs,
        lr=Args.learning_rate,
        batch_size=Args.batch_size,
        input_dim=dataset.shape[1]-2,
        split_ratio=Args.train_test_split_ratio,
        _eps=Args.loglik_eps,
    )
    
