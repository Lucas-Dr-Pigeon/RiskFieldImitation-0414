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

def train(data, model_x, model_y, num_epochs, lr=1e-4, batch_size=64, input_dim=8, split_ratio=0.8, _eps=1e-4):
    train_loss_history = []
    test_loss_history = []
    # split train and tests
    trainlen = int(len(data)*split_ratio)
    trainset, testset = torch.utils.data.random_split(data, [trainlen, len(data)-trainlen])
    # create a dataloader 
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)
    # define the optimizer
    optimizer_x = torch.optim.Adam(model_x.parameters(), lr=lr)
    optimizer_y = torch.optim.Adam(model_y.parameters(), lr=lr)
    # enumerate epochs
    for ep_ in range(num_epochs):
        # batch_loss 
        train_batch_loss = np.zeros(len(trainloader))
        test_batch_loss = np.zeros(len(testloader))
        # enumerate train batches
        print (f"---- start training epoch #{ep_}/{num_epochs} ----")
        for id_batch, batch in enumerate(tqdm(trainloader)):
            model_x.train()
            model_y.train()
            input = batch[:,:input_dim]
            dist = batch[:,-2:]
            gamma_x = model_x(input)
            gamma_y = model_y(input)
            loglik = torch.log( 1 + _eps - torch.exp(-torch.abs(dist[:,0]/gamma_x)**5-torch.abs(dist[:,1]/gamma_y)**5) )
            ux = torch.autograd.grad(loglik, gamma_x,
                            create_graph=True,
                            allow_unused=True,
                            grad_outputs=torch.ones_like(loglik)
                        )[0]
            uy = torch.autograd.grad(loglik, gamma_y,
                            create_graph=True,
                            allow_unused=True,
                            grad_outputs=torch.ones_like(loglik)
                        )[0]

            uux = torch.autograd.grad(ux, gamma_x,
                            create_graph=True,
                            allow_unused=True,
                            grad_outputs=torch.ones_like(ux)
                        )[0]
            uuy = torch.autograd.grad(uy, gamma_y,
                            create_graph=True,
                            allow_unused=True,
                            grad_outputs=torch.ones_like(uy)
                        )[0]
            loss = torch.mean(uux + uuy)
            optimizer_x.zero_grad()
            optimizer_y.zero_grad()
            loss.backward()
            optimizer_x.step()
            optimizer_y.step()
            train_batch_loss[id_batch] = loss.cpu()
        train_epoch_loss = np.mean(train_batch_loss)
        train_loss_history.append(train_epoch_loss)

        print (f"---- start testing epoch #{ep_}/{num_epochs} ----")
        for id_batch, batch in enumerate(tqdm(testloader)):
            model_x.eval()
            model_y.eval()
            input = batch[:,:input_dim]
            dist = batch[:,-2:]
            gamma_x = model_x(input)
            gamma_y = model_y(input)
            loglik = torch.log( 1 + _eps - torch.exp(-torch.abs(dist[:,0]/gamma_x)**5-torch.abs(dist[:,1]/gamma_y)**5) )
            ux = torch.autograd.grad(loglik, gamma_x,
                            create_graph=True,
                            allow_unused=True,
                            grad_outputs=torch.ones_like(loglik)
                        )[0]
            uy = torch.autograd.grad(loglik, gamma_y,
                            create_graph=True,
                            allow_unused=True,
                            grad_outputs=torch.ones_like(loglik)
                        )[0]

            uux = torch.autograd.grad(ux, gamma_x,
                            create_graph=True,
                            allow_unused=True,
                            grad_outputs=torch.ones_like(ux)
                        )[0]
            uuy = torch.autograd.grad(uy, gamma_y,
                            create_graph=True,
                            allow_unused=True,
                            grad_outputs=torch.ones_like(uy)
                        )[0]
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
    ggd_x = GGDRiskFieldNetwork(input_dim=dataset.shape[1]-2,
                                output_dim=1, 
                              beta=Args.beta,
                              name=Args.model_name,
                              )
    ggd_y = GGDRiskFieldNetwork(input_dim=dataset.shape[1]-2,
                                output_dim=1, 
                              beta=Args.beta,
                              name=Args.model_name,
                              )
    train(
        dataset,
        ggd_x,
        ggd_y,
        num_epochs=Args.num_epochs,
        lr=Args.learning_rate,
        batch_size=Args.batch_size,
        input_dim=dataset.shape[1]-2,
        split_ratio=Args.train_test_split_ratio,
        _eps=Args.loglik_eps,
    )
    
