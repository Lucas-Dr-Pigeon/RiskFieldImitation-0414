import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import os

class GGDRiskFieldNetwork(nn.Module):
    def __init__(self, input_dim, output_dim=2, beta=5, fc1_dim=256, fc2_dim=256, fc3_dim=256, chkpt_dir='./model/temp', name='ggd'):
        # input dimensions: ego_vx, ego_vy, foe_vx, foe_vy, ego_ax, ego_ay, foe_ax, foe_ay, 
        super(GGDRiskFieldNetwork, self).__init__()
        self.input_dim = input_dim
        self.beta = beta
        self.fc1_dim = fc1_dim
        self.fc2_dim = fc2_dim
        self.fc3_dim = fc3_dim
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(chkpt_dir, f"/{name}_chkpt")
        self.fc1 = nn.Linear(self.input_dim, self.fc1_dim)
        self.fc2 = nn.Linear(self.fc1_dim, self.fc2_dim)
        self.fc3 = nn.Linear(self.fc2_dim, self.fc3_dim)
        self.fc4 = nn.Linear(self.fc3_dim, output_dim)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, input):
        x = self.fc1(input)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        return x**2 + 1

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

