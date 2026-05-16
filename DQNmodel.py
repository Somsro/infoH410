import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ConvBlock, self).__init__()
        d = output_dim // 4
        # The model looks at the input with 4 different convolutional filters of sizes 1, 2, 3 and 4. Each filter produces d output channels.
        self.conv1 = nn.Conv2d(input_dim, d, 1, padding='same')
        self.conv2 = nn.Conv2d(input_dim, d, 2, padding='same')
        self.conv3 = nn.Conv2d(input_dim, d, 3, padding='same')
        self.conv4 = nn.Conv2d(input_dim, d, 4, padding='same')

    def forward(self, x):
        output1 = F.relu(self.conv1(x))
        output2 = F.relu(self.conv2(x))
        output3 = F.relu(self.conv3(x))
        output4 = F.relu(self.conv4(x))
        # We stack the outputs of the 4 convolutional filters along the channel dimension to get a tensor of shape (batch_size, output_dim, height, width).
        return torch.cat((output1, output2, output3, output4), dim=1)

class DQN(nn.Module):
    def __init__(self, n_actions=4):
        super(DQN, self).__init__()
        
        # 16 channels en entrée (ton one-hot encoding)
        self.conv1 = ConvBlock(16, 512)
        self.conv2 = ConvBlock(512, 512)
        
        # Le Flatten transforme les 512 filtres de taille 4x4 en une ligne de 8192 neurones
        self.dense1 = nn.Linear(8192, 512)
        self.dense2 = nn.Linear(512, n_actions)
    
    # How the model processes the input state to produce Q-values for each action
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = nn.Flatten()(x)
        x = F.relu(self.dense1(x))
        return self.dense2(x)