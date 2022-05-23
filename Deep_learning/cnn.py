import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 输入 1-channel, 输出 32-channel
        # 5*5*32, stride = 1
        self.cov1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size = 5, stride = 1),
            nn.MaxPool2d(2, padding= 0),
            nn.ReLU(),
        )
        '''
        # 5*5*32, stride = 1
        self.cov2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size = 5, stride = 1),
            nn.MaxPool2d(2, padding= 0),
            nn.ReLU()
        )
        '''
        
        # 5*5*64
        self.cov3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1),
            nn.MaxPool2d(2, padding= 0),
            nn.ReLU()
        )
        
        # Maxpooling 
        #self.max_pooling = nn.MaxPool2d(2, padding= 0)

        # fc layer
        self.fc1 = nn.Sequential(
            nn.Linear(4*4*64, 10),
            nn.LogSoftmax(dim=1)
        )
    
    def forward(self, input):
        
        batch_size = input.size(0)
        cov1 = self.cov1(input)
        cov3 = self.cov3(cov1)
        cov3 = cov3.view(batch_size, -1)
        
        return self.fc1(cov3)

