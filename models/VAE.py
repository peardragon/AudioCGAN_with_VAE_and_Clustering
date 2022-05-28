import torch
import torch.nn as nn

class Encoder_Linear(nn.Module) :
    """
    Encoder using linear layer after convolution
    """
    def __init__(self) :
        super(Encoder_Linear, self).__init__()
        self.conv1d = nn.Sequential(
            nn.Conv1d(2, 1, kernel_size=7, stride=5, dilation=2, padding=2),
            nn.ReLU()
        )
        self.Dense = nn.Sequential(
            nn.Linear(52919, 200)
        )
        
    def forward(self, x) :
        # x : 2 1D arrays (stereo channel)
        hidden = self.conv1d(x)
        y = self.Dense(hidden)
        return y, hidden
    

class Decoder_Linear(nn.Module) : 
    """
    Decoder using linear layer before Upsampling
    """
    def __init__(self) :
        super(Decoder_Linear, self).__init__()
        
        self.upsample = nn.Sequential(
            nn.Upsample((264600), mode='nearest'),
            nn.Conv1d(1, 2, kernel_size=3, stride=1, padding=1)
        )
        self.Dense = nn.Sequential(
            nn.Linear(200, 52919)
        )
        self.iden = nn.Identity()
        
    def forward(self, x, hidden) :
        x = self.Dense(x)
        y = self.upsample(x + self.iden(hidden))
        return y