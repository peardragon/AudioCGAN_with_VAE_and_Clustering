import torch
import torch.nn as nn

class Discriminator(nn.Module):
    r""" It is mainly based on the mobile net network as the backbone network discriminator.
    Args:
        input_size (int): The size of the 1D array. (Default: 200)
        channels (int): The channels of the image. (Default: 1)
        num_classes (int): Number of classes for dataset. (Default: 10)
    """
    """
    GAN model architecture from the `"One weird trick..." <https://arxiv.org/abs/1406.2661>` paper.
    """
    """
    Find Input is True or False
    """

    def __init__(self, input_size: int = 200, channels: int = 1, num_classes: int = 10) -> None:
        super(Discriminator, self).__init__()
        
        # Embedding : (*) -> (*, H) (H: Embeddin_dim)
        self.label_embedding = nn.Embedding(num_embeddings=num_classes, embedding_dim=num_classes)

        self.main = nn.Sequential(
            nn.Linear(channels * input_size + num_classes, 256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            nn.Linear(256, 256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            nn.Linear(256, 128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Initializing all neural network weights.
        self._initialize_weights()

    def forward(self, inputs: torch.Tensor, labels: torch.IntTensor) -> torch.Tensor:
        r""" Defines the computation performed at every call.
        Args:
            inputs (tensor): input tensor into the calculation.
            labels (list):  input tensor label.
        Returns:
            A four-dimensional vector (N*C*L).
        """
        inputs = torch.flatten(inputs, 1)
        conditional = self.label_embedding(labels)
        conditional_inputs = torch.cat([inputs, conditional], dim=-1)
        out = self.main(conditional_inputs)

        return out
    
    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    

class Generator(nn.Module):
    r""" It is mainly based on the mobile net network as the backbone network generator.
    Args:
        image_size (int): The size of the image. (Default: 200)
        channels (int): The channels of the image. (Default: 1)
        num_classes (int): Number of classes for dataset. (Default: 10)
    """

    def __init__(self, input_size: int =200, channels: int = 1, num_classes: int = 10) -> None:
        super(Generator, self).__init__()
        self.input_size = input_size
        self.channels = channels

        self.label_embedding = nn.Embedding(num_classes, num_classes)

        self.main = nn.Sequential(
            nn.Linear(64 + num_classes, 128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Linear(256, channels * input_size),
            nn.Tanh()
        )

        # Initializing all neural network weights.
        self._initialize_weights()

    def forward(self, inputs: torch.Tensor, labels: torch.IntTensor) -> torch.Tensor:
        """
        Args:
            inputs (tensor): input tensor into the calculation.
            labels (list):  input tensor label.
        Returns:
            A four-dimensional vector (N*C*L).
        """
        condition = self.label_embedding(labels)
        conditional_inputs = torch.cat([inputs, condition], dim=-1)
        out = self.main(conditional_inputs)
        out = out.reshape(out.size(0), self.channels, self.input_size)

        return out
    
    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)