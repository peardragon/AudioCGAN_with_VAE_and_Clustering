import torch
import torch.nn as nn
import copy
import numpy as np

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
                    
    def load(self, backup):
        for m_from, m_to in zip(backup.modules(), self.modules()):
            if isinstance(m_to, nn.Linear):
                m_to.weight.data = m_from.weight.data.clone()
                if m_to.bias is not None:
                    m_to.bias.data = m_from.bias.data.clone()
                    

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
                    

class GANLoops():
    def __init__(self, dataset, class_num, device = 'cuda'):
        self.dataset = dataset
        # dataset["X"] = shape : # batches ,batch_size, 1, 200
        # dataset["Y"] = shape : # batches, batch_size
        self.batch_size = dataset[0].shape[1]
        self.batch_num = dataset[1].shape[0]
        self.noise_size = 64
        self.device = device
        self.class_num = class_num
    
    def d_loop(self, G, D, d_optimizer, criterion):
        
        noise_size = self.noise_size
        batch_size = self.batch_size
        device = self.device
        dataset = self.dataset
        class_num = self.class_num
        
        # 1. Train D on real+fake
        d_optimizer.zero_grad()

        #  1A: Train D on real
        random_batch_num = np.random.randint(self.batch_num)
        
        real = torch.Tensor(dataset[0][random_batch_num]).to(device)
        real_class = torch.IntTensor(dataset[1][random_batch_num]).to(device)
        real_label = torch.ones(batch_size, 1).to(device)

        d_real_decision = D(real, real_class)
        d_real_error = criterion(d_real_decision, real_label)  # ones = true

        #  1B: Train D on fake
        noise = torch.randn([batch_size, noise_size]).to(device)
        fake_class =  torch.randint(0, class_num, (batch_size,)).to(device)
        fake_label = torch.zeros(batch_size, 1).to(device)
        
        with torch.no_grad():
            fake = G(noise, fake_class)
        d_fake_decision = D(fake, fake_class)
        d_fake_error = criterion(d_fake_decision, fake_label)  # zeros = fake

        d_loss = d_real_error + d_fake_error
        
        D_G_z = d_fake_decision.detach().cpu().mean().item()
        D_x = d_real_decision.detach().cpu().mean().item()
        

        d_loss.backward()
        d_optimizer.step()  # Only optimizes D's parameters; changes based on stored gradients from backward()

        return d_real_error.cpu().item(), d_fake_error.cpu().item(), D_G_z, D_x
    
    def d_unrolled_loop(self, G, D, d_optimizer, criterion):
        
        noise_size = self.noise_size
        batch_size = self.batch_size
        device = self.device
        dataset = self.dataset
        class_num = self.class_num
        
        # 1. Train D on real+fake
        d_optimizer.zero_grad()

        #  1A: Train D on real
        random_batch_num = np.random.randint(self.batch_num)
        
        real = torch.Tensor(dataset[0][random_batch_num]).to(device)
        real_class = torch.IntTensor(dataset[1][random_batch_num]).to(device)
        real_label = torch.ones(batch_size, 1).to(device)
        
        d_real_decision = D(real, real_class)
        d_real_error = criterion(d_real_decision, real_label)  # ones = true

        #  1B: Train D on fake
        noise = torch.randn([batch_size, noise_size]).to(device)
        fake_class =  torch.randint(0, class_num, (batch_size,)).to(device)
        fake_label = torch.zeros(batch_size, 1).to(device)
        
        with torch.no_grad():
            fake = G(noise, fake_class)
        d_fake_decision = D(fake, fake_class)
        d_fake_error = criterion(d_fake_decision, fake_label)  # zeros = fake

        d_loss = d_real_error + d_fake_error
        
        D_G_z = d_fake_decision.detach().cpu().mean().item()
        D_x = d_real_decision.detach().cpu().mean().item()

        
        d_loss.backward(create_graph=True)
        # Only optimizes D's parameters; changes based on stored gradients from backward()
        d_optimizer.step()  

        return d_real_error.cpu().item(), d_fake_error.cpu().item(), D_G_z, D_x
    
    def g_loop(self, G, D, g_optimizer, d_optimizer, criterion, unrolled_steps):
        
        noise_size = self.noise_size
        batch_size = self.batch_size
        device = self.device
        class_num = self.class_num
        
        # 2. Train G on D's response (but DO NOT train D on these labels)
        g_optimizer.zero_grad()
        d_optimizer.zero_grad()

        noise = torch.randn([batch_size, noise_size]).to(device)
        fake_class =  torch.randint(0, class_num, (batch_size,)).to(device)

        if unrolled_steps > 0:
            backup = copy.deepcopy(D)
            for i in range(unrolled_steps):
                self.d_unrolled_loop(G, D, d_optimizer, criterion)

            fake = G(noise, fake_class)
            fake_decision = D(fake, fake_class)
            real_label = torch.ones(batch_size, 1).to(device)
            
            g_error = criterion(fake_decision, real_label)  # we want to fool, so pretend it's all genuine
            g_error.backward()
            
            g_optimizer.step() # Only optimizes G's parameters
            
            D.load(backup)
            del backup

        else:
            fake = G(noise, fake_class)
            fake_decision = D(fake, fake_class)
            real_label = torch.ones(batch_size, 1).to(device)
            
            g_error = criterion(fake_decision, real_label)  # we want to fool, so pretend it's all genuine
            g_error.backward()
            g_optimizer.step()  # Only optimizes G's parameters

        return g_error.cpu().item()
        