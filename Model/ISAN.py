
# Model/model.py
# Model/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .featureextractor import FIFE, FRFE

class Model(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(Model, self).__init__()
        self.num_classes = num_classes

        self.FRFE = FRFE()
        self.FIFE = FIFE()

        self.main_classifier = GenericClassifier(feature_dim, num_classes)
        self.domain_discriminator = GenericClassifier(feature_dim,  2)
        self.label_invariant_discriminator = GenericClassifier(feature_dim, num_classes)

        self.decoder = Decoder()

        self.local_discriminators = nn.ModuleList([
            nn.Linear(feature_dim, 1) for _ in range(num_classes)
        ])
        self.register_buffer('local_weights', torch.ones(num_classes))

    def forward_shared(self, x):
        return self.shared_encoder(x)

    def forward_private_source(self, x):
        return self.private_source(x)

    def forward_local_discriminator(self, x, c):
        x = x.mean(dim=-1)
        return self.local_discriminators[c](x)

class GenericClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GenericClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        x = x.mean(dim=-1)   # Global Average Pooling (N, C, H, W) -> (N, C)
        x = self.fc(x)
        return x


#decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        #  (batch_size, 320, 64)
        self.conv1 = nn.Sequential(
            nn.Conv1d(640, 640, kernel_size=3, padding=1, groups=320),  # Depthwise
            nn.Conv1d(640, 256, kernel_size=1),  # Pointwise
            nn.LayerNorm([256, 64])  #  LayerNorm
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding=1, groups=256),  # Depthwise
            nn.Conv1d(256, 128, kernel_size=1),  # Pointwise
            nn.LayerNorm([128, 64])  # LayerNorm
        )

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([64, 128])
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([32, 256])
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([16, 512])
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose1d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([8, 1024])
        )
        self.deconv5 = nn.ConvTranspose1d(8, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        #  (batch_size, 320, 64)
        x = F.relu(self.conv1(x))  # -> (batch_size, 256, 64)
        x = F.relu(self.conv2(x))  # -> (batch_size, 128, 64)

        x = F.relu(self.deconv1(x))  # -> (batch_size, 64, 128)
        x = F.relu(self.deconv2(x))  # -> (batch_size, 32, 256)
        x = F.relu(self.deconv3(x))  # -> (batch_size, 16, 512)
        x = F.relu(self.deconv4(x))  # -> (batch_size, 8, 1024)
        x = self.deconv5(x)  # -> (batch_size, 1, 2048)

        return x