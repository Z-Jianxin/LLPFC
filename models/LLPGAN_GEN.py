import torch.nn as nn


class LLPGAN_GEN_MNIST(nn.Module):

    def __init__(self, noise_size=100, out_h=28, out_w=28):
        self.out_h, self.out_w = out_h, out_w
        super(LLPGAN_GEN_MNIST, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_size, 500),
            nn.ReLU(),
            nn.BatchNorm1d(500, eps=1e-05, momentum=0.1, ),

            nn.Linear(500, 500),
            nn.ReLU(),
            nn.BatchNorm1d(500, eps=1e-05, momentum=0.1, ),

            nn.Linear(500, self.out_h * self.out_w),
            nn.ReLU(),
            nn.BatchNorm1d(self.out_h * self.out_w, eps=1e-05, momentum=0.1, ),
        )

    def forward(self, noise):
        return self.model(noise).reshape(-1, 1, self.out_h, self.out_w)


class LLPGAN_GEN_COLOR(nn.Module):

    def __init__(self, noise_size=32*32):
        super(LLPGAN_GEN_COLOR, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(noise_size, 4*4*512),
            nn.ReLU(),
            nn.BatchNorm1d(4*4*512, eps=1e-05, momentum=0.1, )
        )
        self.trans_conv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, (5, 5)),
            nn.ReLU(),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, ),

            nn.ConvTranspose2d(256, 128, (5, 5)),
            nn.ReLU(),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, ),

            nn.ConvTranspose2d(128, 3, (5, 5)),
            nn.ReLU(),
            nn.BatchNorm2d(3, eps=1e-05, momentum=0.1, ),
        )

    def forward(self, noise):
        out = self.linear(noise)
        return self.trans_conv(out.reshape((-1, 512, 4, 4)))
