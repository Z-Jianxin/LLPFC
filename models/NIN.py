# code in this file is modified from https://github.com/yangqiongyongyu/Network-In-Network-Pytorch/blob/master/models/nin.py
# hyperparameters selected based on https://worksheets.codalab.org/worksheets/0x7b8f6fbc6b5c49c18ac7ca94aafaa1a7
import torch.nn as nn
import math


class NIN(nn.Module):
	def __init__(self, num_classes, image_size, in_channel=3):
		super(NIN, self).__init__()
		self.num_classes = num_classes
		if isinstance(image_size, int):
			pool_size = math.ceil(math.ceil(image_size/2.0-0.5)/2.0-0.5)
		else:
			pool_size = (math.ceil(math.ceil(image_size[0]/2.0-0.5)/2.0-0.5), math.ceil(math.ceil(image_size[1]/2.0-0.5)/2.0-0.5))
		self.features = nn.Sequential(
			nn.Conv2d(in_channel, 192, 5, padding=2),
			nn.ReLU(inplace=True),
			nn.Conv2d(192, 160, 1),
			nn.ReLU(inplace=True),
			nn.Conv2d(160, 96, 1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(3, stride=2, ceil_mode=True),
			nn.Dropout(inplace=True),

			nn.Conv2d(96, 192, 5, padding=2),
			nn.ReLU(inplace=True),
			nn.Conv2d(192, 192, 1),
			nn.ReLU(inplace=True),
			nn.Conv2d(192, 192, 1),
			nn.ReLU(inplace=True),
			nn.AvgPool2d(3, stride=2, ceil_mode=True),
			nn.Dropout(inplace=True),

			nn.Conv2d(192, 192, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(192, 192, 1),
			nn.ReLU(inplace=True),
			nn.Conv2d(192, self.num_classes, 1),
			nn.ReLU(inplace=True),
			nn.AvgPool2d(pool_size, stride=1)
		)
		self._initialize_weights()

	def forward(self, x):
		x = self.features(x)
		x = x.view(x.size(0), self.num_classes)
		return x

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				m.weight.data.normal_(0, 0.05)
				if m.bias is not None:
					m.bias.data.zero_()