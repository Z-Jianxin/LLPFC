import torch.nn as nn


class LPPGAN_DIS(nn.Module):
	# use the same discriminator as LLP-GAN paper
	def __init__(self, num_class, image_size, in_channel=3):
		super(LPPGAN_DIS, self).__init__()
		self.conv_layers = nn.Sequential(
			nn.Dropout(p=0.2, ),
			nn.Conv2d(in_channel, 64, 3, padding=1, stride=1),
			nn.ReLU(),
			nn.Conv2d(64, 64, 3, padding=1, stride=1),
			nn.ReLU(),
			nn.Conv2d(64, 64, 3, padding=1, stride=2),
			nn.ReLU(),
			nn.Dropout(p=0.5, ),
			nn.Conv2d(64, 128, 3, padding=1, stride=1),
			nn.ReLU(),
			nn.Conv2d(128, 128, 3, padding=1, stride=1),
			nn.ReLU(),
			nn.Conv2d(128, 128, 3, padding=1, stride=2),
			nn.ReLU(),
			nn.Dropout(p=0.5, ),
			nn.Conv2d(128, 256, 3, padding=1, stride=1),
			nn.ReLU(),
			nn.Conv2d(256, 128, 1, padding=0, stride=1),
			nn.ReLU(),
			nn.Conv2d(128, 64, 1, padding=0, stride=1),
			nn.ReLU(),
		)
		if isinstance(image_size, int):
			pool_size = round(round(image_size/2.0)/2.0)
		else:
			pool_size = (round(round(image_size[0]/2.0)/2.0), round(round(image_size[1]/2.0)/2.0))
		self.pool_layer = nn.AvgPool2d(pool_size, stride=pool_size, )
		self.fc_layer = nn.Linear(64, num_class, bias=True)

	def forward(self, x):
		out = self.conv_layers(x)
		out = self.pool_layer(out)
		out = self.fc_layer(out.reshape(-1, 64))
		return out