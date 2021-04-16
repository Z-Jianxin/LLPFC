import torch
import torch.nn as nn

def test_model(model, test_loader, criterion, device):
	# test a model with fully label dataset
	model.eval()
	with torch.no_grad():
		correct = 0
		total = 0
		total_loss = 0
		for images, labels in test_loader:
			images = images.to(device)
			labels = labels.to(device)
			outputs = model(images)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

			prob = nn.functional.softmax(outputs, dim=1)
			loss = criterion(prob, labels, device)
			total_loss += loss.item()
	return correct / total, total_loss / total


def train_model_forward_one_epoch(model, loss_f, optimizer, train_loader, epoch, device, scheduler=None, constr=None, epsilon=1e-6):
	# train the model one epoch with forward correction
	# label input of loss_f must be an integer
	model.train()
	total_step = len(train_loader)
	print(f"Epoch-{epoch} lr: {optimizer.param_groups[0]['lr']}")
	for i, (images, noisy_y, gamma_m) in enumerate(train_loader):
		# Move tensors to the configured device
		images = images.to(device)
		noisy_y = noisy_y.to(device)
		gamma_m = gamma_m.to(device)
		# Forward pass
		outputs = model(images)
		prob = nn.functional.softmax(outputs, dim=1)
		prob_corrected = torch.bmm(gamma_m.float(), prob.reshape(prob.shape[0], -1, 1)).reshape(prob.shape[0], -1)
		loss = loss_f(prob_corrected, noisy_y, device)
		# Backward pass
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		if constr is not None:
			with torch.no_grad():
				for param in model.parameters():
					param.clamp_(*constr)
		if (i + 1) % 100 == 0:
			print('Step [{}/{}], Loss: {:.4f}'.format(i + 1, total_step, loss.item()))

	if scheduler is not None:
		scheduler.step()
