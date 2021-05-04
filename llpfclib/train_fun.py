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


def validate_model_forward(model, loss_f_val, val_loader, device):
	model.eval()
	total_loss = 0
	total = 0
	for i, (images, noisy_y, trans_m, weights) in enumerate(val_loader):
		total_loss += compute_forward_loss_on_minibatch(model, loss_f_val, images, noisy_y, trans_m, weights, device).item()
		total += noisy_y.size(0)
	return total_loss / total


def train_model_forward_one_epoch(model, loss_f, optimizer, train_loader, device, epoch, scheduler, logger):
	# train the model one epoch with forward correction
	# label input of loss_f must be an integer
	model.train()
	total_step = len(train_loader)
	for i, (images, noisy_y, trans_m, weights) in enumerate(train_loader):
		loss = compute_forward_loss_on_minibatch(model, loss_f, images, noisy_y, trans_m, weights, device)
		# Backward pass
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		if (i + 1) % 100 == 0:
			logger.info('				Step [{}/{}], Loss: {:.4f}'.format(i + 1, total_step, loss.item()))
		if type(scheduler) == torch.optim.lr_scheduler.CosineAnnealingWarmRestarts:
			scheduler.step(epoch + i / total_step)
	if type(scheduler) == torch.optim.lr_scheduler.MultiStepLR:
		scheduler.step()
	elif type(scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
		scheduler.step(validate_model_forward(model, loss_f, train_loader, device))


def compute_forward_loss_on_minibatch(model, loss_f, images, noisy_y, trans_m, weights, device):
	# Move tensors to the configured device
	images = images.to(device)
	noisy_y = noisy_y.to(device)
	trans_m = trans_m.to(device)
	weights = weights.to(device)
	# Forward pass
	outputs = model(images)
	prob = nn.functional.softmax(outputs, dim=1)
	prob_corrected = torch.bmm(trans_m.float(), prob.reshape(prob.shape[0], -1, 1)).reshape(prob.shape[0], -1)
	loss = loss_f(prob_corrected, noisy_y, weights, device)
	return loss
