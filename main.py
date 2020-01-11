import random

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import os

width, height = 128, 72
# size = (1280, 720)

def image_to_tensor(img):
	arr = np.array(img, dtype="float")
	arr /= 256
	return torch.from_numpy(arr)

def tensor_to_image(tsr):
	arr = np.array(tsr.cpu())
	arr *= 256
	return Image.fromarray(arr.astype("uint8"))

def test1():
	img_in = Image.open("input_image.jpg")
	t_1 = image_to_tensor(img_in)

	t_rand = torch.rand(t_1.shape)
	t_1 *= t_rand

	plt.imshow(t_1)
	plt.show()

	img_out = tensor_to_image(t_1)
	img_out.save("output_image.jpg")

def load_folder(path, split = 0.9):
	print("Loading from: ", path)
	train = []
	test = []
	for file_name in os.listdir(path):
		file_path = path + '\\' + file_name
		file_pre_name, file_ext = os.path.splitext(file_path)
		if file_ext == ".jpg":
			img = Image.open(file_path).resize((width, height))
			tsr = image_to_tensor(img)
			if random.random() < split:
				train += [tsr]
			else:
				test += [tsr]
			print("Loaded: ", file_name)
	train = torch.stack(train)
	train = torch.stack((train, train))
	# test = torch.stack(test)
	# test = torch.stack((test, test))
	return train, test

class Net(nn.Module):
	def __init__(self):
		super().__init__()
		thk = 128
		# self.fc1 = nn.Linear((width * height * 3), thk)
		self.conv1 = nn.Conv2d(1, 32, 5)
		self.conv1 = nn.Conv2d(32, 64, 5)
		self.conv3 = nn.Conv2d(64, 128, 5)
		# self.fc2 = nn.Linear(thk, thk)
		# self.fc3 = nn.Linear(thk, thk)
		# self.fc4 = nn.Linear(thk, thk)
		# self.fc5 = nn.Linear(thk, (width * height * 3))

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = F.relu(self.fc4(x))
		x = self.fc5(x)
		return x


def test2():
	train, test = load_folder("C:\\Users\\tifa-\\Pictures\\Camera Roll", 1)

	if torch.cuda.is_available():
		device = torch.device("cuda")  # a CUDA device object
	else:
		device = torch.device("cpu")

	train_set = torch.utils.data.DataLoader(train, batch_size=5, shuffle=True)
	# test_set = torch.utils.data.DataLoader(test, batch_size=10, shuffle=False)

	net = Net().to(device)
	lr = 0.01
	EPOCHS = 300


	print("Begining training")
	for epoch in range(EPOCHS):
		lr *= 0.99
		optimizer = optim.Adam(net.parameters(), lr=lr)
		for data in train_set:
			x, y = data
			x = x.to(device, torch.float)
			y = y.to(device, torch.float)
			net.zero_grad()
			output = net(x.view(-1, (width * height * 3)))  # flatten input and predict using net
			# loss = F.nll_loss(output, y)  # used for scalar outputs
			loss = F.mse_loss(output, y.view(-1, (width * height * 3)))  # used for vector outputs
			loss.backward()  # apply backpropagation
			optimizer.step()  # apply adjustments

		if epoch % 10 == 0:
			with torch.no_grad():
				print("LR: ", lr, "\tLoss: ", loss)
				plt.imshow(output.view(-1, height, width, 3)[0].cpu())
				plt.show()

	with torch.no_grad():
		img_in = Image.open("input_image.jpg").resize((width, height))
		t_1 = image_to_tensor(img_in).view(1, (width * height * 3)).to(device, torch.float)
		t_2 = net(t_1)

		plt.imshow(t_1.view(height, width, 3).cpu())
		plt.show()
		plt.imshow(t_2.view(height, width, 3).cpu())
		plt.show()

		img_out = tensor_to_image(t_2)
		img_out.save("output_image.jpg")


def test3():
	x = torch.from_numpy(np.array(range(1000), dtype=float))
	x -= 500
	x /= 100
	y = F.selu(x)

	plt.plot(np.array(x), np.array(y))
	plt.show()

if __name__ == "__main__":
	# test1()
	test2()
	# test3()
