
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt

def test1():
	img_in = Image.open("input_image.jpg")
	np_in = np.array(img_in, dtype="float")
	np_in /= 256
	t_1 = torch.from_numpy(np_in)

	t_rand = torch.rand(t_1.shape)
	t_1 *= t_rand

	plt.imshow(t_1)
	plt.show()

	np_out = np.array(t_1)
	np_out *= 256
	img_out = Image.fromarray(np_out.astype("uint8"))
	img_out.save("output_image.jpg")

def load_folder(path):
	result = np.empty(0, dtype="float")
	for _ in []:  # fixme
		img_in = Image.open("input_image.jpg")
		np_in = np.array(img_in, dtype="float")
		np_in /= 256
		result += np_in
	return torch.from_numpy(np_in)


def test2():
	torch.utils.data.DataLoader()
	print("moo")

if __name__ == "__main__":
	test1()
