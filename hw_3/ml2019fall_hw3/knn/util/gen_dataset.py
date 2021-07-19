from pathlib import Path
import numpy as np

from extract_image import extract_image

captchas_path = Path('./captchas')
cap_list = captchas_path.glob('*.png')
cap_list = sorted(cap_list)
cap_train_list = cap_list[:25]
cap_test_list = cap_list[25:45]
x_train = []
y_train = []
x_test = []
y_test = []
with open('./labels.txt', 'r') as labels:
	for cap in cap_train_list:
		nums = labels.readline().strip('\n')
		for x in nums:
			y_train.append(int(x))
		x_train.append(extract_image(cap))

	for cap in cap_test_list:
		nums = labels.readline().strip('\n')
		for x in nums:
			y_test.append(int(x))
		x_test.append(extract_image(cap))

y_train = np.array(y_train)
x_train = np.concatenate(x_train, axis=0)
y_test = np.array(y_test)
x_test = np.concatenate(x_test, axis=0)

np.savez('./hack_data.npz', x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
