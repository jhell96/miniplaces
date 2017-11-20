import shutil
import os

dirs = [x[0] for x in os.walk('../data/images/train/') ][1:]

# for i in range(len(dirs)):
# 	os.makedirs('../data/images/val/' + dirs[i].split('/')[-1])

# f = open('../data/val.txt')
# for line in f:
# 	img, index = line.split(' ')[0], line.split(' ')[1]
# 	src = '../data/images/val/' + img.split('/')[-1].strip()
# 	dest = '../data/images/val/' + dirs[int(index)].split('/')[-1].strip() + '/' + img.split('/')[-1].strip()

# 	shutil.move(src, dest)

# mapping = [0] * 100
# f = open("../data/categories.txt", "r")
# for line in f:
# 	result = [x.strip() for x in line.split(' ')]
# 	category, i = result[0], int(result[1])
# 	mapping[i] = category

# print(mapping)