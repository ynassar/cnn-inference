import cv2
import sys
import numpy as np


in_file = sys.argv[1]
out_file = sys.argv[2]
mean_img_file = sys.argv[3]
out_channels = int(sys.argv[3])
out_height = int(sys.argv[4])
out_width = int(sys.argv[5])

img = cv2.imread(in_file)
if out_channels == 1:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img = cv2.resize(img, (out_height, out_width))
print img.shape
if out_channels == 3:
    img = img.reshape((img.shape[2], img.shape[0], img.shape[1]))
file = open(out_file, 'w')
file.write(' '.join([str(out_channels), str(out_height), str(out_width)]))
file.write('\n')
img = img.flatten()
img_list = [str(x) for x in img]
file.write((' ').join(img_list))

