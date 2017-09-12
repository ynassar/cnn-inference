import cv2
import sys
import numpy as np
import caffe
from caffe.proto import caffe_pb2


in_file = sys.argv[1]
out_file = sys.argv[2]
mean_img_file = sys.argv[3]
out_channels = int(sys.argv[4])
out_height = int(sys.argv[5])
out_width = int(sys.argv[6])

mean_img_blob = caffe.proto.caffe_pb2.BlobProto()
data = open( mean_img_file , 'rb' ).read()
mean_img_blob.ParseFromString(data)
mean_img = caffe.io.blobproto_to_array(mean_img_blob)[0]

img = cv2.imread(in_file).astype('float32')
if out_channels == 1:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_img = mean_img[0]
img = cv2.resize(img, (out_height, out_width))
img -= mean_img
print img.shape
if out_channels == 3:
    img = img.reshape((img.shape[2], img.shape[0], img.shape[1]))
file = open(out_file, 'w')
file.write(' '.join([str(out_channels), str(out_height), str(out_width)]))
file.write('\n')
img = img.flatten()
img_list = [str(x) for x in img]
file.write((' ').join(img_list))

