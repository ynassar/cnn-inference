import cv2
import sys
import numpy as np
import caffe
from caffe.proto import caffe_pb2

mean_img_file = sys.argv[1]
out_file = sys.argv[2]

mean_img_blob = caffe.proto.caffe_pb2.BlobProto()
data = open( mean_img_file , 'rb' ).read()
mean_img_blob.ParseFromString(data)
mean_img = caffe.io.blobproto_to_array(mean_img_blob)[0]

mean_img_ch = mean_img.shape[0]
mean_img_height = mean_img.shape[1]
mean_img_width = mean_img.shape[2]

mean_img = mean_img.flatten()
mean_img.fill(mean_img.mean())
mean_img_list = [str(x) for x in mean_img]

file_stream = open(out_file, 'w')
file_stream.write(' '.join([str(mean_img_ch), str(mean_img_height), str(mean_img_width)] + mean_img_list))
file_stream.close()
