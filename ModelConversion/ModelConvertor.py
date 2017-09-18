import caffe
import numpy as np
import sys

from caffe.proto import caffe_pb2
from google.protobuf.text_format import Merge

prototxt_file = sys.argv[1]
caffemodel_file = sys.argv[2]
out_modeldescriptor = sys.argv[3]

prototxt_file_contents = open(prototxt_file, 'r').read()

net_param = caffe_pb2.NetParameter()
Merge(prototxt_file_contents, net_param)

input_n = net_param.input_shape[0].dim[0]
input_channels = net_param.input_shape[0].dim[1]
input_height = net_param.input_shape[0].dim[2]
input_width = net_param.input_shape[0].dim[3]

out_descriptorfile = open(out_modeldescriptor, 'w')

out_descriptorfile.write(' '.join([str(input_n), str(input_channels), str(input_height), str(input_width)]) + '\n')

net = caffe.Net(prototxt_file, caffemodel_file, caffe.TEST)
print net_param.layer
for layer in net_param.layer:
    if layer.type == 'Convolution':
        num_channels_out = layer.convolution_param.num_output
        filter_size = layer.convolution_param.kernel_size
        stride = layer.convolution_param.stride[0]
        try:
            padding = layer.convolution_param.pad[0]
        except:
            padding = 0
        weights = net.params[layer.name][0].data
        bias = net.params[layer.name][1].data
        num_channels_in = weights.shape[1]
        num_channels_out = weights.shape[0]
        filter_height = weights.shape[2]
        filter_width = weights.shape[3]
        bias_size = bias.shape[0]
        weights_list = []
        bias_list = []
        for i in range(num_channels_out):
            for j in range(num_channels_in):
                for k in range(filter_height):
                    for l in range(filter_width):
                        weights_list.append(str(weights[i][j][k][l]))
        print bias
        for i in range(bias_size):
            bias_list.append(str(bias[i]))
        out_descriptorfile.write(' '.join(['Convolution', layer.name, str(stride), str(padding), str(num_channels_out), str(num_channels_in), str(filter_height), str(filter_width)] + weights_list + [str(bias_size)] + bias_list) + '\n')
    elif layer.type == 'ReLU':
        out_descriptorfile.write(' '.join(['ReLU', layer.name]) + '\n')
    elif layer.type == 'Pooling':
        out_descriptorfile.write(' '.join(['Pooling', layer.name, str(layer.pooling_param.kernel_size), str(layer.pooling_param.stride)]) + '\n')
    elif layer.type == 'InnerProduct':
        weights = net.params[layer.name][0].data
        weights = weights.T
        bias = net.params[layer.name][1].data
        input_size = weights.shape[0]
        output_size = weights.shape[1]
        bias_list = []
        weights_list = []
        for i in range(input_size):
            for j in range(output_size):
                weights_list.append(str(weights[i][j]))
        for i in range(output_size):
            bias_list.append(str(bias[i]))
	print [str(output_size)]
        out_descriptorfile.write(' '.join(['InnerProduct', layer.name, str(input_size), str(output_size)] + weights_list + [str(output_size)] + bias_list) + '\n')
    elif layer.type == 'Softmax':
        out_descriptorfile.write(' '.join(['Softmax', layer.name]) + '\n')
    elif layer.type == 'TanH':
        out_descriptorfile.write(' '.join(['TanH', layer.name]) + '\n')
    elif layer.type == 'Sigmoid':
        out_descriptorfile.write(' '.join(['Sigmoid', layer.name]) + '\n')

out_descriptorfile.close()
