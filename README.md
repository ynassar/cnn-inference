# cnn-inference

To generate the predictions for any image from any caffemodel / prototxt files that involve only Convolution, Max Pooling, Fully Connected, Relu, Tanh, Sigmoid, and/or Softmax layers,

1) Convert the prototxt and caffemodel files to a descriptor file by running python2 ModelConversion/ModelConvertor.py "/path/to/prototxt" "/path/to/caffemodel" "/path/to/outputdescriptor"
2) Convert the mean_img file to our format by running python2 ModelConversion/MeanImageConvertor.py "/path/to/mean_img.binaryproto" "/path/to/out_converted"
3) Build the project and run with arguments "/path/to/outputdescriptor" "/path/to/out_converted" "/path/to/any_image"
