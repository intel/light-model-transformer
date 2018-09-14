## To Run the Test Script
1) Some prework is needed: 
    - download tensorflow source code: 'git clone https://github.com/tensorflow/tensorflow.git'
    - download tensorflow model: 'git clone https://github.com/tensorflow/models.git'
    - install tensorflow (either from source code or by pip)
2) Configure test.cfg to make sure the path is correct: 
    - 'tensorflow' points to the path of tensorflow source code 
    - 'tensorflow_slim' points to the 'research/slim' directory of the model
3) run test.py: python test.py

## Output Example
```
[root@SKL tests]# python test.py
 Test model: inception_v1 start ............
INFO:root:bazel build ...
INFO:root:summarize graph ...
INFO:root:tensorflow do inference!
INFO:root:convert tf pb file to topo
INFO:root:convert topo to inference code
INFO:root:build and run inference code
             inception_v1 passed! tensorflow used time: ['157.08'] ms, mkldnn used time: ['7.30'] ms.
 Test model: inception_v2 start ............
INFO:root:bazel build ...
INFO:root:summarize graph ...
INFO:root:tensorflow do inference!
INFO:root:convert tf pb file to topo
INFO:root:convert topo to inference code
INFO:root:build and run inference code
             inception_v2 passed! tensorflow used time: ['207.59'] ms, mkldnn used time: ['12.60'] ms.
 Test model: inception_v3 start ............
INFO:root:bazel build ...
INFO:root:summarize graph ...
INFO:root:tensorflow do inference!
INFO:root:convert tf pb file to topo
INFO:root:convert topo to inference code
INFO:root:build and run inference code
             inception_v3 passed! tensorflow used time: ['239.14'] ms, mkldnn used time: ['12.40'] ms.
 Test model: inception_v4 start ............
INFO:root:bazel build ...
INFO:root:summarize graph ...
INFO:root:tensorflow do inference!
INFO:root:convert tf pb file to topo
INFO:root:convert topo to inference code
INFO:root:build and run inference code
             inception_v4 passed! tensorflow used time: ['601.18'] ms, mkldnn used time: ['40.50'] ms.
 Test model: resnet_v1_50 start ............
INFO:root:bazel build ...
INFO:root:summarize graph ...
INFO:root:tensorflow do inference!
INFO:root:convert tf pb file to topo
INFO:root:convert topo to inference code
INFO:root:build and run inference code
             resnet_v1_50 passed! tensorflow used time: ['242.51'] ms, mkldnn used time: ['11.10'] ms.
 Test model: resnet_v2_50 start ............
INFO:root:bazel build ...
INFO:root:summarize graph ...
INFO:root:tensorflow do inference!
INFO:root:convert tf pb file to topo
INFO:root:convert topo to inference code
INFO:root:build and run inference code
             resnet_v2_50 passed! tensorflow used time: ['228.04'] ms, mkldnn used time: ['12.30'] ms.
 Test model: vgg_16 start ............
INFO:root:bazel build ...
INFO:root:summarize graph ...
INFO:root:tensorflow do inference!
INFO:root:convert tf pb file to topo
INFO:root:convert topo to inference code
INFO:root:build and run inference code
             vgg_16 passed! tensorflow used time: ['442.17'] ms, mkldnn used time: ['21.50'] ms.
 All tests done!
```
