# Light Model Transformer

Light Model Transformer is a light tool that could transform trained tensorflow model into C++ code. The generated code is based on Intel MKLDNN and provides inferface to do inference without any framework, intent to accelerate the inference performance.

## Usage:

### Install Intel mkl-dnn
- install Intel mkl-dnn via script
    - `python install.py`
    - See detail by cmd: `python install.py -h`

### Convert model

- make sure you have tensorflow installed. (can check it by `import tensorflow` in python)

- prepare tensorflow model in pb format. (need to freeze one if you do not have it, and here assume you have one named frozen.pb)

- run the scripts, `eg:` 

  ```python
  # Transform the model to internal representation
  python tf2topo.py --input_model_filename=./frozen.pb \
                --weights_file=saved_model/weights.bin \
                --pkl_file=saved_model/weights.pkl \
                --topo_file=saved_model/topo.txt

  # Transform to C++ inference code which is based on MKLDNN
  python topo2code.py --topo=saved_model/topo.txt
  ```

### Compile and test generated inference code

- compile and test generated code `as below:`
    - `cd inference_code`
    - `vi build.sh` and make sure the path of MKLDNN_ROOT is correct
    - `sh build.sh` (Note: opencv is needed to compile the code, and it will create an executable file named 'test')
    - `./test -W saved_model/weights.bin -b 1 -l 100` (Type `./test -H` for help)

### Integrate generated code to your own project

- Please look into inference_code/Main.cpp for how to use the generated code. In general, the inferface looks like:

  ```C++
  // Create a Model object
  Model model(weight_path, batch_size);
  
  // Do inference by providing input and return the output
  output = model.inference(input);
  ```

## Note:

- 'Light' means it is a simple implementation, currently only support CNN networks. And even for CNN, many ops are still not supported.
- You may also consider [OpenVINO(TM) toolkit](https://software.intel.com/en-us/openvino-toolkit) for inference acceleration.
