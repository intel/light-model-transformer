# TensorFlow model modification tools

## Requirements

Build requirements:
- protoc: to generate Python classes from .proto files \
    ```bash
    $ apt install -y protobuf-compiler libprotobuf-dev
    ```
- mypy-protobuf (optional): to generate .pyi stubs from .proto files \
    ```bash
    $ pip install mypy-protobuf
    ```
- tensorflow: contains additional required .proto definitions \
    ```bash
    $ pip install tensorflow
    ```

Runtime requirements:
- tensorflow

## Protobuf compilation

To compile the protobuf files:

```sh
$ cd <repo_root>/python/proto
$ ./compile_proto.sh $tensorflow_include_dir
```
For TF2, `tensorflow_include_dir` will be `<...>/site-packages/tensorflow/include`. \
TF1 does not seem to include `.proto` files with the package. In this case, Tensorflow sources can be used.
`tensorflow_include_dir` will then be the root of the Tensorflow repo.

## Overview

This module provides command line tools to facilitate replacing BERT calculations in TensorFlow models with an
optimized solution. The tools are:

1. [Pattern extraction](#pattern-extraction)
2. [Pattern replacement](#pattern-replacement)
3. [Bert op configuration](#bert-op-configuration)

## Pattern extraction

This tool is used to extract a pattern of graph nodes from a model, which can later be used to locate the same pattern
in another model.

The pattern extraction tool is used on a model by specifying:
- the model to work on
- a set of *seed nodes* (essentially "output" of the subgraph)
- a set of *barrier nodes* ("input" into the subgraph)

The program will traverse the graph backwards starting at the first seed node and stopping at barrier nodes. This is
repeated for each seed node. The pattern consists of all the visited nodes (barrier nodes excluded).

Not all barrier nodes must be visited for the output to be valid. This allows for easier definition of barrier nodes
(see the *barrier_ops* parameter in the *usage* section below).
However, the boundary of the pattern must be completely enclosed by barrier nodes. The program will fail if it reaches
the beginning of the graph (meaning any node with no inputs, like *Placeholder*) and it is not a barrier node.

### extract_pattern.py usage

The tool works on both *saved model* and *frozen* formats. To use it on a saved model, provide the **directory** of the
model (**not** the saved_model.pb). This is done to stay consistent with TensorFlow's own tools. To use it on a frozen
model, provide the path to the .pb file.

The output of the program is a protobuf file containing the pattern. Refer to `pattern.proto` for details regarding
the contents of that file.

Below is an example of how the tool can be used to extract the BERT pattern from an 
[uncased_L-12_H-768_A-12](https://github.com/google-research/bert) model.

```bash
$ ./extract_pattern.py /path/to/uncased_L-12_H-768_A-12 \
-o /path/to/bert_pattern.pb \
-s \
    bert/encoder/layer_11/output/LayerNorm/batchnorm/add_1 \
-b \
    bert/encoder/Reshape_1 \
    bert/encoder/Reshape \
    bert/encoder/strided_slice \
    bert/encoder/strided_slice_2 \
-B \
    Identity \
    Const \
-m 0
```

The arguments provided are:
- path to the model (positional argument)
- -o (--output): name of the output file
- -s (--seed-nodes): list of seed nodes to start the algorithm at
- -b (--barrier-nodes): list of node names to use as barrier nodes
- -B (--barrier-ops): list of operators to use as barrier nodes (all nodes of these types will be treated as barriers)
- -m (--meta-graph): which meta graph to use (defaults to 0, meaning the first meta graph)

Refer to `extract_pattern.py` for a full list of available arguments.

## Pattern replacement

This tool is used to locate a pattern of graph nodes in a model and replace it with an optimized equivalent. It is what
clients of the solution will use to optimize their models.

The user must provide two inputs:
- the source model (owned by the client, based on one of the BERT models supported by this module)
- [the recipe protobuf file](#the-recipe-file) (provided by the owners of this module)

The program will take the pattern from the recipe file and attempt to locate the same pattern in the source model.
If found, it will remove the nodes of that pattern from the model, insert the optimized equivalent specified in the
recipe, and rewire the inputs/outputs of nodes accordingly.

### replace_pattern.py usage

Same as `extract_pattern.py`, this tool works both on frozen and saved models. Same rules for using these formats apply.

The output of this program is a modified model protobuf, where the pattern specified in the Recipe file is replaced with
the optimized equivalent.

An important note:
- For a frozen model, the optimized .pb file contains the graph and the weights, i.e. the output of the program is an
fully functional, self-contained TF model.
- For a saved model, the optimized .pb file is just the graph definition (no weights), since the saved model's weights
are kept in a separate file. This avoids an unnecessary copy of the model's weights (potentially hundreds of MB).
The saved_model.pb of the original model can be simply replaced with the optimized protobuf.

An example of the script in use is shown below:

```bash
$ ./replace_pattern.py /path/to/uncased_L-12_H-768_A-12 \
    -r /path/to/bert_recipe.pb \
    -o /path/to/uncased_L-12_H-768_A-12/modified_saved_model.pb
```

The arguments are:
- path to the model (positional argument)
- -r (--recipe): path to the recipe protobuf
- -o (--output): where to save the output protobuf

In this case, the modified graph is saved in the original model's directory. This makes both the original and the
optimized graphs easily available and transferable. To enable the optimized graph, one can simply do:

```bash
$ cd /path/to/uncased_L-12_H-768_A-12
$ mv saved_model.pb original_saved_model.pb
$ ln -s modified_saved_model.pb saved_model.pb
```

At this point, the user can freely switch between the original and optimized graphs by creating the appropriate symlink.

### The recipe file

The *recipe* proto currently consists of two fields:
```
message Recipe {
    Pattern source_pattern = 1;
    tensorflow.NodeDef target_node = 2;
}
```

Preparing the recipe file requires a bit of manual work. First, a **Pattern** protobuf must be generated using
the pattern extraction tool. Then, the optimized equivalent must be defined. At the moment this can only be a single
node, so it takes the form of a **tensorflow.NodeDef** protobuf. In most cases the NodeDef will need to be created
manually. Specifically the inputs of the node must correspond to appropriate nodes in the graph from which the Pattern
was extracted. This is necessary for the tool to later remap these inputs onto the optimized model.

TODO: Provide a visual example of the input remapping described above (hard to explain with text only)

When both the Pattern and the NodeDef are ready, the Recipe protobuf can be created. A simple utility
[script](util/make_recipe.py) was created for this purpose, though it is also possible to do the same in an interactive
python terminal.

By design, a recipe file is created for a specific BERT model variant, and will work only for models based on that
variant. If, by coincidence, two BERT model variants have the exact same BERT structure, a single recipe file should
work on both of them, but it is not an intentional feature.

## Bert op configuration

The `configure_bert_op.py` tool can be used to set the quantization, bfloat16 and other attributes of all Bert operators
in a modified model. The attributes can be set independently. The tool works both on saved and frozen model formats.

See the output of `python -m model_modifier.configure_bert_op --help` for the full list of available parameters.

### Useful BertOp configurations

#### Pure FP32 mode

```sh
$ python -m model_modifier.configure_bert_op \
    --no-quantization \
    --no-bfloat16 \
    /path/to/saved/model/folder
```

In this mode, the Bert op uses FP32 to perform all computations.

#### Calibration mode

```sh
$ python -m model_modifier.configure_bert_op \
    --no-quantization \
    --no-bfloat16 \
    --calibrate \
    --quant-factors-path /path/to/quantization/factors/file \
    /path/to/saved/model/folder
```

In this mode, the Bert op will run all computations using FP32, but will track the highest and lowest values of input
tensors and intermediate buffers. These will be saved to the file indicated by `--quant-factors-path`. The values
can later be loaded for to calculate quantization factors.

#### Quantized mode

```sh
$ python -m model_modifier.configure_bert_op \
    --quantization \
    --no-bfloat16 \
    --no-calibrate \
    --quant-factors-path /path/to/quantization/factors/file \
    /path/to/saved/model/folder
```

In this mode, the Bert operator will load the previously saved values from `--quant-factors-path` and use them to
calculate quantization scaling factors, based on the target quantization datatype (currently INT8).

#### BFloat16 mode

```sh
$ python -m model_modifier.configure_bert_op \
    --no-quantization \
    --bfloat16 \
    --no-calibrate \
    /path/to/saved/model/folder
```

In this mode, BFloat16 will be used for floating point computations.

### Other Examples

To set all Bert ops in a saved model to use both quantization and bfloat16:
```sh
$ python -m model_modifier.configure_bert_op --quantization --bfloat16 --quant-factors-path /path/to/quantization/factors/file /path/to/saved/model/folder
```

If a parameter is not specified, the corresponding attribute will be left unchanged in the model.

So, if we want to disable quantization, but leave bfloat16 usage as-is (no matter if it's enabled or disabled) in a
saved model:
```sh
$ python -m model_modifier.configure_bert_op --no-quantization /path/to/saved/model/folder
```

If we don't want to modify the model graph in place, we can add an output flag to save the graph to a separate file:
```sh
$ python -m model_modifier.configure_bert_op --no-quantization -o /path/to/copy/of/saved/model/saved_model_copy.pb /path/to/saved/model/folder
```
Notice that in this case, even for a saved model, the output path is a .pb file, not a folder - the program makes
a copy of the saved model's execution graph, but not the variables, assets etc.
