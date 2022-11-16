import os
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
import argparse
import tempfile
import shutil
from tensorflow.core.protobuf.saved_model_pb2 import SavedModel
from model_modifier.pattern_extractor import PatternExtractor
import logging
from google.protobuf import text_format
from tensorflow.core.framework.node_def_pb2 import NodeDef
from model_modifier.recipe_pb2 import Recipe
from model_modifier.pattern_replacer import PatternReplacer

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        description='E2E script to launch huggingface bert base uncased after modifying it with custom op')
    parser.add_argument('op_library', metavar='op-library',
                        type=str, help='Path to the .so containing the BertOp.')
    parser.add_argument('output_dir', metavar='out-dir',
                        type=str, help='Path to where the bert model will be saved')

    args = parser.parse_args() 
    tf.load_op_library(args.op_library)

    logging.basicConfig()
    logging.root.setLevel('WARN')
    log = logging.getLogger(f'{__name__}.extract_pattern')

    with tempfile.TemporaryDirectory() as tmpdirname:

        #save default hugging face model

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = TFBertModel.from_pretrained("bert-base-uncased")
        callable = tf.function(model.call)
        concrete_function = callable.get_concrete_function([tf.TensorSpec([None, None], tf.int32, name="attention_mask"),
            tf.TensorSpec([None, None], tf.int32, name="input_ids"), tf.TensorSpec([None, None], tf.int32, name="token_type_ids")])
        tf.saved_model.save(model, tmpdirname, signatures=concrete_function)

        # modify the model

        # equivalent of ./replace_full_bert.sh

        seed_nodes =  ['bert/encoder/layer_._11/output/LayerNorm/batchnorm/add_1']
        barrier_nodes = ['bert/embeddings/dropout/Identity', 'bert/Cast']
        barrier_ops = ['ReadVariableOp', 'Const']
        function_name = '__inference_call_7459'


        saved_model = SavedModel()
        saved_model_path = os.path.join(tmpdirname, 'saved_model.pb')

        with open(saved_model_path, 'rb') as f:
            saved_model.ParseFromString(f.read())
            
        try:
            graph = saved_model.meta_graphs[0].graph_def
        except IndexError as e:
            log.error(f'Error while picking the meta graph: {e}')

        pattern_extractor = PatternExtractor(graph)
        pattern = pattern_extractor.extract(seed_nodes, barrier_nodes, barrier_ops, function_name)

        with open('fused_bert_node_def.pbtxt', 'r') as f:
            node_def = text_format.Parse(f.read(), NodeDef())
        recipe = Recipe()
        recipe.source_pattern.CopyFrom(pattern)
        recipe.target_node.CopyFrom(node_def)
        pattern_replacer = PatternReplacer(graph)
        success: bool = pattern_replacer.replace(recipe)
        if success:
            with open(os.path.join(tmpdirname, 'modified_saved_model.pb'), 'wb') as f:
                f.write(saved_model.SerializeToString())

        shutil.move(tmpdirname + '/saved_model.pb', tmpdirname + '/backup.pb')
        shutil.move(tmpdirname + '/modified_saved_model.pb', tmpdirname + '/saved_model.pb')

        # move the model to output folder
        # configuring the model is done by providing the attirbutes to fused_bert_node_def

        shutil.move(tmpdirname + '/saved_model.pb', args.output_dir)
        shutil.move(tmpdirname + '/variables', args.output_dir)
    # load the model and perform inference

    model_modified =  tf.saved_model.load(args.output_dir)
    text = "Replace me by any text you'd like"
    encoded_input = tokenizer(text, return_tensors='tf',max_length=128, padding='max_length')
    output = model_modified.signatures["serving_default"](**encoded_input)
    print(output)
    print('Successfully performended inference on modified model')
