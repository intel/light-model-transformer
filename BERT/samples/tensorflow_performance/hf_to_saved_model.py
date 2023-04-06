from transformers import TFBertModel
import tensorflow as tf
import argparse

def main(args):
    model = TFBertModel.from_pretrained(args.model)
    callable = tf.function(model.call)
    concrete_function = callable.get_concrete_function([tf.TensorSpec([None, None], tf.int32, name="attention_mask"),
        tf.TensorSpec([None, None], tf.int32, name="input_ids"), tf.TensorSpec([None, None], tf.int32, name="token_type_ids")])
    tf.saved_model.save(model, args.output, signatures=concrete_function)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='Model to convert')
    parser.add_argument('output', type=str, help='Output directory')
    args = parser.parse_args()
    main(args)
