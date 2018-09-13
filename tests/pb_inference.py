import tensorflow as tf
import numpy as np
import time
import argparse
import cv2

# Find a node in graph_def by name
def find_node(graph_def, name):
    for node in graph_def.node:
        if node.name == name:
            return node
    print("Cannot find node(%s) in the graph!" % name)
    exit(-1)

# Find a node with the op of Placeholder
def find_placeholder(graph_def):
    for node in graph_def.node:
        if node.op == "Placeholder":
            return node
    print("Cannot find placeholder in the graph!")

def load(model_file, input_node = "", output_node = "", bz = 1, loop = 100, image_file = ""):
    with tf.Graph().as_default() as g:
        g_def = tf.GraphDef()

        with open(model_file, "rb") as f:
            g_def.ParseFromString(f.read())
            _ = tf.import_graph_def(g_def, name="")

            if input_node == "":
                input_node_def =  find_placeholder(g_def)
            else:
                input_node_def = find_node(g_def, input_node)

            if output_node == "":
                output_node_def = g_def.node[-1]
            else:
                output_node_def = find_node(g_def, output_node)

            with tf.Session() as sess:
                input_node = sess.graph.get_tensor_by_name(input_node_def.name + ":0")
                output_node = sess.graph.get_tensor_by_name(output_node_def.name + ":0")
                
                ipt_shape =  input_node.shape.as_list()
                ipt_shape[0] = bz
                print(ipt_shape[0],ipt_shape[1],ipt_shape[2],ipt_shape[3])
                if image_file == "":
                    im_data = np.ones(shape = ipt_shape)
                else:
                    img = cv2.imread(image_file, cv2.IMREAD_COLOR)
                    img = img.astype('float32')
                    img = np.tile(img, bz)
                    print(img.shape)
                    #img = img.reshape(1, 224, 224, 3)
                    im_data = img.reshape(bz, ipt_shape[1], ipt_shape[2], ipt_shape[3])

                start_time = time.time()
                for _ in range(loop):
                    predictions = sess.run(output_node, feed_dict = {input_node: im_data})
                    
                end_time = time.time()
                esp_time =  (end_time - start_time) / float(loop)
                esp_ms_time = round(esp_time * 1000, 2)
                              
                print( "tensorflow output: %s" % predictions[0,0:10])
                print("TF time used per loop is: %s ms" % esp_ms_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--pb_file", type=str, default="", help="model file (freezed pb file)")
    parser.add_argument("-i", "--input_node", type=str, default="", help="input node name (placeholder)")
    parser.add_argument("-o", "--output_node", type=str, default="", help="output node name")
    parser.add_argument("-b", "--batch_size", type=int, default=1, help="input batch size")
    parser.add_argument("-l", "--loop", type=int, default=1, help="inference loops")
    parser.add_argument("-p", "--picture", type=str, default="", help="picture file as the input")
    args = parser.parse_args()
    load(args.pb_file, args.input_node, args.output_node, args.batch_size, args.loop, args.picture)
