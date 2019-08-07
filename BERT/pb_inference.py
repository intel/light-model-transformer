import tensorflow as tf
import numpy as np
import json
import time
import sys

if len(sys.argv) != 3:
    print("Usage: %s pb_file output_node" % sys.argv[0])
    sys.exit(-1)

bert_module = tf.load_op_library('./tf_ops/bert.so')

with tf.Session() as sess:
    g = tf.Graph().as_default()

    pb_file = sys.argv[1]
    with open(pb_file, "rb") as f:
        g_def = tf.GraphDef()
        g_def.ParseFromString(f.read())
        _ = tf.import_graph_def(g_def, name="")


    input_name = 'placeholder:0'
    output_name = sys.argv[2] + ':0'
    init_input = sess.graph.get_tensor_by_name(input_name)
    
    input_data = np.ones([128])

    # Warm up
    result = sess.run(output_name, feed_dict={init_input: [input_data]})
    print (result)

    loops = 10
    start = time.time()
    for _ in range(loops):
        result = sess.run(output_name, feed_dict={init_input: [input_data]})
    end = time.time()
    print ('Total time of %d loops: %f seconds, each %f ms' % (loops, end - start, (end - start) * 1000 / loops))

