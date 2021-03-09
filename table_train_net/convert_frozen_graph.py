import tensorflow.compat.v1 as tf
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants

export_dir = './saved'
graph_pb = 'trained_models/model__rcnn_inception_momentum_optimizer_1batch/frozen/frozen_inference_graph.pb'

builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

with tf.gfile.GFile(graph_pb, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

sigs = {}

with tf.Session(graph=tf.Graph()) as sess:
    with sess.as_default():
        tf.saved_model.save('saved/rcnn_inception_momentum_optimizer_1batch')
    # name="" is important to ensure we don't get spurious prefixing
    # tf.import_graph_def(graph_def, name="")
    # g = tf.get_default_graph()

    # sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
    #     tf.saved_model.signature_def_utils.predict_signature_def(
    #         {"in": inp}, {"out": out})

    # builder.add_meta_graph_and_variables(sess,
    #                                      [tag_constants.SERVING],
    #                                      signature_def_map=sigs)

# builder.save()
