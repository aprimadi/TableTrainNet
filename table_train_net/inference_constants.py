from dataset_constants import TABLE_DICT

PATH_TO_LABELS = 'data/object-detection.pbtxt'
BMP_IMAGE_TEST_TO_PATH = 'test'

NUM_CLASSES = 1

PATHS_TO_TEST_IMAGE = [
    # 'test/test1.png',
    # 'test/test2.png',
    # 'test/test3.png',
    # 'test/test4.png',
    # 'test/test5.png',
    # 'test/test6.png',
    # 'test/test7.png',
    'test/test8.png',
]

PATHS_TO_CKPTS = [
    # 'data/',
    # 'trained_models/model__rcnn_inception_momentum_optimizer_1batch/frozen/frozen_inference_graph.pb',
    # 'trained_models/model__rcnn_inception_adam_1/frozen/frozen_inference_graph.pb',
    # 'trained_models/model__rcnn_inception_adam_3/frozen/frozen_inference_graph.pb',
    # 'trained_models/model__rcnn_inception_momentum_1/frozen/frozen_inference_graph.pb',
    # 'trained_models/model__rcnn_inception_momentum_10k_jpg/frozen/frozen_inference_graph.pb',
    'trained_models/model__rcnn_inception_momentum_optimizer_1batch/frozen/frozen_inference_graph.pb',
    # 'trained_models/model__rcnn_inception_adam_4/frozen/frozen_inference_graph.pb'
]

# TEST_SCORES = [0.2, 0.4, 0.6, 0.8]
TEST_SCORES = [0.2]

MAX_NUM_BOXES = 10
