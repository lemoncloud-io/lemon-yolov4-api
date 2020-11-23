# coding=utf-8
"""
to service the prediction of image
- use the trained `yolov4.weights`

@copyright  lemoncloud.io 2020
"""
from os import read
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# curren folder.
def curr_dir():
    import os
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return dir_path

def as_path(p: str, read = True):
    import os
    f = os.path.join(curr_dir(), p[2:]) if p.startswith('./') else p
    if read and not os.path.exists(f): raise Exception('file:{} is NOT valid!'.format(p))
    return f

_model = None
def load_model(weights='./checkpoints/yolov4-416'):
    global _model
    if _model is not None: return _model
    ''' prepare model '''
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    # STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    saved_model_loaded = tf.saved_model.load(as_path(weights), tags=[tag_constants.SERVING])
    _model = saved_model_loaded
    return _model


def infer_image(image_path: str, output: str = ''):
    input_size = 416

    print('infer:{}'.format(as_path(image_path)))
    original_image = cv2.imread(as_path(image_path))
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    image_data = cv2.resize(original_image, (input_size, input_size))
    image_data = image_data / 255.
    # image_data = image_data[np.newaxis, ...].astype(np.float32)

    images_data = []
    for i in range(1):
        images_data.append(image_data)
    images_data = np.asarray(images_data).astype(np.float32)

    # run infer...............
    models = load_model()
    infer = models.signatures['serving_default']
    batch_data = tf.constant(images_data)
    pred_bbox = infer(batch_data)

    boxes = None
    pred_conf = None
    for key, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]

    #! generate the result....
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=0.45,
        score_threshold=0.25
    )
    pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
    # print('> pred_bbox = {}'.format(pred_bbox))
    info = info_bbox(original_image, pred_bbox)
    image = utils.draw_bbox(original_image, pred_bbox)
    batch_data = tf.constant(images_data)

    # image = utils.draw_bbox(image_data*255, pred_bbox)
    image = Image.fromarray(image.astype(np.uint8))
    image.show()
    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    cv2.imwrite(as_path(output, read=False), image)

    # returns..
    return {
        'file': output,
        'bbox': info,
    }

# extract infor
def info_bbox(image, bboxes):
    max_classes = 5
    image_h, image_w, _ = image.shape
    out_boxes, out_scores, out_classes, num_boxes = bboxes
    boxes = []
    for i in range(num_boxes[0]):
        if int(out_classes[0][i]) < 0 or int(out_classes[0][i]) > max_classes: continue
        coor = out_boxes[0][i]
        coor0 = int(coor[0] * image_h)
        coor2 = int(coor[2] * image_h)
        coor1 = int(coor[1] * image_w)
        coor3 = int(coor[3] * image_w)
        score = int(100 * out_scores[0][i])
        class_ind = int(out_classes[0][i])
        boxes.append((class_ind, coor1, coor0, coor3, coor2, score))
    return { 'w': image_w, 'h': image_h, 'boxes': boxes }