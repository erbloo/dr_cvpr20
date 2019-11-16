# Copyright 2019 Baidu Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Keras model wrapper for RetinaResnet50 object detection."""

import keras
from models.retina_resnet50.retinanet_resnet_50.utils.image import preprocess_image, resize_image
from models.retina_resnet50.retinanet_resnet_50 import models
import numpy as np
import tensorflow as tf
import pdb


class KerasResNet50RetinaNetModel():
    def __init__(self,
                 model=None,
                 num_classes=80,
                 channel_axis=3,
                 score=0.5,
                 iou=0.5,
                 labels_to_names={
                     0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle',
                     4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck',
                     8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
                     11: 'stop sign', 12: 'parking meter', 13: 'bench',
                     14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse',
                     18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
                     22: 'zebra', 23: 'giraffe', 24: 'backpack',
                     25: 'umbrella', 26: 'handbag', 27: 'tie',
                     28: 'suitcase', 29: 'frisbee', 30: 'skis',
                     31: 'snowboard', 32: 'sports ball', 33: 'kite',
                     34: 'baseball bat', 35: 'baseball glove',
                     36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
                     39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork',
                     43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana',
                     47: 'apple', 48: 'sandwich', 49: 'orange',
                     50: 'broccoli', 51: 'carrot', 52: 'hot dog',
                     53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair',
                     57: 'couch', 58: 'potted plant', 59: 'bed',
                     60: 'dining table', 61: 'toilet', 62: 'tv',
                     63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
                     67: 'cell phone', 68: 'microwave', 69: 'oven',
                     70: 'toaster', 71: 'sink', 72: 'refrigerator',
                     73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors',
                     77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}):

        self._labels_to_names = labels_to_names
        self._min_overlap = iou
        keras.backend.tensorflow_backend.set_session(self._get_session())
        if model is None:
            self._model = model = models.load_model(
                filepath="models/retina_resnet50/retinanet_resnet_50/model_data/resnet50_coco_best_v2.1.0.h5",
                backbone_name='resnet50')
        else:
            self._model = model
        self._th_conf = score
        self._num_classes = num_classes
        self._task = 'det'

    def _get_session(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        return tf.Session(config=config)

    def num_classes(self):
        """Return the number of classes."""
        return self._num_classes

    def class_names(self):
        """Return the class names as list."""
        return list(self._labels_to_names.values())

    def model_task(self):
        """Return the task of the model: `classification` or `detection.`"""
        return self._task

    def predict(self, image_pil):
        image_np = np.array(image_pil).astype(float)
        return self.batch_predictions(np.expand_dims(image_np, axis=0))[0]

    def batch_predictions(self, images):
        """Batch prediction of images.

        Parameters
        ----------
        images : `numpy.ndarray`
            The input image in [b, h, w, c] ndarry format.

        Returns
        -------
        list
            List of batch prediction resutls.
            Each element is a dictionary containing:
            {'boxes', 'scores', 'classes}
        """
        boxes_list, scores_list, labels_list = self._model.predict(images)
        results = []

        for boxes, scores, labels in zip(boxes_list,
                                         scores_list,
                                         labels_list):
            result = {}
            out_boxes = []
            out_scores = []
            out_classes = []
            for temp_box, temp_score, temp_class in zip(boxes,
                                                        scores,
                                                        labels):
                if temp_score >= self._th_conf:
                    temp_box = np.array(
                        [temp_box[1], temp_box[0], temp_box[3], temp_box[2]])
                    out_boxes.append(temp_box)
                    out_scores.append(temp_score)
                    out_classes.append(temp_class)
            result['boxes'] = out_boxes
            result['scores'] = out_scores
            result['classes'] = out_classes
            results.append(result)

        return results
