# coding: utf-8

from __future__ import division, print_function

import tensorflow as tf
import numpy as np
import argparse
import cv2
import os
import json

from utils.misc_utils import parse_anchors, read_class_names
from utils.nms_utils import gpu_nms
from utils.plot_utils import get_color_table, plot_one_box
from utils.data_aug import letterbox_resize
import collections

from model import yolov3

parser = argparse.ArgumentParser(description="YOLO-V3 test single image test procedure.")
parser.add_argument("--anchor_path", type=str, default="./data/yolo_anchors.txt",
                    help="The path of the anchor txt file.")
parser.add_argument("--new_size", nargs='*', type=int, default=[416, 416],
                    help="Resize the input image with `new_size`, size format: [width, height]")
parser.add_argument("--letterbox_resize", type=lambda x: (str(x).lower() == 'true'), default=True,
                    help="Whether to use the letterbox resize.")
parser.add_argument("--class_name_path", type=str, default="./data/my_data/data.names",
                    help="The path of the class names.")
parser.add_argument("--restore_path", type=str,
                    default="./checkpoint/best_model_Epoch_20_step_123626_mAP_0.1232_loss_200.4872_lr_0.0001",
                    help="The path of the weights to restore.")
args = parser.parse_args()

args.anchors = parse_anchors(args.anchor_path)
args.classes = read_class_names(args.class_name_path)
args.num_class = len(args.classes)

color_table = get_color_table(args.num_class)
label_t = []
with open('./data/my_data/data.names', 'r') as f:
    for line in f:
        label_t.append(line.replace('\n', ''))
print(label_t)

with tf.Session() as sess:
    input_data = tf.placeholder(tf.float32, [1, args.new_size[1], args.new_size[0], 3], name='input_data')
    yolo_model = yolov3(args.num_class, args.anchors)
    with tf.variable_scope('yolov3'):
        pred_feature_maps = yolo_model.forward(input_data, False)
    pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)

    pred_scores = pred_confs * pred_probs

    boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, args.num_class, max_boxes=200, score_thresh=0.3,
                                    nms_thresh=0.45)

    saver = tf.train.Saver()
    saver.restore(sess, args.restore_path)
    images = []
    for item in os.listdir('./test'):
        list_t = item.split('.')
        for i in range(len(list_t)):
            images.append(int(list_t[0]))
    images.sort()
    sizes = []
    obj_id = 0
    dict_all = {}
    dict_all = collections.OrderedDict(dict_all)
    for s in images:
        dict_single = {}
        dict_single = collections.OrderedDict(dict_single)
        img_ori = cv2.imread('./test/' + str(s) + '.jpg')
        size = img_ori.shape
        dict_single["height"] = size[0]
        dict_single["width"] = size[1]
        dict_single["depth"] = size[2]
        if args.letterbox_resize:
            img, resize_ratio, dw, dh = letterbox_resize(img_ori, args.new_size[0], args.new_size[1])
        else:
            height_ori, width_ori = img_ori.shape[:2]
            img = cv2.resize(img_ori, tuple(args.new_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.asarray(img, np.float32)
        img = img[np.newaxis, :] / 255.

        boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_data: img})

        # rescale the coordinates to the original image
        if args.letterbox_resize:
            boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
            boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio
        else:
            boxes_[:, [0, 2]] *= (width_ori / float(args.new_size[0]))
            boxes_[:, [1, 3]] *= (height_ori / float(args.new_size[1]))
        print('./test/' + str(s) + '.jpg')
        print("box coords:")
        print(boxes_)
        print('*' * 30)
        print("scores:")
        print(scores_)
        print('*' * 30)
        print("labels:")
        print(labels_)
        box = []
        dict_obj = {}
        dict_obj = collections.OrderedDict(dict_obj)
        for m in range(len(boxes_)):
            x0, y0, x1, y1 = boxes_[m]
            list_box = [int(x0), int(y0), int(x1), int(y1)]
            label_tmp = labels_[m]
            dict_cat = {}
            dict_cat = collections.OrderedDict(dict_cat)
            dict_cat["category"] = label_t[label_tmp]
            dict_cat["bbox"] = list_box
            dict_obj[str(obj_id)] = dict_cat
            obj_id = obj_id + 1
        dict_single["objects"] = dict_obj
        dict_all["" + str(s) + ".jpg"] = dict_single
    file_name = 'test_new.json'
    with open(file_name, 'w') as fj:
        dict_all = collections.OrderedDict(dict_all)
        json.dump(dict_all, fj)

    '''for i in range(len(boxes_)):
        x0, y0, x1, y1 = boxes_[i]
        plot_one_box(img_ori, [x0, y0, x1, y1], label=args.classes[labels_[i]] + ', {:.2f}%'.format(scores_[i] * 100), color=color_table[labels_[i]])
    cv2.imshow('Detection result', img_ori)
    cv2.imwrite('detection_result.jpg', img_ori)
    cv2.waitKey(0)'''

