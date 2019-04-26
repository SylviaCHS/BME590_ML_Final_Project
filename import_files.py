import os
import tensorflow as tf
from PIL import Image
# import matplotlib.pyplot as plt
# import numpy as np

classes = {'TB_Image', 'Non-TB_Image'}
writer = tf.python_io.TFRecordWriter('train.tfrecords')


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


for index, name in enumerate(classes):
    class_path = './' + name + '/'
    for img_name in os.listdir(class_path):
        if img_name.endswith(".jpeg") or img_name.endswith(".jpg"):
            img_path = class_path + img_name
            img = Image.open(img_path)
            img_raw = img.tobytes()  # Convert image to bytes
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": _int64_feature(index),
                "img_raw": _bytes_feature(img_raw),
            }))
            writer.write(example.SerializeToString())

writer.close()



