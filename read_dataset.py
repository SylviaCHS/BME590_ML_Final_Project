import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
import json

from sklearn.model_selection import train_test_split

filename = 'train.tfrecords'
raw_dataset = tf.data.TFRecordDataset(filename)

# Create a description of the features.
feature_description = {
    'label': tf.FixedLenFeature([], tf.int64, default_value=0),
    'img_raw': tf.FixedLenFeature([], tf.string, default_value=''),
}


def _parse_function(example_proto):
    # Parse the input tf.Example proto using the dictionary above.
    return tf.parse_single_example(example_proto, feature_description)


parsed_dataset = raw_dataset.map(_parse_function)
label = []
img = np.zeros([4125, 100, 100, 3]).astype(int)
for i, image_features in enumerate(parsed_dataset):
    label.append(image_features['label'].numpy())
    image = tf.decode_raw(image_features['img_raw'], tf.int8)
    img[i, :, :, :] = tf.reshape(image, [100, 100, 3])

print('saving...')
np.save('images.npy', img)
np.save('label.npy', np.array(label))
print('done!')