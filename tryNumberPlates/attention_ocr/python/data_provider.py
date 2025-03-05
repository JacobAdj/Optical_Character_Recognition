# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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
# ==============================================================================

"""Functions to read, decode and pre-process input data for the Model.
"""
import collections
import functools
import tensorflow as tf

import tf_slim as slim

# from tensorflow.contrib import slim

import inception_preprocessing

# Tuple to store input data endpoints for the Model.
# It has following fields (tensors):
#    images: input images,
#      shape [batch_size x H x W x 3];
#    labels: ground truth label ids,
#      shape=[batch_size x seq_length];
#    labels_one_hot: labels in one-hot encoding,
#      shape [batch_size x seq_length x num_char_classes];
InputEndpoints = collections.namedtuple(
    'InputEndpoints', ['images', 'images_orig', 'labels', 'labels_one_hot'])

# A namedtuple to define a configuration for shuffled batch fetching.
#   num_batching_threads: A number of parallel threads to fetch data.
#   queue_capacity: a max number of elements in the batch shuffling queue.
#   min_after_dequeue: a min number elements in the queue after a dequeue, used
#     to ensure a level of mixing of elements.
ShuffleBatchConfig = collections.namedtuple('ShuffleBatchConfig', [
    'num_batching_threads', 'queue_capacity', 'min_after_dequeue'
])

DEFAULT_SHUFFLE_CONFIG = ShuffleBatchConfig(
    num_batching_threads=8, queue_capacity=3000, min_after_dequeue=1000)


def augment_image(image):
  """Augmentation the image with a random modification.

  Args:
    image: input Tensor image of rank 3, with the last dimension
           of size 3.

  Returns:
    Distorted Tensor image of the same shape.
  """
  with tf.compat.v1.variable_scope('AugmentImage'):
    height = image.get_shape().dims[0].value
    width = image.get_shape().dims[1].value

    # Random crop cut from the street sign image, resized to the same size.
    # Assures that the crop is covers at least 0.8 area of the input image.
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
        image_size=tf.shape(input=image),
        bounding_boxes=tf.zeros([0, 0, 4]),
        min_object_covered=0.8,
        aspect_ratio_range=[0.8, 1.2],
        area_range=[0.8, 1.0],
        use_image_if_no_bounding_boxes=True)
    distorted_image = tf.slice(image, bbox_begin, bbox_size)

    # Randomly chooses one of the 4 interpolation methods
    distorted_image = inception_preprocessing.apply_with_random_selector(
        distorted_image,
        lambda x, method: tf.image.resize(x, [height, width], method),
        num_cases=4)
    distorted_image.set_shape([height, width, 3])

    # Color distortion
    distorted_image = inception_preprocessing.apply_with_random_selector(
        distorted_image,
        functools.partial(
            inception_preprocessing.distort_color, fast_mode=False),
        num_cases=4)
    distorted_image = tf.clip_by_value(distorted_image, -1.5, 1.5)

  return distorted_image


def central_crop(image, crop_size):
  """Returns a central crop for the specified size of an image.

  Args:
    image: A tensor with shape [height, width, channels]
    crop_size: A tuple (crop_width, crop_height)

  Returns:
    A tensor of shape [crop_height, crop_width, channels].
  """
  with tf.compat.v1.variable_scope('CentralCrop'):
    target_width, target_height = crop_size
    image_height, image_width = tf.shape(
        input=image)[0], tf.shape(input=image)[1]
    assert_op1 = tf.Assert(
        tf.greater_equal(image_height, target_height),
        ['image_height < target_height', image_height, target_height])
    assert_op2 = tf.Assert(
        tf.greater_equal(image_width, target_width),
        ['image_width < target_width', image_width, target_width])
    with tf.control_dependencies([assert_op1, assert_op2]):
      offset_width = tf.cast((image_width - target_width) / 2, tf.int32)
      offset_height = tf.cast((image_height - target_height) / 2, tf.int32)
      return tf.image.crop_to_bounding_box(image, offset_height, offset_width,
                                           target_height, target_width)


def preprocess_image(image, augment=False, central_crop_size=None,
                     num_towers=4):
  """Normalizes image to have values in a narrow range around zero.

  Args:
    image: a [H x W x 3] uint8 tensor.
    augment: optional, if True do random image distortion.
    central_crop_size: A tuple (crop_width, crop_height).
    num_towers: optional, number of shots of the same image in the input image.

  Returns:
    A float32 tensor of shape [H x W x 3] with RGB values in the required
    range.
  """
  with tf.compat.v1.variable_scope('PreprocessImage'):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    if augment or central_crop_size:
      if num_towers == 1:
        images = [image]
      else:
        images = tf.split(value=image, num_or_size_splits=num_towers, axis=1)
      if central_crop_size:
        view_crop_size = (int(central_crop_size[0] / num_towers),
                          central_crop_size[1])
        images = [central_crop(img, view_crop_size) for img in images]
      if augment:
        images = [augment_image(img) for img in images]
      image = tf.concat(images, 1)

  return image




def get_data(dataset, batch_size, augment=False, central_crop_size=None, shuffle=True):
    import tensorflow as tf

    def preprocess_image(image, augment, central_crop_size):
        # Define your preprocessing function here
        # For example:
        if central_crop_size:
            image = tf.image.central_crop(image, central_crop_size)
        if augment:
            image = tf.image.random_flip_left_right(image)
        return image

    def parse_function(example_proto):
        # Define your parsing function here to extract 'image' and 'label' from the dataset
        features = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64)
        }
        parsed_features = tf.io.parse_single_example(example_proto, features)
        image = tf.image.decode_jpeg(parsed_features['image'], channels=3)
        label = parsed_features['label']
        return image, label

    # Ensure the dataset is a tf.data.Dataset object
    if isinstance(dataset, tf.data.Dataset):
        if shuffle:
            dataset = dataset.shuffle(buffer_size=10000)

        dataset = dataset.map(lambda image, label: (preprocess_image(image, augment, central_crop_size), label))

        dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset
    else:
        raise ValueError("The provided dataset is not a tf.data.Dataset object")

# Example usage
# Create a tf.data.Dataset object (replace this with your actual dataset)
# dataset = tf.data.TFRecordDataset(filenames=["your_dataset.tfrecord"])

# Call the get_data function
# batch_size = 32
# tf_dataset = get_data(dataset, batch_size, augment=True, central_crop_size=(224, 224), shuffle=True)

# To access num_char_classes (example placeholder, adjust according to actual implementation)
num_char_classes = 10  # Replace with the actual number of character classes in your dataset
print("Number of character classes:", num_char_classes)

# Example usage
# Create a tf.data.Dataset object (replace this with your actual dataset)
# dataset = tf.data.TFRecordDataset(filenames=["your_dataset.tfrecord"])

# Call the get_data function
# batch_size = 32
# dataset = get_data(dataset, batch_size, augment=True, central_crop_size=(224, 224), shuffle=True)
