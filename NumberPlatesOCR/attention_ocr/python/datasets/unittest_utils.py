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

"""Functions to make unit testing easier."""

from io import StringIO
# import StringIO
import numpy as np
from PIL import Image as PILImage
import tensorflow as tf


def create_random_image(image_format, shape):
  """Creates an image with random values.

  Args:
    image_format: An image format (PNG or JPEG).
    shape: A tuple with image shape (including channels).

  Returns:
    A tuple (<numpy ndarray>, <a string with encoded image>)
  """
  image = np.random.randint(low=0, high=255, size=shape, dtype='uint8')
  io = StringIO.StringIO()
  image_pil = PILImage.fromarray(image)
  image_pil.save(io, image_format, subsampling=0, quality=100)
  return image, io.getvalue()


def create_serialized_example(name_to_values):
  """Creates a tf.Example proto using a dictionary.

  It automatically detects type of values and define a corresponding feature.

  Args:
    name_to_values: A dictionary.

  Returns:
    tf.Example proto.
  """
  example = tf.train.Example()
  for name, values in name_to_values.items():
    feature = example.features.feature[name]
    if isinstance(values[0], str):
      add = feature.bytes_list.value.extend
    elif isinstance(values[0], float):
      add = feature.float32_list.value.extend
    elif isinstance(values[0], int):
      add = feature.int64_list.value.extend
    else:
      raise AssertionError('Unsupported type: %s' % type(values[0]))
    add(values)
  return example.SerializeToString()
