# Import necessary libraries
import collections
import functools
import tensorflow as tf
import inception_preprocessing

# Tuple to store input data endpoints for the Model
InputEndpoints = collections.namedtuple(
    'InputEndpoints', ['images', 'images_orig', 'labels', 'labels_one_hot'])

# A namedtuple to define a configuration for shuffled batch fetching
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
    height, width = tf.shape(image)[0], tf.shape(image)[1]

    # Random crop cut from the street sign image, resized to the same size.
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
        image_size=tf.shape(image),
        bounding_boxes=tf.zeros([1, 0, 4]),
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
    target_width, target_height = crop_size
    offset_width = (tf.shape(image)[1] - target_width) // 2
    offset_height = (tf.shape(image)[0] - target_height) // 2
    return tf.image.crop_to_bounding_box(image, offset_height, offset_width,
                                         target_height, target_width)

def preprocess_image(image, augment=False, central_crop_size=None, num_towers=4):
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
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    if augment or central_crop_size:
        if num_towers == 1:
            images = [image]
        else:
            images = tf.split(image, num_or_size_splits=num_towers, axis=1)
        if central_crop_size:
            view_crop_size = (int(central_crop_size[0] / num_towers), central_crop_size[1])
            images = [central_crop(img, view_crop_size) for img in images]
        if augment:
            images = [augment_image(img) for img in images]
        image = tf.concat(images, axis=1)
    return image

def get_data(dataset,
             batch_size,
             augment=False,
             central_crop_size=None,
             shuffle_config=None,
             shuffle=True):
    """Wraps calls to DatasetDataProviders and shuffle_batch.

    Args:
        dataset: a Dataset object.
        batch_size: number of samples per batch.
        augment: optional, if True does random image distortion.
        central_crop_size: A tuple (crop_width, crop_height).
        shuffle_config: A namedtuple ShuffleBatchConfig.
        shuffle: if True use data shuffling.

    Returns:
        InputEndpoints namedtuple with images, images_orig, labels, labels_one_hot.
    """
    if not shuffle_config:
        shuffle_config = DEFAULT_SHUFFLE_CONFIG

    def parse_fn(image, label):
        image_orig = tf.image.decode_image(image)
        image = preprocess_image(image_orig, augment, central_crop_size, num_towers=dataset.num_of_views)
        label_one_hot = tf.one_hot(label, depth=dataset.num_char_classes)
        return image, image_orig, label, label_one_hot

    dataset = tf.data.Dataset.from_tensor_slices((dataset.images, dataset.labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=shuffle_config.queue_capacity, seed=42)

    dataset = dataset.map(parse_fn, num_parallel_calls=shuffle_config.num_batching_threads)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    iterator = iter(dataset)
    images, images_orig, labels, labels_one_hot = next(iterator)

    return InputEndpoints(
        images=images,
        images_orig=images_orig,
        labels=labels,
        labels_one_hot=labels_one_hot)

