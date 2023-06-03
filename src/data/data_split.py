import tensorflow as tf


def train_val_test_split(data_dir, img_height=180, img_width=180, batch_size=32):
    """ Split the data
    
    Keyword arguments:
    data_dir -- data directory
    img_height -- images' height
    img_width -- images' width
    batch_size -- batch size
    
    Return: train/val/split loaders
    """
    
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)


    # Create test set from validation set: determine how many batches of data are available in the validation set,
    # then move 20% of them to a test set.
    val_batches = tf.data.experimental.cardinality(validation_dataset)
    test_dataset = validation_dataset.take(val_batches // 5)
    validation_dataset = validation_dataset.skip(val_batches // 5)

    return train_dataset, validation_dataset, test_dataset
