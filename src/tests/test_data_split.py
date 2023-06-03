import pytest
import tensorflow as tf

from data.data_split import train_val_test_split

@pytest.fixture
def data_dir():
    return "PokemonData"

def test_train_val_test_split(data_dir):
    img_height = 180
    img_width = 180
    batch_size = 32

    train_dataset, validation_dataset, test_dataset = train_val_test_split(
        data_dir, img_height, img_width, batch_size
    )

    # Assert that train_dataset is of type tf.data.Dataset
    assert isinstance(train_dataset, tf.data.Dataset)

    # Assert that validation_dataset is of type tf.data.Dataset
    assert isinstance(validation_dataset, tf.data.Dataset)

    # Assert that test_dataset is of type tf.data.Dataset
    assert isinstance(test_dataset, tf.data.Dataset)

    # Assert that the batch size of train_dataset matches the input batch_size
    assert train_dataset.element_spec[0].shape[0] == batch_size

    # Assert that the batch size of validation_dataset matches the input batch_size
    assert validation_dataset.element_spec[0].shape[0] == batch_size

    # Assert that the batch size of test_dataset is less than or equal to the input batch_size
    assert test_dataset.element_spec[0].shape[0] <= batch_size

    # Assert that the length of the validation_dataset is less than or equal to 20% of the total dataset
    assert len(validation_dataset) <= 0.2 * (len(train_dataset) + len(validation_dataset))

    # Assert that the length of the test_dataset is 20% of the length of the validation_dataset
    assert len(test_dataset) == len(validation_dataset) // 5
