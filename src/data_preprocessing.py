from tensorflow.keras.preprocessing.image import ImageDataGenerator

def train_val_generators(training_dir, validation_dir):
    # Instantiate the ImageDataGenerator class (don't forget to set the rescale argument)
    train_datagen = ImageDataGenerator(rescale = 1.0/255. )
    test_datagen = ImageDataGenerator(rescale = 1.0/255. )
    # Pass in the appropiate arguments to the flow_from_directory method
    train_generator = train_datagen.flow_from_directory(directory=training_dir,
                                                      batch_size=100,
                                                      target_size=(150, 150))

    # Instantiate the ImageDataGenerator class (don't forget to set the rescale argument)
    validation_datagen = ImageDataGenerator(rescale = 1.0/255. )

    # Pass in the appropiate arguments to the flow_from_directory method
    validation_generator = validation_datagen.flow_from_directory(directory=validation_dir,
                                                                batch_size=100,
                                                                target_size=(150, 150))

    return train_generator, validation_generator