import tensorflow as tf

def resnet_model(IMG_SIZE = 224, num_classes=151):



    base_model = tf.keras.applications.ResNet50V2(
                            include_top=False, # Exclude ImageNet classifier at the top
                            weights='imagenet',
                            input_shape=(IMG_SIZE, IMG_SIZE, 3)
                            )

    # Freeze base_model
    base_model.trainable = False

    # Setup inputs based on input image shape
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    data_augmentation = tf.keras.Sequential(
                [tf.keras.layers.RandomFlip('horizontal'), 
                #  layers.RandomRotation(factor=(-0.025, 0.025)),
                #  layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
                #  layers.RandomContrast(factor=0.1),
                ])
    x = data_augmentation(inputs)

    # Apply specific pre-processing function for ResNet v2
    x = tf.keras.applications.resnet_v2.preprocess_input(x)

    # Keep base model batch normalization layers in inference mode (instead of training mode)
    x = base_model(x, training=False)

    # Rebuild top layers
    x = tf.keras.layers.GlobalAveragePooling2D()(x) # Average pooling operation
    x = tf.keras.layers.BatchNormalization()(x) # Introduce batch norm
    x = tf.keras.layers.Dropout(0.2)(x)  # Regularize with dropout

    # Flattening to final layer - Dense classifier with 37 units (multi-class classification)
    outputs = tf.keras.layers.Dense(num_classes, activation='relu')(x)

    return tf.keras.Model(inputs, outputs)
