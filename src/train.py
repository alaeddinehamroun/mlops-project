import os
import json
import matplotlib.pyplot as plt
import mlflow
import tensorflow as tf
import hydra
from metrics import f1_score, precision, recall
# from dotenv import load_dotenv

# load_dotenv()
# MLFLOW_TRACKING_URI=os.getenv('MLFLOW_TRACKING_URI')
# DATABRICKS_USER=os.getenv('DATABRICKS_USER')
# mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
# mlflow.set_experiment(f"/Users/{DATABRICKS_USER}/experiment-1")

mlflow.tensorflow.autolog()




os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
data_dir = 'PokemonData'
batch_size = 32
img_height = 180
img_width = 180


@hydra.main(version_base=None, config_path='../conf', config_name="config")

def main(cfg):
    
    # Load dataset
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


    # print('Number of validation batches: %d' % tf.data.experimental.cardinality(validation_dataset))
    # print('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset))


    # Get number of classes
    num_classes = len(train_dataset.class_names)

    # Configure the dataset for performance
    # AUTOTUNE = tf.data.AUTOTUNE
    # train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    # validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
    # test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
    
    # Data preprocessing

    # Normalization
    normalization_layer = tf.keras.layers.Rescaling(1./255)

    # 


    # Data augmentation
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
    ])

    # Create the model

    model = tf.keras.models.Sequential([
        normalization_layer,
        data_augmentation, # inactive at test time
        tf.keras.layers.Conv2D(cfg.model.layer1, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(cfg.model.layer2, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(num_classes)
    ])



    # Compile the model
    base_learning_rate = 0.0001
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy', f1_score, precision, recall])
    
    #model.summary()
    #len(model.trainable_variables)

    # Train the model
    num_epochs = 2


    history = model.fit(train_dataset,
                    epochs=num_epochs,
                    validation_data=validation_dataset)



    # Evaluate the model
    
    # Learning curves
    acc=history.history['accuracy']
    val_acc=history.history['val_accuracy']
    loss=history.history['loss']
    val_loss=history.history['val_loss']
    epochs=range(len(acc))
    # Plot training and validation accuracy per epoch
    plt.plot(epochs, acc, 'r', "Training Accuracy")
    plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
    plt.title('Training and validation accuracy')
    plt.savefig('accuracy.png', dpi=80)
    plt.close()
    # Plot training and validation loss per epoch
    plt.plot(epochs, loss, 'r', "Training Loss")
    plt.plot(epochs, val_loss, 'b', "Validation Loss")
    plt.title('Training and validation loss')
    plt.savefig('loss.png', dpi=80)
    plt.close()

    
    loss, accuracy, f1_s, prec, rec= model.evaluate(test_dataset)

    # print('Test accuracy :', accuracy)
    # print('Test f1_score :', f1_s)
    # print('Test precision :', prec)
    # print('Test recall :', rec)

    # Print metrics to file
    with open("metrics.json", "w") as outfile:
        json.dump(
            {
                "accuracy": accuracy,
                "f1 score": f1_s,
                "precision": prec,
                "recall": rec
            },
            outfile
        )



    


if __name__=="__main__":
    main()

mlflow.end_run()


