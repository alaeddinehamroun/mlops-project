import os
import matplotlib.pyplot as plt
import mlflow
import mlflow
import tensorflow as tf
import hydra
from omegaconf import DictConfig
from metrics import precision, recall, f1_score
from dotenv import load_dotenv
load_dotenv()
MLFLOW_TRACKING_URI=os.getenv('MLFLOW_TRACKING_URI')
DATABRICKS_USER=os.getenv('DATABRICKS_USER')
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(f"/Users/{DATABRICKS_USER}/testing")
mlflow.tensorflow.autolog()



#os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
data_dir = 'PokemonData'
batch_size = 32
img_height = 180
img_width = 180


@hydra.main(version_base=None, config_path='../conf', config_name="config")

def main(cfg):
    
    # Load dataset
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)


    # Get number of classes
    num_classes = len(train_ds.class_names)

    # Configure the dataset for performance
    # AUTOTUNE = tf.data.AUTOTUNE
    # train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    # val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)    

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
        tf.keras.layers.Dense(151)
    ])



    # Compile the model
    model.compile(
        optimizer='adam',
        
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy', precision, recall, f1_score])
    # Train the model
    epochs = 2

    mlflow.tensorflow.autolog()

    history = model.fit(train_ds,
                    epochs=epochs,
                    verbose=1,
                    validation_data=val_ds)

    

if __name__=="__main__":
    main()

mlflow.end_run()





# acc=history.history['accuracy']
# val_acc=history.history['val_accuracy']
# loss=history.history['loss']
# val_loss=history.history['val_loss']
# precision = history.history['precision']
# recall = history.history['recall']
# auc = history.history['auc']

# epochs=range(len(acc))
# # Plot training and validation accuracy per epoch
# plt.plot(epochs, acc, 'r', "Training Accuracy")
# plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
# plt.title('Training and validation accuracy')
# plt.savefig('accuracy.png', dpi=80)
# plt.close()

# # Plot training and validation loss per epoch
# plt.plot(epochs, loss, 'r', "Training Loss")
# plt.plot(epochs, val_loss, 'b', "Validation Loss")
# plt.title('Training and validation loss')
# plt.savefig('loss.png', dpi=80)
# plt.close()

# # Plot precision per epoch
# plt.plot(epochs, precision, 'r', "Precision")
# plt.title('Precision')
# plt.savefig('precision.png', dpi=80)
# plt.close()

# # Plot recall per epoch
# plt.plot(epochs, recall, 'r', "Recall")
# plt.title('Recall')
# plt.savefig('recall.png', dpi=80)
# plt.close()

# # Plot AUC per epoch
# plt.plot(epochs, auc, 'r', "AUC")
# plt.title('AUC')
# plt.savefig('auc.png', dpi=80)
# plt.close()


