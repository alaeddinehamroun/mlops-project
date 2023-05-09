import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 

from data_preprocessing import train_val_generators
from model import create_model

# Load & preprocess the dataset
trainging_dir = 'data/training'
validation_dir = 'data/validation'
train_generator, validation_generator = train_val_generators(training_dir=trainging_dir, validation_dir=validation_dir)

model = create_model()


history = model.fit(train_generator,
                    epochs=2,
                    verbose=1,
                    validation_data=validation_generator)