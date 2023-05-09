
import os
import random
from shutil import copyfile

root_dir = 'data'
source_dir = 'PokemonData'

def create_train_val_dirs(root_dir, source_dir):
    for p in os.listdir(source_dir):
        os.makedirs(os.path.join(root_dir, 'training/'+p))
        os.makedirs(os.path.join(root_dir, 'validation/'+p))


try:
    create_train_val_dirs(root_dir, source_dir)
except FileExistsError:
    print("You should not be seeing this since the upper directory is removed beforehand")


def split_data(source_dir, training_dir, validation_dir, split_size):
    files = []
    for filename in os.listdir(source_dir):
        file = source_dir + filename
        if os.path.getsize(file)> 0:
            files.append(filename)
        else:
            print(filename + ' is zero length, so ignoring.')
        training_length = int(len(files)* split_size)
        testing_length = int(len(files) - training_length)
        shuffled_set = random.sample(files, len(files))
        training_set = shuffled_set[0: training_length]
        testing_set = shuffled_set[-testing_length:]
    for filename in training_set:
        src_file = source_dir + filename
        dest_file = training_dir + filename
        copyfile(src_file, dest_file)
    for filename in testing_set:
        src_file = source_dir + filename
        dest_file = validation_dir + filename
        copyfile(src_file, dest_file)



# Empty directories in case you run this cell multiple times
for p in os.listdir(source_dir):
    training_dir =os.path.join(root_dir, 'training/'+p)
    val_dir = os.path.join(root_dir, 'validation/'+p)
    if len(os.listdir(training_dir)) > 0:
        for file in os.scandir(training_dir):
            os.remove(file.path)
    if len(os.listdir(val_dir)) > 0:
        for file in os.scandir(val_dir):
            os.remove(file.path)


# Define proportion of images used for training
split_size = .9

# Run the function
for p in os.listdir(source_dir):
    print(p)
    split_data('PokemonData/'+p+"/",'data/training/'+p+"/", 'data/validation/'+p+"/", split_size)

