# image preprocessing imports
import os
import shutil
from keras.preprocessing.image import ImageDataGenerator
# neuronal network imports
from keras import layers
from keras import models
from keras import optimizers
# graph plot import
import matplotlib.pyplot as plt
# testing model import
from keras.models import load_model
from keras.preprocessing import image

import tensorflow as tf

# TODO: test with other model architectures like VGG16
# from keras.applications import VGG16


base_dir = './images'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')


# TODO: get more data for model -> provide overfitting
def prepare_data():
    # init and make directories for training, validation and testing
    src_dir_turtle = './downloads/turtles'
    src_dir_no_turtle = './downloads/no_turtles'

    os.mkdir(base_dir)
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')
    test_dir = os.path.join(base_dir, 'test')
    os.mkdir(train_dir)
    os.mkdir(validation_dir)
    os.mkdir(test_dir)

    # create train subfolders for turtles and no_turtles
    train_turtle_dir = os.path.join(train_dir, 'turtles')
    train_no_turtle_dir = os.path.join(train_dir, 'no_turtles')
    os.mkdir(train_turtle_dir)
    os.mkdir(train_no_turtle_dir)

    # create validation subfolders for turtles and no_turtles
    validation_turtle_dir = os.path.join(validation_dir, 'turtles')
    validation_no_turtle_dir = os.path.join(validation_dir, 'no_turtles')
    os.mkdir(validation_turtle_dir)
    os.mkdir(validation_no_turtle_dir)

    # create test subfolders for turtles and no_turtles
    test_turtle_dir = os.path.join(test_dir, 'turtles')
    test_no_turtle_dir = os.path.join(test_dir, 'no_turtles')
    os.mkdir(test_turtle_dir)
    os.mkdir(test_no_turtle_dir)

    # TODO: auto rename files in src dir
    # rename all files in src folder
    # fnames = ['turtle.{}.jpg'.format(i) for i in range(len(os.listdir(src_dir_turtle))]
    # for fname in fnames:
    #     os.listdir(src_dir_turtle)[0]

    # Copy images to train_turtles_dir
    fnames = ['turtle.{}.jpg'.format(i) for i in range(10)]
    for fname in fnames:
        src = os.path.join(src_dir_turtle, fname)
        dst = os.path.join(train_turtle_dir, fname)
        shutil.copyfile(src, dst)
    fnames = ['no_turtle.{}.jpg'.format(i) for i in range(10)]
    for fname in fnames:
        src = os.path.join(src_dir_no_turtle, fname)
        dst = os.path.join(train_no_turtle_dir, fname)
        shutil.copyfile(src, dst)
    print("train data copied")

    # Copy images to validation_turtles_dir
    fnames = ['turtle.{}.jpg'.format(i) for i in range(10, 15)]
    for fname in fnames:
        src = os.path.join(src_dir_turtle, fname)
        dst = os.path.join(validation_turtle_dir, fname)
        shutil.copyfile(src, dst)
    fnames = ['no_turtle.{}.jpg'.format(i) for i in range(10, 15)]
    for fname in fnames:
        src = os.path.join(src_dir_no_turtle, fname)
        dst = os.path.join(validation_no_turtle_dir, fname)
        shutil.copyfile(src, dst)
    print("validation data copied")

    # Copy images to test_turtles_dir
    fnames = ['turtle.{}.jpg'.format(i) for i in range(15, 20)]
    for fname in fnames:
        src = os.path.join(src_dir_turtle, fname)
        dst = os.path.join(test_turtle_dir, fname)
        shutil.copyfile(src, dst)
    fnames = ['no_turtle.{}.jpg'.format(i) for i in range(15, 20)]
    for fname in fnames:
        src = os.path.join(src_dir_no_turtle, fname)
        dst = os.path.join(test_no_turtle_dir, fname)
        shutil.copyfile(src, dst)
    print("test data copied")
    return


# TODO: may remove this
# check if a gpu is available for tensorflow
def check_for_gpu():
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)
    # Creates a session with log_device_placement set to True.
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    # Runs the op.
    print(sess.run(c))
    return


def create_small_network():
    # create convolutional model
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    # create classifier
    model.add(layers.Flatten())  # convert image matrix to vector
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.summary()

    # optimize model for images
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=['acc'])
    return model


def preprocess_images():
    # All images will be rescaled by 1./255
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(150, 150),
        batch_size=20,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

    # check
    for data_batch, labels_batch in train_generator:
        print('data batch shape:', data_batch.shape)
        print('labels batch shape:', labels_batch.shape)
        break

    return train_generator, validation_generator


# preprocess images with augmentation -> provide overfitting
def preprocess_images_with_augmentation():
    # All images will be rescaled by 1./255
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    # TODO: bearbeite images
    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(150, 150),
        batch_size=20,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

    # check
    for data_batch, labels_batch in train_generator:
        print('data batch shape:', data_batch.shape)
        print('labels batch shape:', labels_batch.shape)
        break

    return train_generator, validation_generator


def train_model(train_generator, validation_generator):
    model = create_small_network()
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=5,
        validation_data=validation_generator,
        validation_steps=50)

    return model, history


# plot history data from model training
def plot_history_graph(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()
    return


def predict(filename, model):
    file_dir = os.path.join(test_dir, filename)
    img = image.load_img(file_dir, target_size=(150, 150))
    x = image.img_to_array(img)
    x = x.reshape(1, 150, 150, 3)

    classes = model.predict(x, batch_size=10)
    print(classes)
    names = {0: 'turtle', 1: 'no turtle'}
    print(filename, names.get(round(classes[0][0], 0), 'not defined'))
    plt.imshow(img)
    plt.show()
    return


# prepare_data()
# train_generator, validation_generator = preprocess_images()
# model, history = train_model(train_generator, validation_generator)
# plot_history_graph(history)
# model.save('trained_turtle_model.h5')

train_generator, validation_generator = preprocess_images_with_augmentation()
model, history = train_model(train_generator, validation_generator)
plot_history_graph(history)
model.save('trained_turtle_model2.h5')

# TODO: test CNN with test data
# trained_model = load_model('trained_turtle_model2.h5')
# predict(test1, trained_model)

# TODO: create test statistics
