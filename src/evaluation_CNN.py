'''
Using Bottleneck Features for Multi-Class CLassification in Keras

Using VGG16 network plus a Fully-connected classifier as a top model of this network

We use this technique to build powerful and high accuracy Image Classificatiion systems with small amount of training data and without overfitting.

Source
https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
https://www.codesofinterest.com/2017/08/bottleneck-features-multi-class-classification-keras.html
'''

import math
import os

import matplotlib.pyplot as plt
import numpy as np
from keras import applications
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.utils.np_utils import to_categorical

from DB import Database

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Only for MacOS 12.0 and higher

if not os.path.exists('cache'):
    os.makedirs('cache')

# dimensions of the images.
img_width, img_height = 120, 80

top_model_weights_path = 'cache/bottleneck_fc_model.h5'
train_data_dir = 'CorelDBDataSet/train'
validation_data_dir = 'CorelDBDataSet/val'

# number of epochs to train top model
epochs = 30
# batch size used by flow_from_directory and predict_generator
batch_size = 16


def save_bottleneck_features():
    ''' Save the bottleneck features from the VGG16 model.
    In this function, we create the VGG16 model without the top model (fully-connected layers).
    '''
    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    datagen = ImageDataGenerator(rescale=1. / 255)

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    nb_train_samples = len(generator.filenames)
    num_classes = len(generator.class_indices)

    predict_size_train = int(math.ceil(nb_train_samples / batch_size))

    bottleneck_features_train = model.predict_generator(
        generator, predict_size_train)

    np.save('cache/bottleneck_features_train.npy',
            bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    nb_validation_samples = len(generator.filenames)

    predict_size_validation = int(
        math.ceil(nb_validation_samples / batch_size))

    bottleneck_features_validation = model.predict_generator(
        generator, predict_size_validation)

    np.save('cache/bottleneck_features_validation.npy',
            bottleneck_features_validation)


def train_top_model():
    ''' Train the top model of the VGG16 network (fully-connected classifier).
    In order to train the top model, we need the class labels for each of the training/validation samples.
    We use a data generator for that also.
    We also need to convert the labels to categorical vectors.
    '''
    datagen_top = ImageDataGenerator(rescale=1. / 255)
    generator_top = datagen_top.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

    nb_train_samples = len(generator_top.filenames)
    num_classes = len(generator_top.class_indices)

    # save the class indices to use use later in predictions
    np.save('cache/class_indices.npy', generator_top.class_indices)

    # load the bottleneck features saved earlier
    train_data = np.load('cache/bottleneck_features_train.npy')

    # get the class lebels for the training data, in the original order
    train_labels = generator_top.classes

    # convert the training labels to categorical vectors
    train_labels = to_categorical(train_labels, num_classes=num_classes)

    generator_top = datagen_top.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    nb_validation_samples = len(generator_top.filenames)

    validation_data = np.load('cache/bottleneck_features_validation.npy')

    validation_labels = generator_top.classes
    validation_labels = to_categorical(
        validation_labels, num_classes=num_classes)

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_data, train_labels,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(validation_data, validation_labels))

    model.save_weights(top_model_weights_path)

    (eval_loss, eval_accuracy) = model.evaluate(
        validation_data, validation_labels, batch_size=batch_size, verbose=1)

    print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))
    print("[INFO] Loss: {}".format(eval_loss))

    # draw the history for accuracy and loss of training and validation
    #

    plt.figure(1)

    # summarize history for accuracy

    plt.subplot(211)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')

    # summarize history for loss

    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


def predict():
    ''' Predict the class of an image.
    We need to run it through the same pipeline as before (run the image through the pre-trained VGG16 model and then run the bottleneck prediction through the trained top model).
    '''
    # load the class_indices saved in the earlier step
    class_dictionary = np.load(
        'cache/class_indices.npy', allow_pickle=True).item()

    num_classes = len(class_dictionary)

    # add the path to your test image below
    dbTest = Database(DB_dir="CorelDBDataSet/test",
                      DB_csv="CorelDBDataSetTest.csv")

    print("[INFO] loading and preprocessing image...")
    for image_path in dbTest.get_data().img:
        image = load_img(image_path, target_size=(img_width, img_height))
        image = img_to_array(image)

        # important! otherwise the predictions will be '0'
        image = image / 255

        image = np.expand_dims(image, axis=0)

        # build the VGG16 network
        model = applications.VGG16(include_top=False, weights='imagenet')

        # get the bottleneck prediction from the pre-trained VGG16 model
        bottleneck_prediction = model.predict(image)

        # build top model
        model = Sequential()
        model.add(Flatten(input_shape=bottleneck_prediction.shape[1:]))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='sigmoid'))

        model.load_weights(top_model_weights_path)

        # use the bottleneck prediction on the top model to get the final classification
        class_predicted = model.predict_classes(bottleneck_prediction)

        inID = class_predicted[0]

        inv_map = {v: k for k, v in class_dictionary.items()}

        label = inv_map[inID]

        probabilities = model.predict_proba(bottleneck_prediction)[0][inID]

        # get the predicted label
        print("Image: {}, Predicted label: {}, Probability: {}".format(
            image_path, label, probabilities))


if __name__ == "__main__":
    save_bottleneck_features()
    train_top_model()
    predict()
