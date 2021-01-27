import numpy as np
import os
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

from main import cv_load_img
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

from models import my_model, transformer

import json


def get(path):
    try:
        cv_load_img(path)
    except:
        return None
    img = load_img('D:\\write.jpg', target_size=(224, 224))
    img = img_to_array(img)
    img = preprocess_input(img)
    return img


def load(path, _name_to_label=None, write=False):
    folders = os.listdir(path)
    x = []
    y = []
    names = set()

    for folder in folders:
        imgs = os.listdir(path + '/' + folder)
        print(folder)
        for img in imgs:
            face = get(path + '/' + folder + '/' + img)
            if face is not None:
                x.append(face)
                y.append(folder)
                names.add(folder)

    if _name_to_label is None:
        _name_to_label = {v: i for i, v in enumerate(names)}
    _label_to_name = {v: k for k, v in _name_to_label.items()}

    if write:
        file = open('data/label_to_name.json', 'w')
        file.write(json.dumps(_label_to_name))
        file.close()

        file = open('data/name_to_label.json', 'w')
        file.write(json.dumps(_name_to_label))
        file.close()

    y = [_name_to_label[folder] for folder in y]

    return np.asarray(x), np.asarray(y), _label_to_name, _name_to_label


def load_registered(path, name_to_label):
    folders = os.listdir(path)
    for i, folder in enumerate(folders):
        if folder not in name_to_label:
            name_to_label[folder] = len(name_to_label)
    return load(path, name_to_label, write=True)


with tf.device('/CPU:0'):
    vgg_transformer = transformer()


def transform_x(x, y):
    with tf.device('/CPU:0'):
        x_transformed, y_transformed = vgg_transformer((x, y))
        x_transformed = x_transformed.numpy()
        y_transformed = y_transformed.numpy().astype(np.int32).reshape((len(y_transformed),))
        return x_transformed, y_transformed


def build_model(num_classes):
    model = my_model(num_classes)
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                  metrics=['accuracy'])
    return model


def predict(model, path, label_to_name):
    transformed, _ = transform_x(np.asarray([get(path)]), np.asarray([0]))
    predictions = model.predict(np.asarray([transformed]))
    score = predictions[0]
    return label_to_name[np.argmax(score)], np.max(score) * 100


def train():
    name_to_label = json.loads(open('data/name_to_label.json').read())

    x = np.load('data/x_transformed.npy')
    y = np.load('data/y_transformed.npy')
    x_test = np.load('data/x_test_transformed.npy')
    y_test = np.load('data/y_test_transformed.npy')

    x_registered, y_registered, label_to_name, name_to_label = load_registered('D:/registered', name_to_label)
    x_registered, y_registered = transform_x(x_registered, y_registered)
    x_registered_test, y_registered_test, _, _ = load_registered('D:/registered_test', name_to_label)
    x_registered_test, y_registered_test = transform_x(x_registered_test, y_registered_test)

    x = np.concatenate([x, x_registered])
    y = np.concatenate([y, y_registered])
    x_test = np.concatenate([x_test, x_registered_test])
    y_test = np.concatenate([y_test, y_registered_test])

    registered_labels = np.unique(y_registered)
    class_weights = {label: 5. if label in registered_labels else 1. for label in np.unique(y)}
    batch_size = 32
    epochs = 250

    with tf.device('/GPU:0'):
        model = build_model(len(np.unique(y)))
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=25)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.25)
        model.fit(x, y, batch_size=batch_size, epochs=epochs, class_weight=class_weights,
                  validation_data=(x_test, y_test), validation_batch_size=batch_size,
                  callbacks=[early_stop, reduce_lr])
        return model, label_to_name, name_to_label


def load_trained_model():
    model = load_model('models/model')
    name_to_label = json.loads(open('data/name_to_label.json').read())
    label_to_name = {v: k for k, v in name_to_label.items()}
    return model, label_to_name, name_to_label
