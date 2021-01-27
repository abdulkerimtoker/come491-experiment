from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import Input


def my_model(num_classes):
    model = Sequential([
        Dense(1024, activation='relu', input_shape=(2622,)),
        Dropout(0.5),
        Dense(2048, activation='relu'),
        Dropout(0.5),
        Dense(2048, activation='relu'),
        Dropout(0.5),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    return model


def transformer():
    from vgg_model import vgg_face
    vgg_model = vgg_face()
    input_x = Input((224, 224, 3))
    input_y = Input((1,))
    feature_vector = vgg_model(input_x)
    model = Model(inputs=(input_x, input_y), outputs=(feature_vector, input_y))
    return model
