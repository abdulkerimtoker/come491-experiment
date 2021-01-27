import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Input
from vgg_model import vgg_face
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from datetime import datetime
import cv2

face_cascade = cv2.CascadeClassifier('C:\\Users\\toker\\PycharmProjects\\tensor\\frontal.xml')


def cv_load_img(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x, y, w, h = face_cascade.detectMultiScale(gray, 1.1, 4)[0]
    cropped = img[y:y+h, x:x+w]
    cv2.imwrite('D:\\write.jpg', cropped)


def get_siamese():
    siamese_net = None
    input_shape = [224, 224, 3]

    with tf.device('/CPU:0'):
        input_1 = Input(input_shape)
        input_2 = Input(input_shape)

        vgg_model = vgg_face()

        similarity = tf.keras.losses.cosine_similarity(vgg_model(input_1), vgg_model(input_2), axis=1)

        siamese_net = tf.keras.models.Model(inputs=[input_1, input_2], outputs=similarity)


def calculate_similarity(name_1, name_2):
    with tf.device('/CPU:0'):
        base_dir = 'C:\\Users\\toker\\Pictures\\Camera Roll\\'

        cv_load_img(base_dir + name_1)
        img_1 = load_img('D:\\write.jpg', target_size=(224, 224))
        img_1 = img_to_array(img_1)
        img_1 = preprocess_input(img_1)

        cv_load_img(base_dir + name_2)
        img_2 = load_img('D:\\write.jpg', target_size=(224, 224))
        img_2 = img_to_array(img_2)
        img_2 = preprocess_input(img_2)

        return siamese_net([np.array([img_1]), np.array([img_2])])


def calculate_similarities(name_1, names):
    with tf.device('/CPU:0'):
        base_dir = 'C:\\Users\\toker\\Pictures\\Camera Roll\\'

        time = datetime.now()

        cv_load_img(base_dir + name_1)
        img = load_img('D:\\write.jpg', target_size=(224, 224))
        img = img_to_array(img)
        img = preprocess_input(img)

        images = []
        for name in names:
            cv_load_img(base_dir + name)
            img_2 = load_img('D:\\write.jpg', target_size=(224, 224))
            img_2 = img_to_array(img_2)
            img_2 = preprocess_input(img_2)
            images.append(img_2)

        time = datetime.now() - time
        print('Preprocessing: ' + str(time.total_seconds()))

        return siamese_net.predict([np.array([img for _ in range(len(names))]), np.array(images)], batch_size=10)


def get_similarity(name_1, name_2):
    time = datetime.now()
    calculated_similarity = calculate_similarity(name_1, name_2)
    time = datetime.now() - time
    print('Total: ' + str(time.total_seconds()))
    return calculated_similarity


def get_similarities(name_1, *names):
    time = datetime.now()
    calculated_similarities = calculate_similarities(name_1, names)
    time = datetime.now() - time
    print('Total: ' + str(time.total_seconds()))
    return calculated_similarities
