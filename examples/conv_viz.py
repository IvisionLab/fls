#!/usr/bin/python3

import keras
import keras.backend as K
import keras.layers as KL
import model as M
import numpy as np
import gemini
import h5py
import re
import cv2
import os
from keras.engine import saving
from keras.preprocessing.image import save_img
from gemini import InferenceConfig


MEAN_PIXEL = np.array([123.7, 116.8, 103.9])

def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# def deprocess_image(x):
#     # Util function to convert a tensor into a valid image.
#     if K.image_data_format() == 'channels_first':
#         x = x.reshape((3, x.shape[2], x.shape[3]))
#         x = x.transpose((1, 2, 0))
#     else:
#         x = x.reshape((x.shape[1], x.shape[2], 3))
#     x /= 2.
#     x += 0.5
#     x *= 255.
#     x = np.clip(x, 0, 255).astype('uint8')
#     return x

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())

annotations = gemini.load_dataset_json("gemini_annotations_test.json", shuffle=False)
results = []

model_path = 'logs/gemini20180702T1433/mask_rcnn_gemini_0160.h5'
config = InferenceConfig()
model = M.MaskRCNN(mode="inference", config=config, model_dir=gemini.DEFAULT_LOGS_DIR)
model.load_weights(model_path, by_name=True)

for l in model.keras_model.layers:
    if re.match("res.*", l.name):
        print("{0},{1},{2}".format(l, l.name, l.output))

layer_name = "res2c_out"
layer_dict = dict([(layer.name, layer) for layer in model.keras_model.layers[1:]])

img_width = config.IMAGE_MAX_DIM
img_height = config.IMAGE_MAX_DIM


input_img = model.keras_model.input[0]

for item in annotations:
    image_path = os.path.join(item['basefolder'], item['filepath'])
    image = cv2.imread(image_path)

    # Mold inputs to format expected by the neural network
    molded_images, image_metas, windows = model.mold_inputs([image])

    input_img_data = molded_images

    kept_filters = []

    for filter_index in range(100):
        print('Processing filter %d' % filter_index)
        # start_time = time.time()

        # we build a loss function that maximizes the activation
        # of the nth filter of the layer considered
        layer_output = layer_dict[layer_name].output

        loss = K.mean(layer_output[:, :, :, filter_index])

        # we compute the gradient of the input picture wrt this loss
        grads = K.gradients(loss, input_img)[0]

        # normalization trick: we normalize the gradient
        grads = normalize(grads)

        # this function returns the loss and grads given the input picture
        iterate = K.function([input_img], [loss, grads])

        step = 1.

        loss_value, grads_value = iterate([input_img_data])
        out = deprocess_image(grads_value[0])

        cv2.imshow("", out)
        cv2.waitKey(50)


