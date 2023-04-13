import tensorflow as tf


class data_set_generator():

    def __init__(self, Train_image):
        self.Train_image = Train_image

    def __call__(self, *args, **kwargs):
        for item in tf.data.Dataset.from_tensor_slices(self.Train_image).batch(1).take(300):
            yield [item]
