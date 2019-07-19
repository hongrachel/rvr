import tensorflow as tf
from tensorflow.keras.layers import Activation


class MLP(object):

    def __init__(self, name, shapes, activ):
        self.name = name
        self.shapes = shapes
        self.weights = self.make_wts_biases()
        self.activ = activ

    def make_wts_biases(self):
        w_dict = {}

        for i in range(len(self.shapes) - 1):
            w_dict[i] = {}
            w = tf.get_variable("{}_w{:d}".format(self.name, i), shape=[self.shapes[i], self.shapes[i + 1]],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable("{}_b{:d}".format(self.name, i), shape=[self.shapes[i + 1]],
                                initializer=tf.contrib.layers.xavier_initializer())
            w_dict[i]['w'] = w
            w_dict[i]['b'] = b
        return w_dict

    def forward(self, x):
        prev_L = x
        num_layers = len(self.weights)
        for layer in range(num_layers - 1):
            L = tf.add(tf.matmul(prev_L, self.weights[layer]['w']), self.weights[layer]['b'])
            if self.activ == 'softplus':
                L = tf.nn.softplus(L)
            elif self.activ == 'sigmoid':
                L = tf.nn.sigmoid(L)
            elif self.activ == 'relu':
                L = tf.nn.relu(L)
            elif self.activ == 'leakyrelu':
                L = tf.nn.leaky_relu(L)
            elif self.activ == 'None':
                pass
            else:
                raise Exception('bad activation function')
            prev_L = L
        L = tf.add(tf.matmul(prev_L, self.weights[num_layers - 1]['w']), self.weights[num_layers - 1]['b'])
        return L

class CNNEncoder(object):
    def __init__(self, name, shapes, activ):
        self.name = name
        self.shapes = shapes
        self.weights = self.make_wts_biases()
        self.activ = activ

    def make_wts_biases(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Conv2D(32, (4, 4), strides=(2, 2), padding='same'))
        model.add(tf.keras.layers.BatchNormalization(axis=-1))
        #model.add(tf.keras.layers.ReLU())
        model.add(Activation('relu'))
        #model.add(tf.keras.layers.Dropout(0.3))

        model.add(tf.keras.layers.Conv2D(32, (4, 4), strides=(2, 2), padding='same'))
        model.add(tf.keras.layers.BatchNormalization(axis=-1))
        #model.add(tf.keras.layers.ReLU())
        model.add(Activation('relu'))
        #model.add(tf.keras.layers.Dropout(0.3))


        model.add(tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same'))
        model.add(tf.keras.layers.BatchNormalization(axis=-1))
        # model.add(tf.keras.layers.ReLU())
        model.add(Activation('relu'))

        model.add(tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same'))
        model.add(tf.keras.layers.BatchNormalization(axis=-1))
        model.add(Activation('relu'))
        # model.add(tf.keras.layers.ReLU())

        model.add(tf.keras.layers.Conv2D(512, (4, 4)) )
        model.add(tf.keras.layers.BatchNormalization(axis=-1))
        # model.add(tf.keras.layers.ReLU())
        model.add(Activation('relu'))

        model.add(tf.keras.layers.Conv2D(self.shapes[-1], (1,1)) )


        #model.add(tf.keras.layers.Flatten())
        #model.add(tf.keras.layers.Dense(1))

        return model

    def forward(self, x):
        x = tf.reshape(x, [-1, 64, 64, 1])
        out = self.weights(x)
        return tf.reshape(out, [-1, 10])

class CNNDecoder(object):
    def __init__(self, name, shapes, activ):
        self.name = name
        self.shapes = shapes
        self.weights = self.make_wts_biases()
        self.activ = activ

    def make_wts_biases(self):
        model = tf.keras.Sequential()

        # model.add(tf.image.resize_nearest_neighbor(size=()))
        # model.add(tf.keras.layers.Dense(4*4))
        # model.add(tf.keras.layers.BatchNormalization())
        # model.add(Activation('relu'))

        # model.add(tf.keras.layers.Reshape((4, 4, 1)))
        # assert model.output_shape == (None, 4, 4, 1)

        model.add(tf.keras.layers.Conv2DTranspose(512, 1, 1, padding='valid'))
        model.add(tf.keras.layers.BatchNormalization(axis=-1))
        model.add(Activation('relu'))


        model.add(tf.keras.layers.Conv2DTranspose(64, 4, 1, padding='valid'))
        model.add(tf.keras.layers.BatchNormalization(axis=-1))
        model.add(Activation('relu'))

        model.add(tf.keras.layers.Conv2DTranspose(64, 4, 2, padding='same'))
        model.add(tf.keras.layers.BatchNormalization(axis=-1))
        model.add(Activation('relu'))

        model.add(tf.keras.layers.Conv2DTranspose(32, 4, 2, padding='same'))
        model.add(tf.keras.layers.BatchNormalization(axis=-1))
        model.add(Activation('relu'))

        model.add(tf.keras.layers.Conv2DTranspose(32, 4, 2, padding='same'))
        model.add(tf.keras.layers.BatchNormalization(axis=-1))
        model.add(Activation('relu'))

        model.add(tf.keras.layers.Conv2DTranspose(1, 4, 2, padding='same'))
        #model.add(Activation('sigmoid'))

        return model

    def forward(self, x):
        x = tf.reshape(x, [-1, 1, 1, self.shapes[0]])
        out = self.weights(x)
        return tf.reshape(out, [-1, 4096])

