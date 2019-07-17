import tensorflow as tf


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
        model.add(tf.keras.layers.ReLU())
        #model.add(tf.keras.layers.Dropout(0.3))

        model.add(tf.keras.layers.Conv2D(32, (4, 4), strides=(2, 2), padding='same'))
        model.add(tf.keras.layers.BatchNormalization(axis=-1))
        model.add(tf.keras.layers.ReLU())
        #model.add(tf.keras.layers.Dropout(0.3))


        model.add(tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same'))
        model.add(tf.keras.layers.BatchNormalization(axis=-1))
        model.add(tf.keras.layers.ReLU())

        model.add(tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same'))
        model.add(tf.keras.layers.BatchNormalization(axis=-1))
        model.add(tf.keras.layers.ReLU())

        model.add(tf.keras.layers.Conv2D(512, (4, 4)) )
        model.add(tf.keras.layers.BatchNormalization(axis=-1))
        model.add(tf.keras.layers.ReLU())

        model.add(tf.keras.layers.Conv2D(self.shapes[-1], (1,1)) )


        #model.add(tf.keras.layers.Flatten())
        #model.add(tf.keras.layers.Dense(1))

        return model

    def forward(self, x):
        self.weights(x)

class CNNDecoder(object):
    def __init__(self, name, shapes, activ):
        self.name = name
        self.shapes = shapes
        self.weights = self.make_wts_biases()
        self.activ = activ

    def make_wts_biases(self):
        #Todo

    def forward(self, x):
        #Todo

