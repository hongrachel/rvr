import tensorflow as tf
#from tensorflow.keras.layers import Activation
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Flatten, Reshape, Input, concatenate
from tensorflow.keras.layers import Conv2D, UpSampling2D
from tensorflow.keras.layers import LeakyReLU, Dropout
from tensorflow.keras.layers import BatchNormalization


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
        print ("NUM LAYERS: " + str(num_layers))
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

class MnistCnnEncoder(object):
    def __init__(self, name, shapes, activ, adim):
        self.name = name
        self.shapes = shapes
        self.activ = activ
        self.adim = adim
        self.weights = self.make_wts_biases()

    def make_wts_biases(self):
        sequence = Input(shape=(28, 28, 3), name='Sequence')
        if self.adim != 0:
            features = Input(shape=(self.adim,), name='Features')  # self.adim

        net = Sequential()
        input_shape = (28, 28, 3)
        dropout_prob = 0.4

        net.add(Conv2D(64, 5, strides=2, input_shape=input_shape, padding='same'))
        net.add(LeakyReLU())
        
        net.add(Conv2D(128, 5, strides=2, padding='same'))
        net.add(LeakyReLU())
        net.add(Dropout(dropout_prob))
        
        # net.add(Conv2D(256, 5, strides=2, padding='same'))
        # net.add(LeakyReLU())
        # net.add(Dropout(dropout_prob))
        
        # net.add(Conv2D(512, 5, strides=2, padding='same'))
        # net.add(LeakyReLU())
        # net.add(Dropout(dropout_prob))
        
        net.add(Flatten())
        
        part1 = net(sequence)
        if self.adim != 0:  # feed A into encoder
            merged = concatenate([part1, features])
        else:
            merged = part1

        final = Dense(self.shapes[-1])(merged)
        if self.adim != 0:
            model = Model(inputs=[sequence, features], outputs=[final])
        else:
            model = Model(inputs=[sequence], outputs=[final])

        return model

        # old model --------

        # net = Sequential()
        # input_shape = (28, 28, 3)
        # dropout_prob = 0.4
        # # experiment.log_parameter('dis_dropout_prob', dropout_prob)

        # net.add(Conv2D(64, 5, strides=2, input_shape=input_shape, padding='same'))
        # net.add(LeakyReLU())

        # net.add(Conv2D(128, 5, strides=2, padding='same'))
        # net.add(LeakyReLU())
        # net.add(Dropout(dropout_prob))

        # # net.add(Conv2D(256, 5, strides=2, padding='same'))
        # # net.add(LeakyReLU())
        # # net.add(Dropout(dropout_prob))

        # # net.add(Conv2D(512, 5, strides=1, padding='same'))
        # # net.add(LeakyReLU())
        # # net.add(Dropout(dropout_prob))

        # net.add(Flatten())
        # net.add(Dense(self.shapes[-1]))  # final dim
        # #net.add(Activation('sigmoid'))

        # return net

    def forward(self, x, a):
        x = tf.reshape(x, [-1, 28, 28, 3])
        a = tf.reshape(a, [-1, self.adim,])
        out = self.weights([x, a])
        return tf.reshape(out, [-1, self.shapes[-1]])

class MnistCnnDecoder(object):
    def __init__(self, name, shapes, activ, a):
        self.name = name
        self.shapes = shapes
        self.weights = self.make_wts_biases()
        self.activ = activ
        self.a = a

    def make_wts_biases(self):
        net = Sequential()
        dropout_prob = 0.4
        #experiment.log_parameter('adv_dropout_prob', dropout_prob)

        net.add(Dense(7 * 7 * 256, input_dim=self.shapes[0]))
        net.add(BatchNormalization(momentum=0.9))
        net.add(Activation('relu'))
        net.add(Reshape((7, 7, 256)))
        net.add(Dropout(dropout_prob))

        net.add(UpSampling2D())
        net.add(Conv2D(128, 5, padding='same'))
        net.add(BatchNormalization(momentum=0.9))
        net.add(Activation('relu'))

        net.add(UpSampling2D())
        net.add(Conv2D(64, 5, padding='same'))
        net.add(BatchNormalization(momentum=0.9))
        net.add(Activation('relu'))

        net.add(Conv2D(32, 5, padding='same'))
        net.add(BatchNormalization(momentum=0.9))
        net.add(Activation('relu'))

        net.add(Conv2D(3, 5, padding='same'))
        #net.add(Activation('sigmoid'))

        return net

    def forward(self, x):
        #x = tf.reshape(x, [-1, 1, 1, self.shapes[0]])
        out = self.weights(x)
        print('===================decoding\n\n============here')
        out_flatten = tf.reshape(out, [-1])
        # a_flatten = tf.reshape(self.a, [-1])
        # X_and_A = tf.concat((out_flatten, a_flatten), axis=0)
        # return tf.reshape(X_and_A, [-1, 28*28*3+3])
        
        # no A:
        return tf.reshape(out, [-1, self.shapes[-1]])

class PacsCnnEncoder(object):
    def __init__(self, name, shapes, activ, adim):
        self.name = name
        self.shapes = shapes
        self.activ = activ
        self.adim = adim
        self.weights = self.make_wts_biases()

    def make_wts_biases(self):
        sequence = Input(shape=(64, 64, 3), name='Sequence')
        if self.adim != 0:
            features = Input(shape=(self.adim,), name='Features')  # self.adim

        net = Sequential()
        input_shape = (64, 64, 3)
        dropout_prob = 0.4

        net.add(Conv2D(64, 5, strides=2, input_shape=input_shape, padding='same'))
        net.add(LeakyReLU())
        
        net.add(Conv2D(128, 5, strides=2, padding='same'))
        net.add(LeakyReLU())
        net.add(Dropout(dropout_prob))
        
        net.add(Conv2D(256, 5, strides=2, padding='same'))
        net.add(LeakyReLU())
        net.add(Dropout(dropout_prob))
        
        net.add(Conv2D(512, 5, strides=2, padding='same'))
        net.add(LeakyReLU())
        net.add(Dropout(dropout_prob))
        
        net.add(Flatten())
        
        part1 = net(sequence)
        if self.adim != 0:  # feed A into encoder
            merged = concatenate([part1, features])
        else:
            merged = part1

        final = Dense(self.shapes[-1])(merged)
        if self.adim != 0:
            model = Model(inputs=[sequence, features], outputs=[final])
        else:
            model = Model(inputs=[sequence], outputs=[final])

        return model

        # old model --------

        # net = Sequential()
        # input_shape = (28, 28, 3)
        # dropout_prob = 0.4
        # # experiment.log_parameter('dis_dropout_prob', dropout_prob)

        # net.add(Conv2D(64, 5, strides=2, input_shape=input_shape, padding='same'))
        # net.add(LeakyReLU())

        # net.add(Conv2D(128, 5, strides=2, padding='same'))
        # net.add(LeakyReLU())
        # net.add(Dropout(dropout_prob))

        # # net.add(Conv2D(256, 5, strides=2, padding='same'))
        # # net.add(LeakyReLU())
        # # net.add(Dropout(dropout_prob))

        # # net.add(Conv2D(512, 5, strides=1, padding='same'))
        # # net.add(LeakyReLU())
        # # net.add(Dropout(dropout_prob))

        # net.add(Flatten())
        # net.add(Dense(self.shapes[-1]))  # final dim
        # #net.add(Activation('sigmoid'))

        # return net

    def forward(self, x, a):
        x = tf.reshape(x, [-1, 64, 64, 3])
        a = tf.reshape(a, [-1, self.adim,])
        out = self.weights([x, a])
        return tf.reshape(out, [-1, self.shapes[-1]])

class PacsCnnDecoder(object):
    def __init__(self, name, shapes, activ, a):
        self.name = name
        self.shapes = shapes
        self.weights = self.make_wts_biases()
        self.activ = activ
        self.a = a

    def make_wts_biases(self):
        net = Sequential()
        dropout_prob = 0.4
        #experiment.log_parameter('adv_dropout_prob', dropout_prob)

        net.add(Dense(8 * 8 * 256, input_dim=self.shapes[0]))
        net.add(BatchNormalization(momentum=0.9))
        net.add(Activation('relu'))
        net.add(Reshape((8, 8, 256)))
        net.add(Dropout(dropout_prob))

        net.add(UpSampling2D())
        net.add(Conv2D(128, 5, padding='same'))
        net.add(BatchNormalization(momentum=0.9))
        net.add(Activation('relu'))

        net.add(UpSampling2D())
        net.add(Conv2D(128, 5, padding='same'))
        net.add(BatchNormalization(momentum=0.9))
        net.add(Activation('relu'))

        net.add(UpSampling2D())
        net.add(Conv2D(64, 5, padding='same'))
        net.add(BatchNormalization(momentum=0.9))
        net.add(Activation('relu'))

        net.add(Conv2D(32, 5, padding='same'))
        net.add(BatchNormalization(momentum=0.9))
        net.add(Activation('relu'))

        net.add(Conv2D(3, 5, padding='same'))
        #net.add(Activation('sigmoid'))

        return net

    def forward(self, x):
        #x = tf.reshape(x, [-1, 1, 1, self.shapes[0]])
        out = self.weights(x)
        print('===================decoding\n\n============here')
        out_flatten = tf.reshape(out, [-1])
        # a_flatten = tf.reshape(self.a, [-1])
        # X_and_A = tf.concat((out_flatten, a_flatten), axis=0)
        # return tf.reshape(X_and_A, [-1, 28*28*3+3])
        
        # no A:
        return tf.reshape(out, [-1, self.shapes[-1]])
