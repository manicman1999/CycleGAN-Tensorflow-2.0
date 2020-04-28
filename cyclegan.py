from PIL import Image
from math import floor, log2
import numpy as np
import time
from functools import partial
from random import random, randint
import os

from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.initializers import *
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.utils import to_categorical

from datagen import dataGenerator, printProgressBar

im_size = 128
latent_size = 512
BATCH_SIZE = 8
directory = ["horse", "zebra"]

cha = 16

mixed_prob = 0.9

def noise(n):
    return np.random.normal(0.0, 1.0, size = [n, latent_size]).astype('float32')

def noiseList(n):
    return [noise(n)] * n_layers

def mixedList(n):
    tt = int(random() * n_layers)
    p1 = [noise(n)] * tt
    p2 = [noise(n)] * (n_layers - tt)
    return p1 + [] + p2


#Loss functions
def gradient_penalty(samples, output, k = 0.1):
    gradients = K.gradients(output, samples)[0]
    #[batch, 32, 32, 3]
    gradients_sqr = K.square(gradients)
    #[batch, 32, 32, 3]
    gradient_penalty = K.sqrt(K.sum(gradients_sqr, axis=[1,2,3]))
    #[batch]
    gradient_penalty = K.maximum(gradient_penalty - k, K.zeros_like(gradient_penalty))
    #1
    return K.mean(gradient_penalty)

def hinge_d(y_true, y_pred):
    return K.mean(K.relu(1.0 + (y_true * y_pred)))

def w_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


#Lambdas
def crop_to_fit(x):

    height = x[1].shape[1]
    width = x[1].shape[2]

    return x[0][:, :height, :width, :]

def upsample(x):
    return K.resize_images(x,2,2,"channels_last",interpolation='bilinear')


#Blocks
def conv_block(inp, fil):

    out = Conv2D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal')(inp)
    out = LeakyReLU(0.2)(out)
    out = Conv2D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal')(out)
    out = LeakyReLU(0.2)(out)

    return out


def d_block(inp, fil, p = True):

    res = Conv2D(fil, 1, kernel_initializer = 'he_normal')(inp)

    out = Conv2D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal')(inp)
    out = LeakyReLU(0.2)(out)
    out = Conv2D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal')(out)
    out = LeakyReLU(0.2)(out)

    out = add([res, out])

    if p:
        out = AveragePooling2D()(out)

    return out





class GAN(object):

    def __init__(self, steps = 1, lr = 0.0001, decay = 0.00001):

        #Models
        self.D = None
        self.G1 = None
        self.G2 = None

        self.GE = None

        #Config
        self.LR = lr
        self.steps = steps
        self.beta = 0.999

        #Init Models
        self.discriminator()
        self.generator()

        self.GM1 = Adam(lr = self.LR, beta_1 = 0, beta_2 = 0.99)
        self.GM2 = Adam(lr = self.LR, beta_1 = 0, beta_2 = 0.99)
        self.DMO = Adam(lr = self.LR * 2, beta_1 = 0, beta_2 = 0.99)

        self.GE = clone_model(self.G1)
        self.GE.set_weights(self.G1.get_weights())

        self.G2 = clone_model(self.G1)

    def discriminator(self):

        if self.D:
            return self.D

        inp = Input(shape = [None, None, 3])


        x = d_block(inp, 1 * cha)   #128
        x = d_block(x, 2 * cha)   #64
        x = d_block(x, 4 * cha)   #32
        #x = d_block(x, 6 * cha)  #16
        x = d_block(x, 8 * cha)  #8
        x = d_block(x, 16 * cha, p = False)  #4

        x = GlobalAveragePooling2D()(x)

        x = Dense(1, kernel_initializer = 'he_normal')(x)

        self.D = Model(inputs = inp, outputs = x)

        return self.D

    def generator(self):

        if self.G1:
            return self.G1

        # === Generator ===

        #Inputs
        inp_image = Input([None, None, 3])

        #Actual Model
        x = conv_block(inp_image, cha)
        x = AveragePooling2D()(x)
        x = conv_block(x, cha*2)
        x = AveragePooling2D()(x)
        x1 = Conv2D(cha*4, 1, kernel_initializer = 'he_normal')(x)

        x = conv_block(x, cha*4)
        x = conv_block(x, cha*4)
        x = conv_block(x, cha*4)
        x = add([x1, x])

        x = UpSampling2D()(x)
        x = conv_block(x, cha*2)
        x = UpSampling2D()(x)
        x = conv_block(x, cha*1)

        x = Conv2D(3, 1, padding='same', kernel_initializer = 'he_normal')(x)

        self.G1 = Model(inputs = inp_image, outputs = x)

        return self.G1

    def EMA(self):

        #Parameter Averaging

        for i in range(len(self.G1.layers)):
            up_weight = self.G1.layers[i].get_weights()
            old_weight = self.GE.layers[i].get_weights()
            new_weight = []
            for j in range(len(up_weight)):
                new_weight.append(old_weight[j] * self.beta + (1-self.beta) * up_weight[j])
            self.GE.layers[i].set_weights(new_weight)

    def MAinit(self):
        #Reset Parameter Averaging
        self.GE.set_weights(self.G1.get_weights())






class StarGAN(object):

    def __init__(self, steps = 1, lr = 0.0001, decay = 0.00001, silent = True):

        #Init GAN and Eval Models
        self.GAN = GAN(steps = steps, lr = lr, decay = decay)

        self.GAN.G1.summary()

        #Data generator (my own code, not from TF 2.0)
        self.im = []
        for i in range(len(directory)):
            self.im.append(dataGenerator(directory[i], im_size))

        #Set up variables
        self.lastblip = time.clock()

        self.silent = silent

        self.av = np.zeros([44])

    def train(self):

        source_class = 0
        target_class = 1

        source_images = self.im[source_class].get_batch(BATCH_SIZE).astype('float32')
        target_images = self.im[target_class].get_batch(BATCH_SIZE).astype('float32')

        apply_gradient_penalty = self.GAN.steps % 4 == 0

        a, b, c = self.train_step(source_images, target_images, apply_gradient_penalty)

        #Adjust path length penalty mean
        #d = pl_mean when no penalty is applied
        if self.GAN.steps % 10 == 0 and self.GAN.steps > 20000:
            self.GAN.EMA()

        if self.GAN.steps <= 25000 and self.GAN.steps % 1000 == 2:
            self.GAN.MAinit()

        if np.isnan(a):
            print("NaN Value Error.")
            exit()


        #Print info
        if self.GAN.steps % 100 == 0 or self.GAN.steps < 10 and not self.silent:
            print("\n\nRound " + str(self.GAN.steps) + ":")
            print("D:", np.array(a))
            print("G:", np.array(b))
            print("R:", np.array(c))

            s = round((time.clock() - self.lastblip), 4)
            self.lastblip = time.clock()

            steps_per_second = 100 / s
            steps_per_minute = steps_per_second * 60
            steps_per_hour = steps_per_minute * 60
            print("Steps/Second: " + str(round(steps_per_second, 2)))
            print("Steps/Hour: " + str(round(steps_per_hour)))

            min1k = floor(1000/steps_per_minute)
            sec1k = floor(1000/steps_per_second) % 60
            print("1k Steps: " + str(min1k) + ":" + str(sec1k))
            steps_left = 200000 - self.GAN.steps + 1e-7
            hours_left = steps_left // steps_per_hour
            minutes_left = (steps_left // steps_per_minute) % 60

            print("Til Completion: " + str(int(hours_left)) + "h" + str(int(minutes_left)) + "m")
            print()

            #Save Model
            if self.GAN.steps % 500 == 0:
                self.save(floor(self.GAN.steps / 10000))
            if self.GAN.steps % 1000 == 0 or (self.GAN.steps % 100 == 0 and self.GAN.steps < 2500):
                self.evaluate(floor(self.GAN.steps / 1000))


        printProgressBar(self.GAN.steps % 100, 99, decimals = 0)

        self.GAN.steps = self.GAN.steps + 1

    @tf.function
    def train_step(self, source_images, target_images, perform_gp = True):

        with tf.GradientTape() as disc_tape, tf.GradientTape() as g1_tape, tf.GradientTape() as g2_tape:
            #Get style information

            #Generate images
            generated_images = self.GAN.G1(source_images)
            recon_images = self.GAN.G2(generated_images)

            #Discriminate
            real_output = self.GAN.D(target_images, training=True)
            fake_output = self.GAN.D(generated_images, training=True)

            #Hinge loss function
            gan_loss = K.mean(fake_output)
            recon_loss = K.mean(K.square(source_images - recon_images))

            disc_loss = K.mean(K.relu(1.0 + real_output) + K.relu(1.0 - fake_output))

            if perform_gp:
                #R1 gradient penalty
                disc_loss += gradient_penalty(target_images, real_output) * 10

            g1_loss = gan_loss + (recon_loss * 10)
            g2_loss = recon_loss



        #Get gradients for respective areas
        gradients_of_g1 = g1_tape.gradient(g1_loss, self.GAN.G1.trainable_variables)
        gradients_of_g2 = g2_tape.gradient(g2_loss, self.GAN.G2.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.GAN.D.trainable_variables)

        #Apply gradients
        self.GAN.GM1.apply_gradients(zip(gradients_of_g1, self.GAN.G1.trainable_variables))
        self.GAN.GM2.apply_gradients(zip(gradients_of_g2, self.GAN.G2.trainable_variables))
        self.GAN.DMO.apply_gradients(zip(gradients_of_discriminator, self.GAN.D.trainable_variables))

        return disc_loss, gan_loss, recon_loss

    def evaluate(self, num = 0):

        source_images = self.im[0].get_batch(32).astype('float32')

        generated_images = self.GAN.G1.predict(source_images, batch_size = BATCH_SIZE)

        r = []
        for i in range(0, 32, 8):
            r.append(np.concatenate(source_images[i:i+8], axis = 1))
            r.append(np.concatenate(generated_images[i:i+8], axis = 1))

        c1 = np.concatenate(r, axis = 0)
        c1 = np.clip(c1, 0.0, 1.0)
        x = Image.fromarray(np.uint8(c1*255))

        x.save("Results/i"+str(num)+".png")

        # Moving Average
        generated_images = self.GAN.GE.predict(source_images, batch_size = BATCH_SIZE)

        r = []
        for i in range(0, 32, 8):
            r.append(np.concatenate(source_images[i:i+8], axis = 1))
            r.append(np.concatenate(generated_images[i:i+8], axis = 1))

        c1 = np.concatenate(r, axis = 0)
        c1 = np.clip(c1, 0.0, 1.0)
        x = Image.fromarray(np.uint8(c1*255))

        x.save("Results/i"+str(num)+"-ma.png")

    def generateSamples(self, n = 300):

        try:
            os.mkdir("Results/Samples")
        except:
            pass

        source_images = self.im[0].get_batch(n).astype('float32')

        generated_images = self.GAN.GE.predict(source_images, batch_size = BATCH_SIZE)

        for i in range(n):
            x = np.concatenate([source_images[i], generated_images[i]], axis = 1)
            x = np.clip(x, 0.0, 1.0)
            x = Image.fromarray(np.uint8(x*255))
            x.save("Results/Samples/i"+str(i)+".png")

    def saveModel(self, model, name, num):
        json = model.to_json()
        with open("Models/"+name+".json", "w") as json_file:
            json_file.write(json)

        model.save_weights("Models/"+name+"_"+str(num)+".h5")

    def loadModel(self, name, num):

        file = open("Models/"+name+".json", 'r')
        json = file.read()
        file.close()

        mod = model_from_json(json)
        mod.load_weights("Models/"+name+"_"+str(num)+".h5")

        return mod

    def save(self, num): #Save JSON and Weights into /Models/
        self.saveModel(self.GAN.D, "dis", num)
        self.saveModel(self.GAN.G1, "gen1", num)
        self.saveModel(self.GAN.G2, "gen2", num)

        self.saveModel(self.GAN.GE, "genMA", num)


    def load(self, num): #Load JSON and Weights from /Models/

        #Load Models
        self.GAN.D = self.loadModel("dis", num)
        self.GAN.G1 = self.loadModel("gen1", num)
        self.GAN.G2 = self.loadModel("gen2", num)

        self.GAN.GE = self.loadModel("genMA", num)









if __name__ == "__main__":
    model = StarGAN(lr = 0.0001, silent = False)
    model.load(5)
    model.GAN.steps = 51000
    model.generateSamples()

    while model.GAN.steps < 1000001:
        model.train()
