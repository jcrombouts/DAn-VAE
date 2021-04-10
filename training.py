import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
import datetime
from numpy.random import seed

from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Lambda, MaxPool2D, UpSampling2D, Flatten, Reshape, BatchNormalization, Dropout, LeakyReLU, Concatenate, Conv2DTranspose, Activation
from tensorflow.keras import Model, layers, models, Input, activations
from tensorflow.keras.initializers import glorot_normal
from tensorflow.keras import backend as K
from tensorboard.plugins.hparams import api as hp
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.preprocessing import dataset_utils
from tensorflow.python.keras.layers.preprocessing import image_preprocessing
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import io_ops
import tensorflow_probability as tfp
from tensorflow.keras.losses import MeanSquaredError


# decoder
def create_px(input_shape, zx_dim, zy_dim): # p(x|zx, zy)
    zx_input = Input(shape = (zx_dim, ))
    zy_input = Input(shape = (zy_dim, ))

    n_layers = 3
    first_filters = input_shape[0] // (2**(n_layers))
    fist_dim = first_filters * first_filters * 128

    concatenate_latent_repr = Concatenate(axis = 1)([zx_input, zy_input])
    
    h = Dense(fist_dim, activation = 'relu', kernel_initializer=glorot_normal)(concatenate_latent_repr)
    h = Reshape((first_filters,first_filters, 128))(h)
    h = Conv2DTranspose(64, kernel_size = 3, strides=(2,2), activation = 'relu', padding = 'same', kernel_initializer=glorot_normal)(h)
    h = Conv2DTranspose(32, kernel_size = 3, strides=(2,2), activation = 'relu', padding = 'same', kernel_initializer=glorot_normal)(h)

    x_output = Conv2DTranspose(3, kernel_size = 3, strides = (2,2), activation='sigmoid', padding='same', kernel_initializer=glorot_normal)(h)

    decoder = Model([zx_input, zy_input], x_output)

    return decoder

def create_pzy(zy_dim, y_dim): # p(z_y | y)
    y_input = Input(shape = (y_dim, ))
    h = Dense(zy_dim, activation='relu', kernel_initializer=glorot_normal)(y_input)

    zy_mean = Dense(zy_dim, kernel_initializer=glorot_normal)(h)
    zy_logvar = Dense(zy_dim, activation= 'softplus', kernel_initializer=glorot_normal)(h)

    model_zy = Model(y_input, [zy_mean, zy_logvar])

    return model_zy

# encoders
def create_qzx(input_shape, zx_dim): # q(z_x|x)
    x_input = Input(shape = input_shape)

    h = Conv2D(32, kernel_size = 3, strides=(2,2), activation = 'relu', padding = 'same', kernel_initializer=glorot_normal)(x_input)
    h = Conv2D(64, kernel_size = 3, strides=(2,2), activation = 'relu', padding = 'same', kernel_initializer=glorot_normal)(h)
    h = Conv2D(128, kernel_size = 3, strides=(2,2), activation = 'relu', padding = 'same', kernel_initializer=glorot_normal)(h)

    h = Flatten()(h)
    zx_mean = Dense(zx_dim, kernel_initializer=glorot_normal)(h)
    zx_logvar = Dense(zx_dim, activation='softplus', kernel_initializer=glorot_normal)(h)

    model_qzx = Model(x_input, [zx_mean, zx_logvar])

    return model_qzx

def create_qzy(input_shape, zy_dim): # q(z_y|x)
    x_input = Input(shape = input_shape)

    h = Conv2D(32, kernel_size = 3, strides=(2,2), activation = 'relu', padding = 'same', kernel_initializer=glorot_normal)(x_input)
    h = Conv2D(64, kernel_size = 3, strides=(2,2), activation = 'relu', padding = 'same', kernel_initializer=glorot_normal)(h)
    h = Conv2D(128, kernel_size = 3, strides=(2,2), activation = 'relu', padding = 'same', kernel_initializer=glorot_normal)(h)

    h = Flatten()(h)
    zy_mean = Dense(zy_dim, kernel_initializer=glorot_normal)(h)
    zy_logvar = Dense(zy_dim, activation='softplus', kernel_initializer=glorot_normal)(h)

    model_qzy = Model(x_input, [zy_mean, zy_logvar])

    return model_qzy

def create_qy(zy_dim, y_dim): # regressor: q(y|z_y)
    zy_input = Input(shape=(zy_dim, ))
    relu_input = Activation(activations.relu)(zy_input)
    y_output = Dense(1, activation='sigmoid', kernel_initializer=glorot_normal)(relu_input)

    classifiery_qy = Model(zy_input, y_output)

    return classifiery_qy

def qy_evaluation_classifier(dim, num_classes):
    zy_input = Input(shape=(dim, ))
    relu_input = Activation(activations.relu)(zy_input)
    y_output = Dense(1, activation='sigmoid', kernel_initializer=glorot_normal)(relu_input)

    classifiery_qy = Model(zy_input, y_output)
    classifiery_qy.compile(optimizer='adam', loss='mean_squared_error', metrics = [tf.keras.metrics.RootMeanSquaredError()])

    return classifiery_qy

def encode_evaluation(model, qzx, qzy, X):
    zx_q_mean, zx_q_logvar = qzx(X, training = False)
    zx_q = model.reparameterize(zx_q_mean, zx_q_logvar)

    zy_q_mean, zy_q_logvar = qzy(X, training = False)
    zy_q = model.reparameterize(zy_q_mean, zy_q_logvar)

    return zx_q.numpy(), zy_q.numpy()

class DAN(keras.Model):
    def __init__(self, zx_dim, zy_dim, y_dim, aux_loss_multiplier_y, beta_x, beta_y, px, pzy, qzx, qzy, qy):
        super(DAN, self).__init__()
        self.zx_dim = zx_dim
        self.zy_dim = zy_dim
        self.y_dim = y_dim

        self.px = px
        self.pzy = pzy

        self.qzx = qzx
        self.qzy = qzy

        self.qy = qy
        self.aux_loss_multiplier_y = aux_loss_multiplier_y

        self.beta_x = tf.Variable(beta_x, trainable=False, name='betax', dtype=tf.float32)
        self.beta_y = tf.Variable(beta_y, trainable=False, name='betay', dtype=tf.float32)
        self.max_beta_x = tf.Variable(beta_x, trainable=False, name='maxbetax', dtype=tf.float32)
        self.max_beta_y = tf.Variable(beta_y, trainable=False, name='maxbetay', dtype=tf.float32)

    def compile(self, optimizer, *args, **kwargs):
        super(DAN, self).compile(*args, **kwargs)
        self.optimizer = optimizer
        self.metric_sup = tf.keras.metrics.Mean()
        self.metric_unsup = tf.keras.metrics.Mean()
        self.metric_classifier = tf.keras.metrics.Mean()
        self.rec_loss = tf.keras.metrics.Mean()
        self.kl_zx = tf.keras.metrics.Mean()
        self.kl_zy = tf.keras.metrics.Mean()
        self.approx_kl_zy = tf.keras.metrics.Mean()     

    def reparameterize(self, z_mu, z_log_sigma):
        dimension_batch  = tf.shape(z_mu)[0]
        dimension_z = tf.shape(z_log_sigma)[1]
        eps = tf.keras.backend.random_normal(shape=(dimension_batch, dimension_z))
        return z_mu + tf.exp(0.5 * z_log_sigma) * eps 

    def compute_rec_loss(self, x_in, x_rec):
        shape_x = x_in.get_shape().as_list()
        dim = np.prod(shape_x[1:]) 
        x_batch_flatten = tf.reshape(x_in, [-1, dim]) # reshape x_in from (batch_dim, width,height,1) --> (batch_dim, width*height*1)
        x_mean_batch_flatten = tf.reshape(x_rec, [-1, dim])
        
        reconstruction_loss = tf.reduce_sum(tf.square(x_batch_flatten - x_mean_batch_flatten), axis = 1)
        reconstruction_loss = tf.reduce_mean(reconstruction_loss)
        return reconstruction_loss

    def compute_anom_kl(self, z_mean, z_log_var):
        mean_anomaly = tf.ones(tf.shape(z_mean)) * 10.0
        mean_anomaly = tf.convert_to_tensor(mean_anomaly)

        kl_loss = tf.reduce_sum(tf.square(z_mean) + tf.exp(2 * z_log_var) - 2 * z_log_var - 1 - 2 * (z_mean * mean_anomaly) + tf.square(mean_anomaly), axis=-1)
        kl_loss = tf.reduce_mean(kl_loss)
        return kl_loss

    def compute_loss(self, x,y, training, is_sup, is_anom):
        # Encode
        zx_q_mean, zx_q_logvar = self.qzx(x, training=training)
        qzx_approx_post = tfp.distributions.Normal(zx_q_mean, tf.exp(0.5 * zx_q_logvar))
        zx_q = self.reparameterize(zx_q_mean, zx_q_logvar)

        zy_q_mean, zy_q_logvar = self.qzy(x, training=training)
        qzy_approx_post = tfp.distributions.Normal(zy_q_mean, tf.exp(0.5 * zy_q_logvar))
        zy_q = self.reparameterize(zy_q_mean, zy_q_logvar)

        batch_dim = tf.shape(zy_q_mean)[0]
        zx_p_mean, zx_p_logvar = tf.zeros([batch_dim, self.zx_dim]), tf.ones([batch_dim, self.zx_dim])
        pzx_prior = tfp.distributions.Normal(zx_p_mean, tf.exp(0.5 * zx_p_logvar))

        # Decode
        x_recon = self.px([zx_q, zy_q], training=training)

        if not is_sup:
            pzy_prior =  tfp.distributions.Normal(0., 1.)
        
            reconstruction_loss = self.compute_rec_loss(x, x_recon)
            if is_anom:
                KL_zx = self.beta_x * self.compute_anom_kl(zx_q_mean, zx_q_logvar)
            else:
                KL_zx = self.beta_x * tf.reduce_mean(tf.reduce_sum(qzx_approx_post.log_prob(zx_q) - pzx_prior.log_prob(zx_q), axis = -1))

            KL_zy = self.beta_y * tf.reduce_mean(tf.reduce_sum(qzy_approx_post.log_prob(zy_q) - pzy_prior.log_prob(zy_q), axis = -1))
            
            return reconstruction_loss, KL_zx, KL_zy, tf.constant(0, dtype=tf.float32)

        else: # supervised
            zy_p_mean, zy_p_logvar = self.pzy(y, training=training)
            pzy_prior = tfp.distributions.Normal(zy_p_mean, tf.exp(0.5 * zy_p_logvar))
            
            # Auxiliary losses
            y_hat = self.qy(zy_q, training=training)
            y_hat = tf.reshape(y_hat, [-1])

            reconstruction_loss = self.compute_rec_loss(x, x_recon)
            if is_anom:
                KL_zx = self.compute_anom_kl(zx_q_mean, zx_q_logvar)
            else:
                KL_zx = self.beta_x * tf.reduce_mean(tf.reduce_sum(qzx_approx_post.log_prob(zx_q) - pzx_prior.log_prob(zx_q), axis = -1))

            KL_zy = self.beta_y * tf.reduce_mean(tf.reduce_sum(qzy_approx_post.log_prob(zy_q) - pzy_prior.log_prob(zy_q), axis = -1))

            # MSE = MeanSquaredError()
            RMSE_y = tf.sqrt(tf.reduce_mean((y - y_hat)**2)) # MSE(y, y_hat)

            return reconstruction_loss, KL_zx, KL_zy, RMSE_y
    
    def train_step(self, data):
        if isinstance(data, tuple):
            # supervised
            if len(data) == 2:
                x, y = data[0], data[1]
                is_anomaly = False
            elif len(data) == 3:
                x, y = data[0], data[1]
                is_anomaly = True
            is_sup = True
        else:
            x = data
            y = None
            is_sup = False
            is_anomaly = False # anomaly batch is always supervised (design choice to make implementation easier )
        
        with tf.GradientTape() as tape:
            reconstruction_loss, KL_zx, KL_zy, RMSE_y = self.compute_loss(x, y, True, is_sup, is_anomaly)
            if is_sup:
                loss = reconstruction_loss + KL_zx + KL_zy + (self.aux_loss_multiplier_y * RMSE_y)
            else:
                loss = reconstruction_loss + KL_zx + KL_zy
            
        gradients = tape.gradient(loss, self.trainable_variables)
        # self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.optimizer.apply_gradients((grad, var) for (grad, var) in zip(gradients, self.trainable_variables) if grad is not None)

        if is_sup:
            self.metric_classifier.update_state(RMSE_y)
            self.metric_sup.update_state(loss)
            self.rec_loss.update_state(reconstruction_loss)
            self.kl_zx.update_state(KL_zx)
            self.kl_zy.update_state(KL_zy)
            
        else:
            self.metric_unsup.update_state(loss)
            self.rec_loss.update_state(reconstruction_loss)
            self.kl_zx.update_state(KL_zx)
            self.approx_kl_zy.update_state(KL_zy)


        return {'loss_sup': self.metric_sup.result(), 'loss_unsup': self.metric_unsup.result(), 'loss_class': self.metric_classifier.result(), 'rec_loss': self.rec_loss.result(), 
        'kl_zx':self.kl_zx.result(), 'kl_zy': self.kl_zy.result(), 'approx_kl_zy': self.approx_kl_zy.result()}

    def test_step(self, data):
        if isinstance(data, tuple):
            # supervised
            if len(data) == 2:
                x, y = data[0], data[1]
                is_anomaly = False
            elif len(data) == 3:
                x, y = data[0], data[1]
                is_anomaly = True
            is_sup = True
        else:
            x = data
            y = None
            is_sup = False
            is_anomaly = False # anomaly batch is always supervised (design choice to make implementation easier )
          
        reconstruction_loss, KL_zx, KL_zy, RMSE_y = self.compute_loss(x, y, False, is_sup, is_anomaly)
        
        if is_sup:
            loss = reconstruction_loss + KL_zx + KL_zy + (self.aux_loss_multiplier_y * RMSE_y)
        else:
            loss = reconstruction_loss + KL_zx + KL_zy

        if is_sup:
            self.metric_classifier.update_state(RMSE_y)
            self.metric_sup.update_state(loss)
            self.rec_loss.update_state(reconstruction_loss)
            self.kl_zx.update_state(KL_zx)
            self.kl_zy.update_state(KL_zy)
            
        else:
            self.metric_unsup.update_state(loss)
            self.rec_loss.update_state(reconstruction_loss)
            self.kl_zx.update_state(KL_zx)
            self.approx_kl_zy.update_state(KL_zy)

        return {'loss_sup': self.metric_sup.result(), 'loss_unsup': self.metric_unsup.result(), 'loss_class': self.metric_classifier.result(), 'rec_loss': self.rec_loss.result(), 
        'kl_zx':self.kl_zx.result(), 'kl_zy': self.kl_zy.result(), 'approx_kl_zy': self.approx_kl_zy.result()}


class WarmUpCallback(tf.keras.callbacks.Callback):
    def __init__(self, verbose):
        super(WarmUpCallback, self).__init__()
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        new_beta_x = min(self.model.max_beta_x, self.model.max_beta_x * (float(epoch+1) / 50.0))
        new_beta_y = min(self.model.max_beta_y, self.model.max_beta_y * (float(epoch+1) / 50.0)) # first 50 epochs warm up beta coefficient...
        tf.keras.backend.set_value(self.model.beta_x, new_beta_x)
        tf.keras.backend.set_value(self.model.beta_y, new_beta_y)

        if (self.verbose > 0) & (epoch + 1 <= 50.0):
            print('Epoch %05d: WarmUpCallback increasing beta_x to %s.' % (epoch + 1, new_beta_x.numpy()))
            print('Epoch %05d: WarmUpCallback increasing beta_y to %s.' % (epoch + 1, new_beta_y.numpy()))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['beta_x'] = tf.keras.backend.get_value(self.model.beta_x)
        logs['beta_y'] = tf.keras.backend.get_value(self.model.beta_y)

def path_to_image(path, image_size, num_channels, interpolation):
    img = io_ops.read_file(path)
    img = image_ops.decode_image(img, channels=num_channels, expand_animations=False)
    img = image_ops.resize_images_v2(img, image_size, method=interpolation)
    img.set_shape((image_size[0], image_size[1], num_channels))
    return img

def path_to_label(path, image_size, num_channels, interpolation):
    img = path_to_image(path, image_size, num_channels, interpolation)
    img = (img / 255.0)
    hsv_img = tf.image.rgb_to_hsv(img)
    avg_hsv = tf.reduce_mean(hsv_img[:,:,2])
    return avg_hsv

def paths_and_labels_to_dataset(image_paths,
                                image_size,
                                num_channels,
                                labels,
                                label_mode,
                                num_classes,
                                interpolation):
  """Constructs a dataset of images and labels."""
  # TODO(fchollet): consider making num_parallel_calls settable
  path_ds = dataset_ops.Dataset.from_tensor_slices(image_paths)
  img_ds = path_ds.map(lambda x: path_to_image(x, image_size, num_channels, interpolation))
  
  if label_mode == 'regression':
    label_ds = path_ds.map(lambda x: path_to_label(x, image_size, num_channels, interpolation))
    img_ds = dataset_ops.Dataset.zip((img_ds, label_ds))
  elif label_mode == 'anomaly':
    label_ds_brightness = path_ds.map(lambda x: path_to_label(x, image_size, num_channels, interpolation))
    label_ds_categorical = dataset_utils.labels_to_dataset(labels, 'categorical', num_classes)
    img_ds = dataset_ops.Dataset.zip((img_ds, label_ds_brightness, label_ds_categorical))
  elif label_mode:
    label_ds = dataset_utils.labels_to_dataset(labels, label_mode, num_classes)
    img_ds = dataset_ops.Dataset.zip((img_ds, label_ds))
  return img_ds

def custom_img_dataset_from_directory(directory, labels = 'inferred', label_mode = 'categorical', class_names = None, 
                                        color_mode = 'rgb', batch_size=32, image_size=(256,256), shuffle = True, 
                                        seed = None, validation_split = None, subset = None, interpolation = 'bilinear', follow_links = False):
    "generates a tf.data.Dataset from image files in a directory, with brightness label for each image"
    WHITELIST_FORMATS = ('.bmp', '.gif', '.jpeg', '.jpg', '.png')

    if color_mode == 'rgb':
        num_channels = 3

    interpolation = image_preprocessing.get_interpolation(interpolation)
    dataset_utils.check_validation_split_arg(validation_split, subset, shuffle, seed)

    if seed is None:
        seed = np.random.randint(1e6)
    
    image_paths, labels, class_names = dataset_utils.index_directory(
      directory,
      labels,
      formats=WHITELIST_FORMATS,
      class_names=class_names,
      shuffle=shuffle,
      seed=seed,
      follow_links=follow_links)

    image_paths, labels = dataset_utils.get_training_or_validation_split(
      image_paths, labels, validation_split, subset)

    dataset = paths_and_labels_to_dataset(
      image_paths=image_paths,
      image_size=image_size,
      num_channels=num_channels,
      labels=labels,
      label_mode=label_mode,
      num_classes=len(class_names),
      interpolation=interpolation)
    
    if shuffle:
        # Shuffle locally at each iteration
        dataset = dataset.shuffle(buffer_size=batch_size * 8, seed=seed)
    dataset = dataset.batch(batch_size)
    # Users may need to reference `class_names`.
    dataset.class_names = class_names

    return dataset

class runDAN:
    def __init__(self, zy_dim, zx_dim, beta_x, beta_y, anom_ratio_train, anom_ratio_test, sup_ratio, run, epochs, batch_size, input_shape):
        # global parameters
        self.run = run
        self.anom_ratio_train = anom_ratio_train
        self.anom_ratio_test = anom_ratio_test
        self.sup_ratio = sup_ratio

        self.zy_dim = zy_dim
        self.zx_dim = zx_dim
        self.aux_loss_multiplier_y = 3500.0 * (((1-self.sup_ratio) + self.sup_ratio) / self.sup_ratio)
        self.beta_x = beta_x
        self.beta_y = beta_y
        
        # fixed parameters
        self.input_shape = input_shape
        self.img_width, self.img_height, self.img_depth = input_shape
        self.batch_size = batch_size
        self.epochs = epochs
        self.dir_data = os.getcwd()
        self.y_dim = 1

    def trainDAN(self):
        seed = 1
        tf.random.set_seed(seed)
        print('\n dataset = plant Village, run = {}, anom_ratio_train = {}, sup_ratio_train = {}, zy_dim = {}, zx_dim = {}, aux_loss = {}, beta_x = {}, beta_y = {}\n'.format(self.run, self.anom_ratio_train, self.sup_ratio, self.zy_dim, self.zx_dim, self.aux_loss_multiplier_y, self.beta_x, self.beta_y))
        
        data_dir_train_normal = os.path.join(self.dir_data, 'data/train/normal')
        data_dir_train_normal = Path(data_dir_train_normal)
        data_dir_train_anomaly = os.path.join(self.dir_data, 'data/train/anomaly')
        data_dir_train_anomaly = Path(data_dir_train_anomaly)

        data_dir_test_normal = os.path.join(self.dir_data, 'data/test/normal')
        data_dir_test_normal = Path(data_dir_test_normal)
        data_dir_test_anomaly = os.path.join(self.dir_data, 'data/test/anomaly')
        data_dir_test_anomaly = Path(data_dir_test_anomaly)

        n_normal_train = len(list(data_dir_train_normal.glob('*/*.JPG')))
        n_anomaly_train = len(list(data_dir_train_anomaly.glob('*/*.JPG')))
        n_normal_test = len(list(data_dir_test_normal.glob('*/*.JPG')))
        n_anomaly_test = len(list(data_dir_test_anomaly.glob('*/*.JPG')))
        
        anom_ratio_train_generator = (n_normal_train * self.anom_ratio_train) / n_anomaly_train
        # anom_ratio_test_generator = (n_normal_test * self.anom_ratio_test) / n_anomaly_test

        tf.random.set_seed(seed)

        if self.sup_ratio == 1.0:
            train_gen_sup = custom_img_dataset_from_directory(
                data_dir_train_normal,
                seed=seed,
                label_mode = 'regression', # 'regression' for supervised, None for unsupervised     
                image_size=(self.img_height, self.img_width),
                batch_size=self.batch_size)

            test_gen_sup = custom_img_dataset_from_directory(
                data_dir_test_normal,
                seed=seed,
                label_mode = 'regression', # 'regression' for supervised, 'anomaly' for supervised+anomaly and None for unsupervised     
                image_size=(self.img_height, self.img_width),
                batch_size=self.batch_size)

        else:
            train_gen_sup = custom_img_dataset_from_directory(
                data_dir_train_normal,
                validation_split=1-self.sup_ratio,
                subset="training",
                seed=seed,
                label_mode = 'regression', # 'regression' for supervised, None for unsupervised     
                image_size=(self.img_height, self.img_width),
                batch_size=self.batch_size)

            train_gen_unsup = custom_img_dataset_from_directory( #           
                data_dir_train_normal,
                subset = 'validation',
                validation_split=1-self.sup_ratio,
                seed=seed,
                label_mode = None,   
                image_size=(self.img_height, self.img_width),
                batch_size=self.batch_size)

            test_gen_sup = custom_img_dataset_from_directory(
                data_dir_test_normal,
                validation_split=1-self.sup_ratio,
                subset="training",
                seed=seed,
                label_mode = 'regression', # 'regression' for supervised, 'anomaly' for supervised+anomaly and None for unsupervised     
                image_size=(self.img_height, self.img_width),
                batch_size=self.batch_size)

            test_gen_unsup = custom_img_dataset_from_directory( # 
                data_dir_test_normal,
                seed=seed,
                subset = 'validation',
                validation_split = 1-self.sup_ratio,
                label_mode = None,   
                image_size=(self.img_height, self.img_width),
                batch_size=self.batch_size)

        train_gen_anom = custom_img_dataset_from_directory(
            data_dir_train_anomaly,
            validation_split=anom_ratio_train_generator,
            subset="validation",
            seed=seed,
            label_mode = 'anomaly',
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size)

        
        test_gen_anom = custom_img_dataset_from_directory(
            data_dir_test_anomaly,
            seed=seed,
            label_mode = 'anomaly',
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size)

        normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255) # normalise images to [0,1] range

        train_ds_sup = train_gen_sup.map(lambda x, y: (normalization_layer(x), y))
        train_ds_anom = train_gen_anom.map(lambda x, y_c, y_a: (normalization_layer(x), y_c, y_a))
        test_ds_sup = test_gen_sup.map(lambda x, y: (normalization_layer(x), y))
        test_ds_anom = test_gen_anom.map(lambda x, y_c, y_a: (normalization_layer(x), y_c, y_a))
        
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        train_ds_sup = train_ds_sup.cache().prefetch(buffer_size=AUTOTUNE)       
        train_ds_anom = train_ds_anom.cache().prefetch(buffer_size=AUTOTUNE)

        test_ds_sup = test_ds_sup.cache().prefetch(buffer_size=AUTOTUNE)       
        test_ds_anom = test_ds_anom.cache().prefetch(buffer_size=AUTOTUNE)

        steps_epoch_train_sup = (tf.data.experimental.cardinality(train_ds_sup).numpy())
        steps_epoch_train_anom = (tf.data.experimental.cardinality(train_ds_anom).numpy())
        steps_epoch_test_sup = (tf.data.experimental.cardinality(test_ds_sup).numpy())     
        steps_epoch_test_anom = (tf.data.experimental.cardinality(test_ds_anom).numpy())

        if self.sup_ratio < 1.0:
            train_ds_unsup = train_gen_unsup.map(lambda x: (normalization_layer(x)))
            test_ds_unsup = test_gen_unsup.map(lambda x: (normalization_layer(x)))
            train_ds_unsup = train_ds_unsup.cache().prefetch(buffer_size=AUTOTUNE)
            test_ds_unsup = test_ds_unsup.cache().prefetch(buffer_size=AUTOTUNE)
            steps_epoch_train_unsup = (tf.data.experimental.cardinality(train_ds_unsup).numpy())
            steps_epoch_test_unsup = (tf.data.experimental.cardinality(test_ds_unsup).numpy())

        # create models
        px = create_px(self.input_shape, self.zx_dim, self.zy_dim) # px 
        pzy = create_pzy(self.zy_dim, self.y_dim) # encoder p(z_y|y)
        qzx = create_qzx(self.input_shape, self.zx_dim) # encoder q(z_x|x)
        qzy = create_qzy(self.input_shape, self.zy_dim) # encoder q(z_y|x)
        qy = create_qy(self.zy_dim, self.y_dim) # regressor

        # # paths for saving checkpoints
        checkpoint_path = os.path.abspath(self.dir_data+'/cp_pvillage/epoch{}_zy{}zx{}/sup{}/ratio{}_run{}'.format(self.epochs, self.zy_dim, self.zx_dim, int(self.sup_ratio*100.0), int(self.anom_ratio_train*100), self.run))
        path_weights = '/vae_weights-{epoch:04d}.ckpt'
        checkpoint_path_save = os.path.abspath(checkpoint_path+path_weights)
        path_weights_epochs = os.path.abspath(checkpoint_path+path_weights.format(epoch = self.epochs))
        checkpoint_dir = os.path.dirname(path_weights_epochs)
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

        # callbacks
        wu_callback = WarmUpCallback(verbose = 1)
        callbacks = [wu_callback]

        # # create model
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        dan = DAN(self.zx_dim, self.zy_dim, self.y_dim, self.aux_loss_multiplier_y, self.beta_x, self.beta_y, px, pzy, qzx, qzy, qy)
        dan.compile(optimizer = optimizer)
        
        if latest:
            try:
                print('loading weights...')
                dan.load_weights(latest).expect_partial()
            except Exception as e:
                print(e)
        else:
            print('training model...')
            # train first 50 epochs on normal sup and unsupervised data
            for epoch in range(0, 50):
                dan.fit(train_ds_sup, initial_epoch = epoch, epochs = epoch + 1, steps_per_epoch = steps_epoch_train_sup, batch_size = self.batch_size, callbacks = callbacks, verbose = 2, validation_data = (test_ds_sup), validation_steps = steps_epoch_test_sup)
                if self.sup_ratio < 1.0:
                    dan.fit(train_ds_unsup, initial_epoch = epoch, epochs = epoch + 1, steps_per_epoch = steps_epoch_train_unsup, batch_size = self.batch_size, callbacks = callbacks, verbose = 2, validation_data = (test_ds_unsup), validation_steps = steps_epoch_test_unsup)
            
            if self.anom_ratio_train > 0.0:
                for epoch in range(50, self.epochs):
                    # get optimizer state, note that learning rate is changed using the LR schedule
                    symbolic_weights = getattr(dan.optimizer, 'weights')
                    weight_values = K.batch_get_value(symbolic_weights)
                    # set px not to trainable and recompile
                    dan.px.trainable = False  
                    dan.compile(optimizer = optimizer)
                    # load optimizer weights
                    dan.optimizer.set_weights(weight_values)
                    dan.fit(train_ds_anom, initial_epoch = epoch, epochs = epoch + 1, steps_per_epoch = steps_epoch_train_anom, batch_size = self.batch_size, callbacks = callbacks, verbose = 2, validation_data = (test_ds_anom), validation_steps = steps_epoch_test_anom)

                    # save optimizer weight
                    symbolic_weights = getattr(dan.optimizer, 'weights')
                    weight_values = K.batch_get_value(symbolic_weights)
                    dan.px.trainable = True
                    dan.compile(optimizer = optimizer)
                    dan.optimizer.set_weights(weight_values)

                    dan.fit(train_ds_sup, initial_epoch = epoch, epochs = epoch + 1, steps_per_epoch = steps_epoch_train_sup, batch_size = self.batch_size, callbacks = callbacks, verbose = 2, validation_data = (test_ds_sup), validation_steps = steps_epoch_test_sup)
                    if self.sup_ratio < 1.0:
                        dan.fit(train_ds_unsup, initial_epoch = epoch, epochs = epoch + 1, steps_per_epoch = steps_epoch_train_unsup, batch_size = self.batch_size, callbacks = callbacks, verbose = 2, validation_data = (test_ds_unsup), validation_steps = steps_epoch_test_unsup)
                
                dan.save_weights(checkpoint_path_save.format(epoch=epoch+1))
            else:
                for epoch in range(50, self.epochs):
                    dan.fit(train_ds_sup, initial_epoch = epoch, epochs = epoch + 1, steps_per_epoch = steps_epoch_train_sup, batch_size = self.batch_size, callbacks = callbacks, verbose = 2, validation_data = (test_ds_sup), validation_steps = steps_epoch_test_sup)
                    if self.sup_ratio < 1.0:
                        dan.fit(train_ds_unsup, initial_epoch = epoch, epochs = epoch + 1, steps_per_epoch = steps_epoch_train_unsup, batch_size = self.batch_size, callbacks = callbacks, verbose = 2, validation_data = (test_ds_unsup), validation_steps = steps_epoch_test_unsup)
                
                dan.save_weights(checkpoint_path_save.format(epoch=(epoch+1)))

        return dan

if __name__ == "__main__":
    zy_dim = 8
    zx_dim = 120
    beta_x = 1.0
    beta_y = 3.0
    anom_ratio_train = 0.10
    anom_ratio_test = 0.30
    sup_ratio = 0.5
    run = 0
    seed = 123
    input_shape = (128,128,3)
    img_height = input_shape[0]
    img_width = input_shape[1]
    epochs = 200
    batch_size = 128
    dir_data = os.getcwd()

    rangeLatentDimX = [24, 56, 120]
    rangeLatentDimY = [8, 8, 8]
    rangeAnomRatioTrain = [0.05, 0.10, 0.20]
    rangeAnomRatioTest = [0.30, 0.30, 0.30]

    # for (zx_dim, zy_dim) in zip(rangeLatentDimX, rangeLatentDimY):
        # for (anom_ratio_train, anom_ratio_test) in zip(rangeAnomRatioTrain, rangeAnomRatioTest):
    run_i = runDAN(zy_dim=zy_dim, zx_dim=zx_dim, beta_x = beta_x, beta_y = beta_y, anom_ratio_train = anom_ratio_train, anom_ratio_test = anom_ratio_test, sup_ratio = sup_ratio, run = run, epochs = epochs, batch_size = batch_size, input_shape = input_shape)
    dan = run_i.trainDAN()