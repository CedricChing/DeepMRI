import h5py
import numpy    as np
import scipy.io as sio
import tensorflow as tf
import os
from model import real2complex, complex2real

def load_data(data_path, batch_size):
    # print 'Loading dataset from '+ data_path
    # with h5py.File(os.path.join(data_path,'2Dpoisson5.mat'),'r') as f:
    #     mask = f['mask'][:]
    #     mask = np.fft.ifftshift(mask)
    mask = sio.loadmat(data_path+'/1Duniform2.98_ac29.mat')
    mask = mask['mask']
    mask = np.fft.ifftshift(mask)
    # with h5py.File(os.path.join(data_path,'label_12ch_v1.h5'),'r') as f:
    #     data = f['label_12ch'][0:nb_samples]
    # data = np.transpose(data,(0, 2, 3, 1))
    # nb_train = nb_samples // 11 * 10
    # channel = data.shape[-1] // 2
    # data_real = data[:,:,:,:channel]
    # data_imag = data[:,:,:,channel:]
    # data =  data_real + 1j*data_imag
    # train_data = data[:nb_train]
    # validate_data = data[nb_train:]
    train_data = read_and_decode(data_path+'/train.tfrecords', batch_size)
    validate_data = read_and_decode(data_path+'/val.tfrecords', batch_size)
    with h5py.File(os.path.join(data_path,'test_data.h5'),'r') as f:
        test_data = f['test'][:]
    test_data = np.transpose(test_data, (0, 2, 3, 1))
    channel = test_data.shape[-1] // 2
    test_data_real = test_data[:, :, :, :channel]
    test_data_imag = test_data[:, :, :, channel:]
    test_data = test_data_real + 1j * test_data_imag
    print('Loading Done.')

    return train_data, validate_data, test_data, mask

def read_and_decode(filename, batch_size):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(filename_queue)
    feature = {'train/label':tf.FixedLenFeature([],tf.string)}
    features = tf.parse_single_example(serialized_example, features=feature)
    img = tf.decode_raw(features['train/label'], tf.float64)
    img = tf.reshape(img, shape=[256, 256, 24])
    # img_batch = tf.train.shuffle_batch([img], batch_size=batch_size, num_threads=64, capacity=30, min_after_dequeue=10)
    return img

def setup_inputs(x, mask, batch_size):

    channel = x.shape[-1].value // 2
    mask = np.tile(mask, (channel, 1, 1))
    mask_tf = tf.cast(tf.constant(mask), tf.float32)
    mask_tf_c = tf.cast(mask_tf, tf.complex64)
    x_complex = real2complex(x)
    x_complex = tf.cast(x_complex, tf.complex64)
    x_complex = tf.transpose(x_complex, [2, 0, 1])
    kx = tf.fft2d(x_complex)
    kx_mask = kx * mask_tf_c
    x_u = tf.ifft2d(kx_mask)
    x_u = tf.transpose(x_u, [1, 2, 0])
    kx_mask = tf.transpose(kx_mask, [1, 2, 0])

    x_u_cat = complex2real(x_u)
    x_cat = tf.cast(x, tf.float32)
    mask_tf_c = tf.transpose(mask_tf_c, [1, 2, 0])

    features, labels, kx_mask, masks = tf.train.shuffle_batch([x_u_cat,x_cat, kx_mask, mask_tf_c],
                                                     batch_size=batch_size,
                                                     num_threads=64,
                                                     capacity=50,
                                                     min_after_dequeue=10)

    return features, labels, kx_mask, masks

def setup_inputs_test(x, mask, norm=None):
    batch = x.shape[0]
    channel = x.shape[-1]
    mask = np.tile(mask, (batch, channel, 1, 1))
    mask = np.transpose(mask, (0, 2, 3, 1))
    kx = np.fft.fft2(x, axes=(1,2), norm=norm)
    kx_mask = kx * mask
    x_u = np.fft.ifft2(kx_mask, axes=(1,2), norm=norm)

    x_u_cat = np.concatenate((np.real(x_u), np.imag(x_u)), axis=-1)
    x_cat = np.concatenate((np.real(x), np.imag(x)), axis=-1)
    mask_c = mask.astype(np.complex64)
    return x_u_cat, x_cat, kx_mask, mask_c