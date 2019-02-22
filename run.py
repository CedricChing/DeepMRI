import tensorflow   as tf
import numpy    as np
from model  import getModel
from losses import mae,mse
from data   import setup_inputs, load_data, setup_inputs_test
import matplotlib.pyplot as plt
import os

EPOCHS = 40
BATCH_SIZE = 4
nb_samples = 3324
nb_train = 2950
nb_val = nb_samples - nb_train
nb_train_samples = nb_train * 8
eval_every = 750
test_every = 2
data_path = '/media/siat/381646F51646B3A2/data'
model_save_path = './saved_model'
model_name = 'model.ckpt'
lr_base = 0.0001
lr_decay_rate = 0.95
loss={'batch':[], 'count':[], 'epoch':[]}
val_loss=[]



def iterate_minibatch(data, batch_size, shuffle=True):
    n = len(data)
    n_idx = np.arange(n)
    n_batch = (n+batch_size-1) // batch_size
    if shuffle:
        np.random.shuffle(n_idx)
    for i in range(n_batch):
        batch_idx = n_idx[i*batch_size : (i+1)*batch_size]
        yield data[batch_idx], n_batch

def loss_plot(loss_type):
    iters = range(len(loss[loss_type]))
    plt.figure()
    plt.plot(iters, loss[loss_type], 'g', label='train loss')
    if loss_type == 'count':
        plt.plot(iters, val_loss, 'r', label='val loss')
    plt.grid(True)
    plt.xlabel(loss_type)
    plt.ylabel('loss')
    plt.legend(loc="upper right")
    plt.savefig(loss_type + '_plot.png')
    plt.show()

def real2complex_array(x):
    x = np.asarray(x)
    channel = x.shape[-1] // 2
    x_real = x[:,:,:,:channel]
    x_imag = x[:,:,:,channel:]
    return x_real + x_imag * 1j

def train(train_batch, validate_batch, test_data, mask):
    # nb_train = len(train_data)
    x = tf.placeholder(tf.float32,shape=(None,256,256,24),name='x_input')
    y_ = tf.placeholder(tf.float32,shape=(None,256,256,24),name='y_label')
    x_k = tf.placeholder(tf.complex64,shape=(None,256,256,12),name='x_kspace')
    mask_k = tf.placeholder(tf.complex64,shape=(None,256,256,12),name='mask')

    features, labels, kx_mask, mask_c = setup_inputs(train_batch, mask, BATCH_SIZE)
    f_val, l_val, kx_val, m_val = setup_inputs(validate_batch, mask, BATCH_SIZE)
    y = getModel(x, x_k, mask_k)
    global_step = tf.Variable(0.,trainable=False)
    with tf.name_scope('mse_loss'):
        total_loss = mae(y_,y)
    lr = tf.train.exponential_decay(lr_base,
                                    global_step=global_step,
                                    decay_steps=(nb_train_samples+BATCH_SIZE-1) // BATCH_SIZE,
                                    decay_rate=lr_decay_rate,
                                    staircase=False)
    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(lr).minimize(total_loss,global_step=global_step)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

        for i in range(EPOCHS):
            loss_sum = 0.0
            count_batch = 0
            ave_loss = 0.
            nb_batches = int(np.ceil(nb_train_samples // BATCH_SIZE))
            for n_batch in range(nb_batches):
                features_trian, labels_train, kx_mask_train, mask_c_train = sess.run([features,labels, kx_mask, mask_c])
                _,loss_value, step = sess.run([train_step, total_loss, global_step],
                                              feed_dict={x: features_trian, y_: labels_train,
                                                             x_k: kx_mask_train, mask_k: mask_c_train})

                loss_sum += loss_value
                count_batch += 1
                ave_loss = loss_sum / count_batch
                loss['batch'].append(ave_loss)
                print ('Epoch %3d-batch %3d/%3d  training loss: %8f' % (i+1, count_batch, nb_batches, ave_loss))

                # evaluate
                if count_batch % eval_every == 0:
                    count_batch_val = 0
                    loss_sum_val = 0.
                    ave_loss_val = 0.
                    loss['count'].append(ave_loss)
                    nb_batches_val = int(np.ceil(nb_val // BATCH_SIZE))
                    for n_batch_val in range(nb_batches_val):
                        features_val, labels_val, kx_mask_val, mask_c_val = sess.run([f_val, l_val, kx_val, m_val])
                        loss_value_val, pred_val = sess.run([total_loss,y],
                                                  feed_dict={x: features_val, y_: labels_val,
                                                             x_k: kx_mask_val, mask_k: mask_c_val})

                        loss_sum_val += loss_value_val
                        count_batch_val += 1
                        ave_loss_val = loss_sum_val / count_batch_val
                        print ('Epoch %3d-batch %3d/%3d  validation loss: %8f' % (
                        i + 1, count_batch_val, n_batch_val, ave_loss_val))

                    val_loss.append(ave_loss_val)
            loss['epoch'].append(ave_loss)
            saver.save(sess, os.path.join(model_save_path,model_name), global_step=global_step)

            # test every 5 epochs
            if (i+1) % test_every == 0:
                count = 0
                for y_test, n_batch in iterate_minibatch(test_data, batch_size=1, shuffle=False):
                    features_test, labels_test, kx_mask_test, mask_c_test = setup_inputs_test(y_test, mask, norm=None)
                    test_loss, prediction = sess.run([total_loss, y],
                                                     feed_dict={x: features_test, y_: labels_test,
                                                                x_k: kx_mask_test, mask_k: mask_c_test})
                    count += 1
                    print('The loss of NO. %2d test data is %.8f' % (count, test_loss))


                    pred_c = real2complex_array(prediction)
                    pred = np.squeeze(np.sqrt(np.sum(np.square(np.abs(pred_c)), axis=-1)))
                    plt.figure(1)
                    plt.imshow(pred, cmap='gray')
                    fig_name = os.path.join('./result', '%d_out_%d.png' % (count, i + 1))
                    plt.savefig(fig_name)

        coord.request_stop()
        coord.join(threads)
        loss_plot('batch')
        loss_plot('count')
        np.savetxt(os.path.join("./result", 'train_batch_loss.txt'), np.asarray(loss['batch']))
        np.savetxt(os.path.join("./result", 'train_count_loss.txt'), np.asarray(loss['count']))
        np.savetxt(os.path.join("./result", 'val_loss.txt'), np.asarray(val_loss))


def main(argv=None):
    train_data, validate_data, test_data, mask = load_data(data_path, BATCH_SIZE)
    train(train_data, validate_data,test_data, mask)

if __name__ == '__main__':
    tf.app.run()