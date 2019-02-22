import tensorflow as tf
import h5py
import os

model_path = '../scripts/saved_model'



def test():

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(os.path.join(model_path, 'model.ckpt-236000.meta'))
        saver.restore(sess, os.path.join(model_path, 'model.ckpt-236000'))
        save_weights_to_hdf5('./demo.h5')




def save_weights_to_hdf5(savepath):
    trainable_variables = [v for v in tf.trainable_variables()]
    with h5py.File(os.path.join(savepath),'w') as f:
        for v in trainable_variables:
            v_value = v.eval()
            v_name = v._shared_name
            # v_shape = v_value.shape
            f.create_dataset(v_name,data=v_value)

if __name__ == '__main__':
    test()