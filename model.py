import tensorflow   as tf
from utils  import ComplexInit


def complex_conv2d(input,name,kw=3,kh=3,n_out=32,sw=1,sh=1,activation=True):
    n_in = input.get_shape()[-1].value // 2
    with tf.name_scope(name) as scope:
        kernel_init = ComplexInit(kernel_size=(kh,kw),
                                  input_dim=n_in,
                                  weight_dim=2,
                                  nb_filters=n_out,
                                  criterion='he')
        kernel = tf.get_variable(scope + 'weights',
                                 shape=[kh,kw,n_in,n_out],
                                 dtype=tf.float32,
                                 initializer=kernel_init)
        bias_init = tf.constant(0.0001,dtype=tf.float32,shape=[n_out*2])
        biases = tf.get_variable(scope+'biases', dtype=tf.float32, initializer=bias_init)
        kernel_real = kernel[:,:,:,:n_out]
        kernel_imag = kernel[:,:,:,n_out:]
        cat_kernel_real = tf.concat([kernel_real, -kernel_imag], axis=-2)
        cat_kernel_imag = tf.concat([kernel_imag, kernel_real], axis=-2)
        cat_kernel_complex = tf.concat([cat_kernel_real,cat_kernel_imag], axis=-1)
        conv = tf.nn.conv2d(input,cat_kernel_complex,strides=[1,sh,sw,1],padding='SAME')
        conv_bias = tf.nn.bias_add(conv,biases)
        if activation:
            act = tf.nn.relu(conv_bias)
            output = act

        else:
            output = conv_bias


        return output

def real2complex(x):
    channel = x.shape[-1] // 2
    if x.shape.ndims == 3:
        return tf.complex(x[:,:,:channel], x[:,:,channel:])
    elif x.shape.ndims == 4:
        return tf.complex(x[:,:,:,:channel], x[:,:,:,channel:])

def complex2real(x):
    x_real = tf.real(x)
    x_imag = tf.imag(x)
    return tf.concat([x_real,x_imag], axis=-1)

def dc(generated, X_k, mask):
    gene_complex = real2complex(generated)
    gene_complex = tf.transpose(gene_complex,[0, 3, 1, 2])
    mask = tf.transpose(mask,[0, 3, 1, 2])
    X_k = tf.transpose(X_k,[0, 3, 1, 2])
    gene_fft = tf.fft2d(gene_complex)
    out_fft = X_k + gene_fft * (1.0 - mask)
    output_complex = tf.ifft2d(out_fft)
    output_complex = tf.transpose(output_complex, [0, 2, 3, 1])
    output_real = tf.cast(tf.real(output_complex), dtype=tf.float32)
    output_imag = tf.cast(tf.imag(output_complex), dtype=tf.float32)
    output = tf.concat([output_real,output_imag], axis=-1)
    return output



def getModel(X, X_k, mask):
    temp = X
    for i in range(10):
        conv1 = complex_conv2d(temp, 'conv' + str(i + 1) + '1', kw=3, kh=3, n_out=32, sw=1, sh=1, activation=True)
        conv2 = complex_conv2d(conv1, 'conv' + str(i + 1) + '2', kw=3, kh=3, n_out=32, sw=1, sh=1, activation=True)
        conv3 = complex_conv2d(conv2, 'conv' + str(i + 1) + '3', kw=3, kh=3, n_out=32, sw=1, sh=1, activation=True)
        conv4 = complex_conv2d(conv3, 'conv' + str(i + 1) + '4', kw=3, kh=3, n_out=32, sw=1, sh=1, activation=True)
        conv5 = complex_conv2d(conv4, 'conv' + str(i + 1) + '5', kw=3, kh=3, n_out=12, sw=1, sh=1, activation=False)
        block = conv5 + temp
        temp = dc(block, X_k, mask)
    return temp