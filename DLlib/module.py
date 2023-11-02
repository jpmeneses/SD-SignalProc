import tensorflow as tf
import tensorflow.keras as keras

def _get_norm_layer(norm):
    if norm == 'none':
        return lambda: lambda x: x
    elif norm == 'batch_norm':
        return keras.layers.BatchNormalization
    elif norm == 'layer_norm':
        return keras.layers.LayerNormalization


def _upsample(filters, kernel_size, strides, padding, method='Conv2DTranspose'):
    if method == 'Conv2DTranspose':
        op = keras.layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)
    elif method == "Interpol_Conv":
        op = keras.Sequential()
        op.add(keras.layers.UpSampling2D(size=(2,2),interpolation='nearest'))
        op.add(keras.layers.Conv2D(filters, kernel_size, strides=1, padding=padding))
    return op


def _residual_block(x, norm):
    Norm = _get_norm_layer(norm)
    dim = x.shape[-1]
    h = x

    h = keras.layers.Conv2D(dim, 3, kernel_initializer='he_normal', padding='same', use_bias=False)(h)
    h = Norm()(h)
    h = tf.nn.leaky_relu(h)

    h = keras.layers.Conv2D(dim, 3, kernel_initializer='he_normal', padding='same', use_bias=False)(h)
    h = Norm()(h)

    return keras.layers.add([x, h])


def encoder(
    input_shape,
    encoded_dims,
    filters=64,
    num_layers=4,
    num_res_blocks=2,
    dropout=0.0,
    norm='batch_norm'):

    x = inputs = keras.Input(input_shape)
    x = keras.layers.Conv2D(filters,3,padding="same",activation=tf.nn.leaky_relu,kernel_initializer="he_normal")(x)

    for l in range(num_layers):
        for n_res in range(num_res_blocks):
            x = _residual_block(x, norm=norm)

        # Double the number of filters and downsample
        filters = filters * 2  
        x = keras.layers.Conv2D(filters,3,strides=2,padding="same",activation=tf.nn.leaky_relu,kernel_initializer="he_normal")(x)

    x = keras.layers.Conv2D(encoded_dims,3,padding="same",activation=tf.nn.leaky_relu,kernel_initializer="he_normal")(x)
    _,ls_hgt,ls_wdt,ls_dims = x.shape

    output = keras.layers.Conv2D(encoded_dims,1,padding="same")(x)

    return keras.Model(inputs=inputs, outputs=output)


def decoder(
    encoded_dims,
    output_shape,
    filters=36,
    num_layers=4,
    num_res_blocks=2,
    dropout=0.0,
    output_activation=None,
    output_initializer='glorot_normal',
    norm='batch_norm'):
    Norm = _get_norm_layer(norm)

    hgt,wdt,n_out = output_shape
    hls = hgt//(2**(num_layers))
    wls = wdt//(2**(num_layers))
    filt_ini = filters*(2**num_layers)
    input_shape = (hls,wls,encoded_dims)
    
    x = inputs1 = keras.Input(input_shape)

    filt_iter = filt_ini
    x = keras.layers.Conv2D(encoded_dims,3,padding="same",activation=tf.nn.leaky_relu,kernel_initializer='he_normal')(x)
    x = keras.layers.Conv2D(filt_iter,3,padding="same",activation=tf.nn.leaky_relu,kernel_initializer='he_normal')(x)
    for cont in range(num_layers):
        filt_iter //= 2  # decreasing number of filters with each layer
        x = _upsample(filt_iter, (2, 2), strides=(2, 2), padding='same', method='Interpol_Conv')(x)
        for n_res in range(num_res_blocks):
            x = _residual_block(x, norm=norm)

    x = Norm()(x)
    output = keras.layers.Conv2D(n_out,3,padding="same",activation=output_activation,kernel_initializer=output_initializer)(x)
    
    return keras.Model(inputs=inputs1, outputs=output)