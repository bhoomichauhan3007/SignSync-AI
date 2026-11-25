import tensorflow as tf
from tensorflow.keras import layers, Model


def build_lstm(input_shape, n_classes):
    x_in=layers.Input(shape=input_shape)
    x=layers.Masking(mask_value=0.0)(x_in)
    x=layers.Bidirectional(layers.LSTM(96, return_sequences=True, dropout=.25))(x)
    x=layers.Bidirectional(layers.LSTM(64, dropout=.25))(x)
    x=layers.Dense(128, activation='relu')(x); x=layers.Dropout(.4)(x)
    out=layers.Dense(n_classes, activation='softmax')(x)
    return Model(x_in,out,name='word_lstm')


def build_gru(input_shape, n_classes):
    x_in=layers.Input(shape=input_shape)
    x=layers.Masking(mask_value=0.0)(x_in)
    x=layers.Bidirectional(layers.GRU(96, return_sequences=True, dropout=.25))(x)
    x=layers.Bidirectional(layers.GRU(64, dropout=.25))(x)
    x=layers.Dense(128, activation='relu')(x); x=layers.Dropout(.4)(x)
    out=layers.Dense(n_classes, activation='softmax')(x)
    return Model(x_in,out,name='word_gru')


def transformer_block(x, heads=4, key_dim=64, ff_dim=256, dropout=.2):
    att=layers.MultiHeadAttention(num_heads=heads,key_dim=key_dim,dropout=dropout)(x,x)
    x=layers.LayerNormalization(epsilon=1e-6)(x+layers.Dropout(dropout)(att))
    ff=layers.Dense(ff_dim,activation='gelu')(x); ff=layers.Dense(x.shape[-1])(ff)
    return layers.LayerNormalization(epsilon=1e-6)(x+layers.Dropout(dropout)(ff))


def build_transformer(input_shape,n_classes):
    inp=layers.Input(shape=input_shape)
    x=layers.Dense(256)(inp)
    pos=tf.range(start=0,limit=input_shape[0],delta=1)
    pe=layers.Embedding(input_dim=input_shape[0],output_dim=256)(pos)
    x=x+pe
    x=transformer_block(x); x=transformer_block(x)
    x=layers.GlobalAveragePooling1D()(x)
    x=layers.Dense(128,activation='gelu')(x); x=layers.Dropout(.4)(x)
    out=layers.Dense(n_classes,activation='softmax')(x)
    return Model(inp,out,name='word_transformer')
