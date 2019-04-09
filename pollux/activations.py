from tensorflow.keras import backend as K
from tensorflow.keras.layers import Activation, PReLU, LeakyReLU
from tensorflow.keras.utils import get_custom_objects

def swish(x):
    return (K.sigmoid(x) * x)

def retanh(x):
    return K.clip(K.tanh(x),0,1) 

def softplus_eps(x):
    return K.softplus(x) + K.epsilon()
    
get_custom_objects().update({'swish': swish, 'retanh': retanh, 'softplus_eps':softplus_eps})

def add_activation(x, activation):
    if activation in ['relu', 'selu', 'elu', 'sigmoid', 'hard_sigmoid', 'linear', 'tanh', 'softplus', 'softsign', 'softmax', 'swish', 'retanh', 'softplus_eps']:
        return Activation(activation)(x)
    elif activation == 'LeakyReLU':
        return LeakyReLU()(x)
    elif activation == 'PReLU':
        return PReLU()(x)
    else:
        raise NotImplemented
