from tensorflow.keras import backend as K
from tensorflow.keras.utils import get_custom_objects

def swish(x):
    return (K.sigmoid(x) * x)

def retanh(x):
    return K.clip(K.tanh(x),0,1) 

def softplus_eps(x):
    return K.softplus(x) + K.epsilon()
    
get_custom_objects().update({'swish': swish, 'retanh': retanh, 'softplus_eps':softplus_eps})
