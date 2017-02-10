from keras import backend as K
from keras.layers import Layer
from numpy import float32

def power_mean_betas(input,p,along_axis):
    #return T.max(x,axis=1) * (1.0**p)
    temp=K.reshape(input* p[None,:,None,None], (input.shape[0]*input.shape[1], input.shape[2]*input.shape[3]))
    x=K.reshape(input, (input.shape[0]*input.shape[1], input.shape[2]*input.shape[3]))
    output = K.sum(K.theano.tensor.nnet.softmax(temp) * x , axis=along_axis)
    return K.reshape(output,(input.shape[0], input.shape[1],1))


def power_mean(input,along_axis):
    #return T.max(x,axis=1) * (1.0**p)
    #temp=K.reshape(input* p[None,:,None,None], (input.shape[0]*input.shape[1], input.shape[2]*input.shape[3]))
    x=K.reshape(input, (input.shape[0]*input.shape[1], input.shape[2]*input.shape[3]))
    output = K.mean(x , axis=along_axis)
    return K.reshape(output,(input.shape[0], input.shape[1],1))

# implemented for keras version 0.3.3. It changes with recent version.

class power_mean_Layer(Layer):
    def __init__(self, betas, **kwargs):
        self.betas = float32(betas)
        super(power_mean_Layer, self).__init__(**kwargs)
    def get_output(self, train=False):  
        x=self.get_input(train)
        return power_mean_betas(x,self.betas,1)
    @property
    def output_shape(self):
        input_shape=self.input_shape
        return (input_shape[0], input_shape[1],1)


class global_Average_Pooling(Layer):
    def __init__(self, **kwargs):
        #self.betas = float32(betas)
        super(global_Average_Pooling, self).__init__(**kwargs)
    def get_output(self, train=False):  
        x=self.get_input(train)
        #K.mean(x,axis=1)
        return power_mean(x,1)
    @property
    def output_shape(self):
        input_shape=self.input_shape
        return (input_shape[0], input_shape[1],1)

#for new version
# from keras.engine.topology import Layer
# class power_mean_Layer(Layer):
#     def __init__(self, betas, **kwargs):
#         self.betas = float32(betas)
#         super(power_mean_Layer, self).__init__(**kwargs)
#     def call(self, x,mask=None):  
#         #x=self.get_input(train)
#         return power_mean_betas(x,self.betas,1)
#     def get_output_shape_for(self,input_shape):
#         #input_shape=self.input_shape
#         return (input_shape[0], input_shape[1],1)