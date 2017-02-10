from keras.optimizers import SGD,Adam

optimizer_dict = {'sgd':SGD(lr=compile_cfgs['lr'], decay=compile_cfgs['decay'], momentum=compile_cfgs['momentum'], nesterov=True),'adam':Adam()}