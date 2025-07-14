import sys
sys.path.insert(0, 'Looking-to-Listen-at-the-Cocktail-Party-master/model/model')
import AV_model as AV
model_path = 'saved_AV_models/AVmodel-2p-001-0.82290.h5'
model = AV.AV_model(people_num=2)
model.load_weights(model_path)
#av_model = load_model(model_path,custom_objects={'tf':tf})
def get_model_memory_usage(batch_size, model):
    import numpy as np
    from keras import backend as K
    
    shapes_mem_count = 0
    internal_model_mem_count = 0
    for l in model.layers:
        layer_type = l.__class__.__name__
        if layer_type == 'Model':
            internal_model_mem_count += get_model_memory_usage(batch_size, l)
        single_layer_mem = 1
        out_shape = l.output_shape
        if type(out_shape) is list:
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(p) for p in model.non_trainable_weights])

    number_size = 4.0
    if K.floatx() == 'float16':
        number_size = 2.0
    if K.floatx() == 'float64':
        number_size = 8.0

    total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
    return gbytes

print(get_model_memory_usage(1, model))