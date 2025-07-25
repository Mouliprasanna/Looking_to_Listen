import sys
#sys.path.append ('./ model / model')
sys.path.insert(0, 'Looking-to-Listen-at-the-Cocktail-Party-master/model/model')
#from keras.models import load_model
from option import ModelMGPU
import os
import scipy.io.wavfile as wavfile
import numpy as np
sys.path.insert(0, 'Looking-to-Listen-at-the-Cocktail-Party-master/model/utils/')
import utils
import AV_model as AV

#parameters
people = 2
num_gpu=1

#path
model_path = 'saved_AV_models\AVmodel-2p-048-0.54663.keras'
result_path = './predict/'
os.makedirs(result_path,exist_ok=True)

database = 'Looking-to-Listen-at-the-Cocktail-Party-master/data/AV_model_database/mix/'
face_emb = 'face1022_emb/'
print('Initialing Parameters......')

#loading data
print('Loading data ......')
test_file = []
with open('Looking-to-Listen-at-the-Cocktail-Party-master/data/AV_model_database/AVdataset_val.txt','r') as f:
    test_file = f.readlines()


def get_data_name(line,people=people,database=database):
    parts = line.split() # get each name of file for one testset
    mix_str = parts[0]
    name_list = mix_str.replace('.npy','')
    name_list = name_list.replace('mix-','',1)
    names = name_list.split('-')
    single_idxs = []
    for i in range(people):
        single_idxs.append(int(names[i]))
    print(single_idxs)
    file_path = database + mix_str
    mix = np.load(file_path)
    face_embs = np.zeros((1,75,1,2048,people), int)
    for i in range(people):
        face_embs[0,:,:,:,i] = np.load("face1022_emb/%05d_face_emb.npy"%single_idxs[i])
    return mix,single_idxs,face_embs

#result predict
model = AV.AV_model(people_num=people)
model.load_weights(model_path)
if num_gpu>1:
    parallel = ModelMGPU(model,num_gpu)
    for line in test_file:
        mix,single_idxs,face_emb = get_data_name(line,people,database,face_emb)
        mix_ex = np.expand_dims(mix,axis=0)
        cRMs = parallel.predict([mix_ex,face_emb])
        cRMs = cRMs[0]
        prefix =''
        for idx in single_idxs:
            prefix +=str(idx)+'-'
        for i in range(len(cRMs)):
            cRM =cRMs[:,:,:,i]
            assert cRM.shape ==(298,257,2)
            F = utils.fast_icRM(mix,cRM)
            T = utils.fast_istft(F,power=False)
            filename = result_path+str(single_idxs[i])+'.wav'
            T_scaled = (T / np.max(np.abs(T)) * 32767).astype(np.int16)  # Normalize and scale to int16
            wavfile.write(filename, 16000, T_scaled)

if num_gpu<=1:
    for line in test_file:
        mix,single_idxs,face_emb = get_data_name(line,people,database)
        mix_ex = np.expand_dims(mix,axis=0)
        cRMs = model.predict([mix_ex,face_emb])
        cRMs = cRMs[0]
        prefix =''
        for idx in single_idxs:
            prefix +=str(idx)+'-'
        for i in range(cRMs.shape[3]):
            cRM =cRMs[:,:,:,i]
            assert cRM.shape ==(298,257,2)
            F = utils.fast_icRM(mix,cRM)
            T = utils.fast_istft(F,power=False)
            filename = result_path+str(single_idxs[i])+'.wav'
            T_scaled = (T / np.max(np.abs(T)) * 32767).astype(np.int16)  # Normalize and scale to int16
            wavfile.write(filename, 16000, T_scaled)
