import os
import librosa
import numpy as np
import mir_eval.separation
import time 
import sys
import matplotlib.pyplot as plt
sys.path.insert(0, 'Looking-to-Listen-at-the-Cocktail-Party-master/model/utils/')
import utils
#from pystoi import stoi
database = 'Looking-to-Listen-at-the-Cocktail-Party-master/data/AV_model_database/mix/'
print('Loading data ......')
test_file = []
with open('Looking-to-Listen-at-the-Cocktail-Party-master/data/AV_model_database/AVdataset_val.txt','r') as f:
    test_file = f.readlines()

 
def getSeparationMetrics(audio1, audio2, audio1_gt, audio2_gt):
	
	reference_sources = np.concatenate((np.expand_dims(audio1_gt, axis=0), np.expand_dims(audio2_gt, axis=0)), axis=0)
	estimated_sources = np.concatenate((np.expand_dims(audio1, axis=0), np.expand_dims(audio2, axis=0)), axis=0)
	(sdr, sir, sar, perm) = mir_eval.separation.bss_eval_sources(reference_sources, estimated_sources, False)
	return np.mean(sdr), np.mean(sir), np.mean(sar)

results_dir = './results/'
audio_sampling_rate = 16000
people = 2
os.makedirs(results_dir,exist_ok=True)
def main():
    for line in test_file:
        parts = line.split() # get each name of file for one testset
        mix_str = parts[0]
        name_list = mix_str.replace('.npy','')
        name_list = name_list.replace('mix-','',1)
        names = name_list.split('-')
        single_idxs = []
        for i in range(people):
            single_idxs.append(int(names[i]))
    audio1, _ = librosa.load('predict/%d.wav'%single_idxs[0], sr=audio_sampling_rate)
    audio2, _ = librosa.load('predict/%d.wav'%single_idxs[1], sr=audio_sampling_rate)
    audio1_gt, _ = librosa.load('audio_train/trim_audio_train%d.wav'%single_idxs[0], sr=audio_sampling_rate)
    audio2_gt, _ = librosa.load('audio_train/trim_audio_train%d.wav'%single_idxs[1], sr=audio_sampling_rate)
    audio_mix, _ = librosa.load('Looking-to-Listen-at-the-Cocktail-Party-master/data/AV_model_database/mix_wav/mix-%05d-%05d.wav'%(single_idxs[0], single_idxs[1]), sr=audio_sampling_rate)

	# SDR, SIR, SAR
    sdr, sir, sar = getSeparationMetrics(audio1, audio2, audio1_gt, audio2_gt)
    sdr_mixed, _, _ = getSeparationMetrics(audio_mix, audio_mix, audio1_gt, audio2_gt)

    # STOI
    #stoi_score1 = stoi(audio1_gt, audio1, audio_sampling_rate, extended=False)
    #stoi_score2 = stoi(audio2_gt, audio2, audio_sampling_rate, extended=False)
    #stoi_score = (stoi_score1 + stoi_score2) / 2

    output_file = open(os.path.join(results_dir, 'eval.txt'),'w')
    output_file.write("sdr, sdr_mixed, sdr - sdr_mixed, sir, sar\n")
    output_file.write("%3f %3f %3f %3f %3f" % (sdr, sdr_mixed, sdr - sdr_mixed, sir, sar))
    output_file.close()
    
    
    stft_frame = 1022
    stft_hop = 256
    plt.switch_backend('agg')
    plt.ioff()
    audio1_mag = utils.generate_spectrogram_magphase(audio1_gt, stft_frame, stft_hop, with_phase=False)
    audio2_mag = utils.generate_spectrogram_magphase(audio2_gt, stft_frame, stft_hop, with_phase=False)
    audio_mix_mag = utils.generate_spectrogram_magphase(audio_mix, stft_frame, stft_hop, with_phase=False)
    separation1_mag = utils.generate_spectrogram_magphase(audio1, stft_frame, stft_hop, with_phase=False)
    separation2_mag = utils.generate_spectrogram_magphase(audio2, stft_frame, stft_hop, with_phase=False)
    utils.visualizeSpectrogram(audio1_mag[0,:,:], os.path.join(results_dir, 'audio1_spec.png'))
    utils.visualizeSpectrogram(audio2_mag[0,:,:], os.path.join(results_dir, 'audio2_spec.png'))
    utils.visualizeSpectrogram(audio_mix_mag[0,:,:], os.path.join(results_dir, 'audio_mixed_spec.png'))
    utils.visualizeSpectrogram(separation1_mag[0,:,:], os.path.join(results_dir, 'separation1_spec.png'))
    utils.visualizeSpectrogram(separation2_mag[0,:,:], os.path.join(results_dir, 'separation2_spec.png'))
if __name__ == '__main__':
	main()