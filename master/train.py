import sys
import os
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback
from keras.models import Model, load_model
from model.model.data_load import AVGenerator
from keras.callbacks import TensorBoard
from keras import optimizers
import matplotlib.pyplot as plt
from model.model.loss import audio_discriminate_loss2 as audio_loss
import model.model.AV_model as AV
# Parameters
people_num = 2
epochs = 5 
initial_epoch = 0
batch_size = 1
gamma_loss = 0.1
beta_loss = gamma_loss * 2

# Accelerate Training Process
workers = 8
MultiProcess = True
NUM_GPU = 1  # Modify this to use multiple GPUs if necessary (NUM_GPU > 1)

# PATH
model_path = './saved_AV_models'  # model path
database_path = 'Looking-to-Listen-at-the-Cocktail-Party-master/data/AV_model_database/'

# create folder to save models
folder = os.path.exists(model_path)
if not folder:
    os.makedirs(model_path)
    print('create folder to save models')
filepath = model_path + "/AVmodel-" + str(people_num) + "p-{epoch:03d}-{val_loss:.5f}.keras"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# automatically change lr
def scheduler(epoch):
    ini_lr = 0.001
    lr = ini_lr
    if epoch >= 25:
        lr = ini_lr / 5
    if epoch >= 50:
        lr = ini_lr / 10
    return lr

rlr = LearningRateScheduler(scheduler, verbose=1)

# Load dataset
trainfile = []
valfile = []
with open((database_path + 'AVdataset_train.txt'), 'r') as t:
    trainfile = t.readlines()
with open((database_path + 'AVdataset_val.txt'), 'r') as v:
    valfile = v.readlines()

# Resume Model
resume_state = False

if resume_state:
    latest_file = latest_file(model_path + '/')
    AV_model = load_model(latest_file, custom_objects={"tf": tf})
    info = latest_file.strip().split('-')
    initial_epoch = int(info[-2])
else:
    AV_model = AV.AV_model(people_num)

train_generator = AVGenerator(trainfile, database_path=database_path, batch_size=batch_size, shuffle=True)
val_generator = AVGenerator(valfile, database_path=database_path, batch_size=batch_size, shuffle=True)

# Setting up the GPU configuration
if NUM_GPU > 1:
    # Use TensorFlow MirroredStrategy for multi-GPU
    strategy = tf.distribute.MirroredStrategy()
    print(f"Using {NUM_GPU} GPUs")

    with strategy.scope():
        adam = optimizers.Adam()
        loss = audio_loss(gamma=gamma_loss, beta=beta_loss, people_num=people_num)
        AV_model.compile(loss=loss, optimizer=adam)
        print(AV_model.summary())

        def main():
            history = AV_model.fit(
                train_generator,
                validation_data=val_generator,
                epochs=epochs,
                workers=workers,
                use_multiprocessing=MultiProcess,
                callbacks=[TensorBoard(log_dir='./log_AV'), checkpoint, rlr],
                initial_epoch=initial_epoch
            )
            # Plotting accuracy
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.title('Model Accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='upper left')
            plt.show()

            # Plotting loss
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='upper left')
            plt.show()

        if __name__ == '__main__':
            main()

else:  # Use a single GPU or CPU
    adam = optimizers.Adam()
    loss = audio_loss(gamma=gamma_loss, beta=beta_loss, people_num=people_num)
    AV_model.compile(optimizer=adam, loss=loss, metrics=['accuracy'])
    print(AV_model.summary())

    def main():
        history = AV_model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=[TensorBoard(log_dir='./log_AV'), checkpoint, rlr],
            initial_epoch=initial_epoch
        )
        # Plotting accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

        # Plotting loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

    if __name__ == '__main__':
        main()
