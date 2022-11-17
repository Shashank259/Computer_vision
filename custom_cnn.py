import pandas as pd       
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , BatchNormalization
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import warnings
warnings.filterwarnings('ignore')
import random
import os
import glob
from numpy.random import seed
seed(42)
random.seed(42)
os.environ['PYTHONHASHSEED'] = str(42)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
from tensorflow.random import set_seed
set_seed(42)

#constants
IMG_SIZE = 224
BATCH = 32
SEED = 42
MAIN_DIR = "/home/jupyter/project_xray_image/chest_xray"


def data_load_augment(main_path):
    train_path = os.path.join(main_path,"train")
    test_path=os.path.join(main_path,"test")
    val_path=os.path.join(main_path,"val")
    train_normal = glob.glob(train_path+"/NORMAL/*.jpeg")
    train_pneumonia = glob.glob(train_path+"/PNEUMONIA/*.jpeg")
    test_normal = glob.glob(test_path+"/NORMAL/*.jpeg")
    test_pneumonia = glob.glob(test_path+"/PNEUMONIA/*.jpeg")
    val_normal = glob.glob(val_path+"/NORMAL/*.jpeg")
    val_pneumonia = glob.glob(val_path+"/PNEUMONIA/*.jpeg")
    train_list = [x for x in train_normal]
    train_list.extend([x for x in train_pneumonia])
    df_train = pd.DataFrame(np.concatenate([['Normal']*len(train_normal) , ['Pneumonia']*len(train_pneumonia)]), columns = ['class'])
    df_train['image'] = [x for x in train_list]
    test_list = [x for x in test_normal]
    test_list.extend([x for x in test_pneumonia])
    df_test = pd.DataFrame(np.concatenate([['Normal']*len(test_normal) , ['Pneumonia']*len(test_pneumonia)]), columns = ['class'])
    df_test['image'] = [x for x in test_list]
    val_list = [x for x in val_normal]
    val_list.extend([x for x in val_pneumonia])
    df_val = pd.DataFrame(np.concatenate([['Normal']*len(val_normal) , ['Pneumonia']*len(val_pneumonia)]), columns = ['class'])
    df_val['image'] = [x for x in val_list]
    train_df, val_df = train_test_split(df_train, test_size = 0.20, random_state = SEED, stratify = df_train['class'])
    train_datagen = ImageDataGenerator(rescale=1/255.,
                                      zoom_range = 0.1,
                                      width_shift_range = 0.1,
                                      height_shift_range = 0.1)
    val_datagen = ImageDataGenerator(rescale=1/255.)
    ds_train = train_datagen.flow_from_dataframe(train_df,
                                                 x_col = 'image',
                                                 y_col = 'class',
                                                 target_size = (IMG_SIZE, IMG_SIZE),
                                                 class_mode = 'binary',
                                                 batch_size = BATCH,
                                                 seed = SEED)
    ds_val = val_datagen.flow_from_dataframe(val_df,
                                                x_col = 'image',
                                                y_col = 'class',
                                                target_size = (IMG_SIZE, IMG_SIZE),
                                                class_mode = 'binary',
                                                batch_size = BATCH,
                                                seed = SEED)
    ds_test = val_datagen.flow_from_dataframe(df_test,
                                                x_col = 'image',
                                                y_col = 'class',
                                                target_size = (IMG_SIZE, IMG_SIZE),
                                                class_mode = 'binary',
                                                batch_size = 1,
                                                shuffle = False)
    return train_df, val_df, ds_train, ds_val, ds_test




def set_callbacks():
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        min_delta=1e-7,
        restore_best_weights=True)
    plateau = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor = 0.2,                                     
        patience = 2,                                   
        min_delt = 1e-7,                                
        cooldown = 0,                               
        verbose = 1 )
    return early_stopping, plateau




def get_model(IMG_SIZE):
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    # Block One
    x = layers.Conv2D(filters=16, kernel_size=3, padding='valid')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPool2D()(x)
    x = layers.Dropout(0.2)(x)
    # Block Two
    x = layers.Conv2D(filters=32, kernel_size=3, padding='valid')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPool2D()(x)
    x = layers.Dropout(0.2)(x)
    # Block Three
    x = layers.Conv2D(filters=64, kernel_size=3, padding='valid')(x)
    x = layers.Conv2D(filters=64, kernel_size=3, padding='valid')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPool2D()(x)
    x = layers.Dropout(0.4)(x)
    # Head
    #x = layers.BatchNormalization()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    #Final Layer (Output)
    output = layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs=[inputs], outputs=output)
    return model


def compile_train_model(train_df, val_df, ds_train, ds_val, early_stopping, plateau):
    keras.backend.clear_session()
    model = get_model()
    model.compile(loss='binary_crossentropy'
                  , optimizer = keras.optimizers.Adam(learning_rate=3e-5), metrics='binary_accuracy')
    model.summary()
    model.fit(ds_train,
              batch_size = BATCH, epochs = 10,
              validation_data=ds_val,
              callbacks=[early_stopping, plateau],
              steps_per_epoch=(len(train_df)/BATCH),
              validation_steps=(len(val_df)/BATCH));
    model.save("xray_image_classification_cnn")
    return model

def evaluate_model(model, ds_val, ds_test, val_df, df_test):
    score_val = model.evaluate(ds_val, steps = len(val_df)/BATCH, verbose = 0)
    val_loss = score_val[0]
    val_accuracy= score_val[1]
    score_test = model.evaluate(ds_test, steps = len(df_test), verbose = 0)
    test_loss = score_test[0]
    test_accuracy = score_test[1]
    num_label = {'Normal': 0, 'Pneumonia' : 1}
    Y_test = df_test['class'].copy().map(num_label).astype('int')
    ds_test.reset()
    predictions = model.predict(ds_test, steps=len(ds_test), verbose=0)
    pred_labels= np.where(predictions>0.5, 1, 0)
    confusion_matrix = metrics.confusion_matrix(Y_test, pred_labels)
    clf_report = classification_report(Y_test, pred_labels, target_names = ['Pneumonia (Class 0)','Normal (Class 1)'])
    return val_loss, val_accuracy, test_loss, test_accuracy, confusion_matrix, clf_report

if __name__ == "__main__":
    main_path = MAIN_DIR
    train_df, val_df, ds_train, ds_val, ds_test = data_load_augment(main_path)
    early_stopping, plateau = set_callbacks()
    model = get_model(IMG_SIZE)
    model = compile_train_model(train_df, 
                                val_df, 
                                ds_train, 
                                ds_val, 
                                early_stopping, 
                                plateau)
    val_loss, val_accuracy, test_loss, test_accuracy, confusion_matrix, clf_report = evaluate_model(model, 
                                                                                                    ds_val, 
                                                                                                    ds_test, 
                                                                                                    val_df, 
                                                                                                    df_test)
    print('Val loss:', val_loss)
    print('Val accuracy:', val_accuracy)
    print('Test loss:', test_loss)
    print('Test accuracy:', test_accuracy)
    
    sns.heatmap(confusion_matrix, annot=True, fmt="d")
    plt.xlabel("Predicted Label", fontsize= 12)
    plt.ylabel("True Label", fontsize= 12)
    plt.show()
    
    print(clf_report)


