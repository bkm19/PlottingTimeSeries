########################################### Imports ###########################################
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imageio
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
import re
import math
import gc
import csv
import time
import os

from pathlib import Path
from zipfile import ZipFile
from PIL import Image
from sktime.datasets import load_from_tsfile_to_dataframe
from sklearn.model_selection import KFold

from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, MaxPooling2D, Flatten, Dense, InputLayer, BatchNormalization, Dropout
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras

########################################### Functions ###########################################

def pureBlackAndWhiteImageArrayUpdated( imageArray, normalized = True):
  aux = imageArray != 0
  result = aux.astype("uint8")[:,:,:1]
  return result if normalized else (result*255)

def get_order(file):
    match = re.compile(r'.*?(\d+).*?').match(Path(file).name)
    if not match:
        return math.inf
    return int(match.groups()[0])

def get_model(width, height, channels):
  model = Sequential()

  model.add(Conv2D(16, (1, 1), activation='relu', input_shape=(width, height, channels)))
  model.add(MaxPooling2D((2, 2)))
  model.add(Conv2D(32, (1, 1), activation='relu'))
  model.add(MaxPooling2D((2, 2)))
  model.add(Conv2D(64, (1, 1), activation='relu'))

  model.add(Flatten())

  model.add(Dense(256, activation='relu'))
  model.add(Dropout(0.7)) #Aumentar depois maybe o dropout
  model.add(Dense(128, activation='relu'))
  model.add(Dropout(0.7))
  model.add(Dense(n_classes, activation='softmax'))

  # early stopping
  callback = EarlyStopping(monitor='loss')
  # compile model
  opt = keras.optimizers.Adam(learning_rate=0.0001)

  model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  
  return model

def get_model2(width, height, channels):
  # define model
  model = Sequential()
  model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(width, height, channels)))
  model.add(MaxPooling2D((2, 2)))
  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(MaxPooling2D((2, 2)))
  model.add(Flatten())
  model.add(Dense(64, activation='relu'))
  model.add(Dense(n_classes, activation='softmax'))
  # early stopping
  callback = EarlyStopping(monitor='loss', patience=3)
  # compile model
  model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

  return model


########################################### Configs ###########################################

DATA_SET_NAMES = [
"ACSF1",
"Adiac",
"ArrowHead",
"Beef",
"BeetleFly",
"BirdChicken",
"BME",
"Car",
"CBF",
"Chinatown",
"ChlorineConcentration",
"CinCECGTorso",
"Coffee",
"Computers",
"Crop", #14
"DiatomSizeReduction",
"DistalPhalanxOutlineAgeGroup",
"DistalPhalanxOutlineCorrect",
"DistalPhalanxTW",
"Earthquakes",
"ECG200",
"ECG5000", #21,
"ECGFiveDays",
"ElectricDevices",
"EthanolLevel",
"FaceAll",
"FaceFour",
"FacesUCR", #27
"FiftyWords",
"Fish",
"FordA",
"FordB",
"FreezerRegularTrain", #32
"FreezerSmallTrain",
"GunPoint",
"GunPointAgeSpan",
"GunPointMaleVersusFemale",
"GunPointOldVersusYoung",
"Ham",
"Haptics",
"Herring",
"HouseTwenty",
"InlineSkate",
"InsectEPGRegularTrain",
"InsectEPGSmallTrain",
"ItalyPowerDemand",
"LargeKitchenAppliances", #46
"Lightning2",
"Lightning7",
"Mallat",
"Meat",
"MedicalImages", #51
"MiddlePhalanxOutlineAgeGroup",
"MiddlePhalanxOutlineCorrect",
"MiddlePhalanxTW",
"MixedShapesRegularTrain",
"MixedShapesSmallTrain",
"MoteStrain",
"OliveOil", #58
"OSULeaf",
"PhalangesOutlinesCorrect",
"Phoneme",
"PigAirwayPressure",
"PigArtPressure",
"PigCVP", #64
"Plane",
"ProximalPhalanxOutlineAgeGroup",
"ProximalPhalanxOutlineCorrect",
"ProximalPhalanxTW",
"RefrigerationDevices", #69
"Rock",
"ScreenType", #71
"SemgHandGenderCh2",
"SemgHandMovementCh2",
"SemgHandSubjectCh2",
"ShapeletSim",
"ShapesAll", #76
"SmallKitchenAppliances",
"SmoothSubspace",
"SonyAIBORobotSurface1",
"SonyAIBORobotSurface2", #80
"StarLightCurves", #81
"Strawberry",
"SwedishLeaf",
"Symbols",
"SyntheticControl",
"ToeSegmentation1",
"ToeSegmentation2",
"Trace",
"TwoLeadECG",
"TwoPatterns",
"UMD",
"UWaveGestureLibraryAll",
"Wafer", #93
"Wine",
"WordSynonyms",
"Worms",
"WormsTwoClass",
"Yoga"]


csv_name = 'deep_insight_results.csv'
#csv_name = '5_boxplots_no_fliers_results.csv'

pd.DataFrame(
    columns=
       ['Dataset',
         'Fold 1: Acc','Fold 1: Loss',
         'Fold 2: Acc','Fold 2: Loss',
         'Fold 3: Acc','Fold 3: Loss',
         'Fold 4: Acc','Fold 4: Loss',
         'Fold 5: Acc','Fold 5: Loss',
         'Average: Acc','Average: Loss',
         'Median',
         'Standard Deviation',
         'batch_size', 'epochs', 'ExecutionTime']
         ).to_csv(csv_name, index = False, header = True)

########################################### Loop Datasets ###########################################

for name in DATA_SET_NAMES:

  ################# CONFIGS #################
  print("Dataset name: ", name)
  start_time = time.time()

  normalized = False

  pathName = '/PATH_CHANGE/' + "DeepInsight" + '/'
  pathHistory = '/PATH_CHANGE/history/' + "DeepInsight" + '_' + name + '_history/'

  train_x = []
  test_x = []

  ################# LOAD DATASET #################

  _, train_y = load_from_tsfile_to_dataframe("/PATH_CHANGE/Univariate_ts/" + name + "/" + name + "_TRAIN.ts")
  _, test_y = load_from_tsfile_to_dataframe("/PATH_CHANGE/Univariate_ts/" + name + "/" + name + "_TEST.ts")

  train_x = np.empty((len(train_y), 50, 50, 3), dtype=np.uint8)
  test_x = np.empty((len(test_y), 50, 50, 3), dtype=np.uint8)

  trainOrTest = 'TRAIN'
  path = pathName + name + '/' + trainOrTest + '/' + '*.png*'

  for i, img_path in enumerate(sorted(glob.glob(path), key=get_order)):
    train_x[i] = cv2.imread(img_path)

  trainOrTest = 'TEST'
  path = pathName + name + '/' + trainOrTest + '/' + '*.png*'

  for i, img_path in enumerate(sorted(glob.glob(path), key=get_order)):
   #print(img_path)
   test_x[i] = cv2.imread(img_path)

  ################# PREPARE DATA #################

  train_y = train_y.astype('uint8')
  test_y = test_y.astype('uint8')

  while(min(train_y) > 0):
    train_y = train_y - 1
  while(min(test_y) > 0):
    test_y = test_y - 1

  n_classes = np.unique(train_y).size
  train_length, width, height, channels = train_x.shape[0], train_x.shape[1], train_x.shape[2], train_x.shape[3]

  datagen = ImageDataGenerator(rescale=1.0/255.0)

  ################# TRAINING #################

  #Configs
  #BATCH_SIZE = 32 if train_length >= 500 else 16 if train_length >= 50 else 8
  BATCH_SIZE = 16 if train_length >= 50 else 8
  EPOCHS = 200 if name == "Crop" else 250 #300
  N_SPLIT = 5
  verbose = 0

  # Storing the average of all predictions
  main_pred = []
  data_kfold = pd.DataFrame()
  acc_per_fold = []
  loss_per_fold = []

  row = []
  row.append(name)

  kfold = KFold(n_splits=N_SPLIT, shuffle=True)

  fold_no = 0

  for train_idx, val_idx in list(kfold.split(train_x, train_y)):

      fold_no+=1

      training_set = datagen.flow(train_x[train_idx], train_y[train_idx], batch_size=BATCH_SIZE)

      validation_set = datagen.flow(train_x[val_idx], train_y[val_idx], batch_size=BATCH_SIZE)

      model_test = get_model(width, height, channels)

      modelPath = "/PATH_CHANGE/saved_models/" + "DeepInsight" + '_' + name + '_fold' + str(fold_no) + '_model.h5'

      callbacks = [
      #keras.callbacks.ReduceLROnPlateau(
      #    monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
      #),
      #keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
       keras.callbacks.CSVLogger(pathHistory, separator = ',', append = True),
       keras.callbacks.ModelCheckpoint(modelPath,
              monitor='val_loss', verbose=0,
              save_best_only=True, save_weights_only=True, mode='min')
      ]

      #BECAUSE this datasets have val_loss nan and will not create model.h5
      if name in ["ECG200", "FordA", "FordB", "Lightning2", "Wafer"]:
       callbacks = [
        keras.callbacks.CSVLogger(pathHistory, separator = ',', append = True),
        keras.callbacks.ModelCheckpoint(modelPath,
              monitor='val_accuracy', verbose=0,
              save_best_only=True, save_weights_only=True, mode='max')
       ]

      history = model_test.fit( training_set,
                                validation_data=validation_set,
                                epochs = EPOCHS,
                                steps_per_epoch = len(training_set) ,
                                callbacks = callbacks,
                                verbose = verbose
                                )

      del(training_set)

      test_set = datagen.flow(test_x, test_y, batch_size=BATCH_SIZE)

      model_test.load_weights(modelPath)

      pred = model_test.evaluate(test_set, steps=len(test_set))

      acc_per_fold.append(pred[1])
      loss_per_fold.append(pred[0])

      del(test_set)

      os.remove(modelPath)

      keras.backend.clear_session()
      gc.collect()


  ################# OUTPUT #################


  for i in range(0, len(acc_per_fold)):
    row.append(acc_per_fold[i])
    row.append(loss_per_fold[i])

  row.append(np.mean(acc_per_fold))
  row.append(np.mean(loss_per_fold))
  row.append(np.median(acc_per_fold))
  row.append(np.std(acc_per_fold))

  row.append(BATCH_SIZE)
  row.append(EPOCHS)

  row.append(time.time() - start_time)


  ################# SAVE DATA #################

  with open(csv_name, 'a') as f:
      writer = csv.writer(f)
      writer.writerow(row)

  del(row, main_pred, data_kfold, acc_per_fold, loss_per_fold, datagen)
  del(train_x, test_x)
  gc.collect()
