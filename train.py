import argparse
import os
import sys
os.environ['PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT'] = '600.0'


import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_score, classification_report


import keras
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0 as EfficientNet
from tensorflow.keras.applications import ResNet50 as ResNet
from tensorflow.keras.applications import VGG16 as VGG


from notifier import *
from utils import *


from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder
from datetime import datetime


# ------------ Some parameters ------------ #
# csv_file = r"C:\Users\EEE\Downloads\Fitzpatrick17kImages\FitzpatrickData.csv"
# images_folder = r"C:\Users\EEE\Downloads\Fitzpatrick17kImages"

categoryName = "three_partition_label"
imageColName = "imgName"
fitzpatrickColName = "fitspatrick_scale"
fitzpatrickColIndex = 1


epochs = 10
batch_size = 32
input_shape = (224, 224, 3)
numOfDuplicates = 1



# ------------ Arguments ------------ #
parser = argparse.ArgumentParser()
parser.add_argument('--csv_file', type=str, help='csv file path')
parser.add_argument('--images_folder', type=str, help='images folder path')

parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--sample', action='store_true', help='Sample usage flag')  # This will be True if specified, otherwise False
parser.add_argument('--augAll', action='store_true', help='Augment data flag')  # This will be True if specified, otherwise False
parser.add_argument('--dupCount', type=int, default=0, help='Number of duplicates to create')
parser.add_argument('--sampleSize', type=str, default="F5", help='sample size')
parser.add_argument('--trainOnly', type=str, default="", help='only use F1 or F2 ... for training')
parser.add_argument('--trainOnlyPercent', type=int, default=100, help='only use the percent of F1 or F2 ... for training')
parser.add_argument('--trainOnlyCount', type=int, default=100, help='only use the Count of F1 or F2 ...for training')
parser.add_argument('--totalNumRuns', type=int, default=20, help='Total number of runs')
parser.add_argument('--augmentation', type=str, default="None", help='Type of data augmentation')
parser.add_argument('--aug_class', type=str, default="None", help='which skin class to augment')
parser.add_argument('--testName', type=str, help='Name of the test')
parser.add_argument('--output_folder', type=str, help='Output folder path')



args = parser.parse_args()

args.notifier_prefix = "test"
args.checkpoints_folder = "out"

args.batch_size = 32
batch_size = args.batch_size

# the following is for debug only
# args.sample = True
# args.totalNumRuns = 1
# args.augmentation = "clahe"
# args.sampleSize = "F6"
# args.augAll = True
# args.dupCount = 1
# #args.trainOnly = "F1"
# args.trainOnlyPercent = 25
# args.aug_class = "F6"

csv_file = args.csv_file 
images_folder = args.images_folder 

# Get the current date and time
current_datetime = datetime.now()


# Format the date and time as a string suitable for a filename
# For example, YYYYMMDD_HHMMSS
filename_datetime_string = current_datetime.strftime("%Y%m%d_%H%M%S")

if(args.trainOnly == ""): # if not train only F1 or F2,.....
    args.testName = "T_EffNet_" + "trainonly_" + str(args.trainOnly) + "_Percent_" + str(args.trainOnlyPercent) + "_TestSizeF6"  + filename_datetime_string 
elif(args.aug_class != "None"):
    args.testName = "T_EffNet_" + "_Runs_" + str(args.totalNumRuns) + "_" + args.augmentation + "_smpleS_" + str(args.sampleSize)+"_dupC_" +str(args.dupCount) + "_augClass_" +str(args.aug_class) + "_" + filename_datetime_string
else:
    if (args.trainOnlyCount > 0):
        args.testName = "T_EffNet_" + "trainonly_" + str(args.trainOnly) + "_Rs_" + str(args.totalNumRuns)  + "_trainOnlyCount_" + str(args.trainOnlyCount) + "_TestSizeF6"  + filename_datetime_string
    else:
        args.testName = "T_EffNet_" + "trainonly_" + str(args.trainOnly) + "_Rs_" + str(args.totalNumRuns)  + "_Percent_" + str(args.trainOnlyPercent) + "_TestSizeF6"  + filename_datetime_string
print(args.csv_file)
print(args.images_folder)
print(args.sample)
print(args.totalNumRuns)
print(args.sampleSize)
print(args.augmentation)
print(args.testName)
print(args.output_folder)
print(epochs)
print(args.augAll)
print(args.dupCount)
print(args.trainOnly)
print(args.aug_class)
print(args.trainOnlyPercent)

notifier = TelegramNotifier(prefix=args.notifier_prefix)
notifier.send_message("Started")


os.makedirs(args.checkpoints_folder, exist_ok=True)
os.makedirs("preloaded_data", exist_ok=True)

# Check if the output folder exists, and if not, create it.
if not os.path.exists(args.output_folder):
    os.makedirs(args.output_folder)
    print(f"The directory {args.output_folder} has been created.")
else:
    print(f"The directory {args.output_folder} already exists.")


# Backup the original stdout and stderr
original_stdout = sys.stdout


# ------------ Load images and metadata ------------ #
metadata = pd.read_csv(csv_file)

# Split the metadata into Fitzpatrick types
fitzpatrick1 = metadata[metadata[fitzpatrickColName] == 1]
fitzpatrick2 = metadata[metadata[fitzpatrickColName] == 2]
fitzpatrick3 = metadata[metadata[fitzpatrickColName] == 3]
fitzpatrick4 = metadata[metadata[fitzpatrickColName] == 4]
fitzpatrick5 = metadata[metadata[fitzpatrickColName] == 5]
fitzpatrick6 = metadata[metadata[fitzpatrickColName] == 6]


len_fitzpatrick6 = len(fitzpatrick6)


if (args.sampleSize == "F6"):
    # every skin type has the same number of samples as skin type 6
    numberofSample = len_fitzpatrick6
elif (args.sampleSize == "F5"):
    # every skin type has the same number of samples as skin type 5
    numberofSample = len(fitzpatrick5)
elif (args.sampleSize == "F4"):
    # every skin type has the same number of samples as skin type 4
    numberofSample = len(fitzpatrick4)
elif (args.sampleSize == "F3"):
    # every skin type has the same number of samples as skin type 3
    numberofSample = len(fitzpatrick3)
elif (args.sampleSize == "F2"):
    # every skin type has the same number of samples as skin type 2
    numberofSample = len(fitzpatrick2)
elif (args.sampleSize == "F1"):
    # every skin type has the same number of samples as skin type 1
    numberofSample = len(fitzpatrick1)


# if only train F1 or F2,....., the testing size will be all F6 size
if (args.trainOnly != ""):
    numberofTestingSample = int(len_fitzpatrick6 * 1 )
else:
    # Randomly sample 30% of the data without replacement
    numberofTestingSample = int(len_fitzpatrick6 * 0.3)

numberOfF6Training = len_fitzpatrick6 - numberofTestingSample

# if augment all, use F6 datasize, augment all, hence set training sample size to be the same as F6
if (args.augAll):
    numberofTrainingSample = len(fitzpatrick6) - numberofTestingSample
else:
    numberofTrainingSample = numberofSample-numberofTestingSample


# Initialize an empty list to store accuracy data for each iteration
accuracies = []

# Loop for the total number of runs
numRuns = 0
while (numRuns < args.totalNumRuns):
    output_file_path = os.path.join(args.output_folder + args.testName + str(numRuns) + '.txt')

    with open(output_file_path, 'w') as output_file:
        print (output_file_path)
        print(args.sample)
        print(args.totalNumRuns)
        print(args.sampleSize)
        print(args.augmentation)
        print(args.testName)
        print(epochs)
        print(args.augAll)
        print(args.dupCount)
        print(args.trainOnly)
        print(args.aug_class)
        print(args.trainOnlyPercent)

        sys.stdout = output_file

        if args.augAll or args.aug_class != "None":
            if args.dupCount > 0:
                numOfDuplicates = args.dupCount
            else:  
                numOfDuplicates = int((numberofSample - numberofTestingSample) / numberOfF6Training)

        # augment the data
        augmentSampleImage = pd.DataFrame()  # Initialize as an empty DataFrame
        augmented_Sample = pd.DataFrame()  # Initialize as an empty DataFrame


        # foreach fitzpatrick type, split the data into test and train
        if (args.aug_class == "None"): # if not only aug F1 or F2,.....
            fitzpatrick1_test = fitzpatrick1.sample(n=numberofTestingSample, random_state=None)  # Setting a random_state for reproducibility
            fitzpatrick1_rest = fitzpatrick1.drop(fitzpatrick1_test.index)

            fitzpatrick2_test = fitzpatrick2.sample(n=numberofTestingSample, random_state=None)  # Setting a random_state for reproducibility
            fitzpatrick2_rest = fitzpatrick2.drop(fitzpatrick2_test.index)

            fitzpatrick3_test = fitzpatrick3.sample(n=numberofTestingSample, random_state=None)  # Setting a random_state for reproducibility
            fitzpatrick3_rest = fitzpatrick3.drop(fitzpatrick3_test.index)

            fitzpatrick4_test = fitzpatrick4.sample(n=numberofTestingSample, random_state=None)  # Setting a random_state for reproducibility
            fitzpatrick4_rest = fitzpatrick4.drop(fitzpatrick4_test.index)

            fitzpatrick5_test = fitzpatrick5.sample(n=numberofTestingSample, random_state=None)  # Setting a random_state for reproducibility
            fitzpatrick5_rest = fitzpatrick5.drop(fitzpatrick5_test.index)

            fitzpatrick6_test = fitzpatrick6.sample(n=numberofTestingSample, random_state=None)  # Setting a random_state for reproducibility
            fitzpatrick6_rest = fitzpatrick6.drop(fitzpatrick6_test.index)
        else:
            numberofSample = len(fitzpatrick6) - numberofTestingSample
            if(args.aug_class == "F1"):
                fitzpatrick1_test = fitzpatrick1.sample(n=numberofTestingSample, random_state=None)
                fitzpatrick_rest = fitzpatrick1.drop(fitzpatrick1_test.index)
            if(args.aug_class == "F2"):
                fitzpatrick2_test = fitzpatrick2.sample(n=numberofTestingSample, random_state=None)
                fitzpatrick_rest = fitzpatrick2.drop(fitzpatrick2_test.index)
            if(args.aug_class == "F3"):
                fitzpatrick3_test = fitzpatrick3.sample(n=numberofTestingSample, random_state=None)
                fitzpatrick_rest = fitzpatrick3.drop(fitzpatrick3_test.index)
            if(args.aug_class == "F4"):
                fitzpatrick4_test = fitzpatrick4.sample(n=numberofTestingSample, random_state=None)
                fitzpatrick_rest = fitzpatrick4.drop(fitzpatrick4_test.index)
            if(args.aug_class == "F5"):
                fitzpatrick5_test = fitzpatrick5.sample(n=numberofTestingSample, random_state=None)
                fitzpatrick_rest = fitzpatrick5.drop(fitzpatrick5_test.index)
            if(args.aug_class == "F6"):
                fitzpatrick6_test = fitzpatrick6.sample(n=numberofTestingSample, random_state=None)
                fitzpatrick_rest = fitzpatrick6.drop(fitzpatrick6_test.index)

            fitzpatrick_rest = fitzpatrick_rest.sample(n=numberofSample, random_state=None)
            resultMetadata = fitzpatrick_rest
            augment_imageMetadata = duplicate_images(fitzpatrick_rest, imageColName, images_folder, numOfDuplicates, args.augmentation)
            augmented_Sample = pd.concat([augment_imageMetadata], ignore_index=True)
            print (f"Length of augmented_Sample: {len(augmented_Sample)}")

        # if not only train F1 or F2,.....
        if (args.trainOnly == "" and args.aug_class == "None"   ):

            # sample the rest
            if numberofTrainingSample < len(fitzpatrick1_rest) and args.sample:
                fitzpatrick1S = fitzpatrick1_rest.sample(numberofTrainingSample, random_state=None)
            else :
                fitzpatrick1S = fitzpatrick1_rest

            if numberofTrainingSample < len(fitzpatrick2_rest) and args.sample:
                fitzpatrick2S = fitzpatrick2_rest.sample(numberofTrainingSample, random_state=None)
            else :
                fitzpatrick2S = fitzpatrick2_rest

            if numberofTrainingSample < len(fitzpatrick3_rest) and args.sample:
                fitzpatrick3S = fitzpatrick3_rest.sample(numberofTrainingSample, random_state=None)
            else :
                fitzpatrick3S = fitzpatrick3_rest

            if numberofTrainingSample < len(fitzpatrick4_rest) and args.sample:
                fitzpatrick4S = fitzpatrick4_rest.sample(numberofTrainingSample, random_state=None)
            else :
                fitzpatrick4S = fitzpatrick4_rest

            if numberofTrainingSample < len(fitzpatrick5_rest) and args.sample:
                fitzpatrick5S = fitzpatrick5_rest.sample(numberofTrainingSample, random_state=None)
            else :
                fitzpatrick5S = fitzpatrick5_rest

            if numberofTrainingSample < len(fitzpatrick6_rest) and args.sample:
                fitzpatrick6S = fitzpatrick6_rest.sample(numberofTrainingSample, random_state=None)
            else :
                fitzpatrick6S = fitzpatrick6_rest

            if args.augAll:
                if args.dupCount > 0:
                    numOfDuplicates = args.dupCount
                else:  
                    numOfDuplicates = int((numberofSample - len(fitzpatrick1_test)) / len(fitzpatrick6_rest))

            if (args.aug_class == "None"):
                # since F6 has least amount of data, when augAll, numOfDuplicates is based on F6
                # the amount of duplicates is based on the numberofTrainingSample needed
                if (numberofTrainingSample > len(fitzpatrick6_rest) or args.augAll):
                    if not args.augAll:
                        numOfDuplicates = int(numberofTrainingSample/len(fitzpatrick6_rest))
                        augment_imageMetadata = duplicate_images(fitzpatrick6_rest, imageColName, images_folder, numOfDuplicates, args.augmentation)
                        augSampleNum = numberofTrainingSample - len(fitzpatrick6_rest)
                        augment_image = augment_imageMetadata.sample(augSampleNum, random_state=None)
                    else:
                        augment_imageMetadata = duplicate_images(fitzpatrick6S, imageColName, images_folder, numOfDuplicates, args.augmentation)
                        augment_image = augment_imageMetadata
                    augmentSampleImage = pd.concat([augmentSampleImage, augment_image], ignore_index=True)
                    print (f"Length of augmentSampleImage F6: {len(augmentSampleImage)}")
                if (numberofTrainingSample > len(fitzpatrick5_rest) or args.augAll):
                    if not args.augAll:
                        numOfDuplicates = int(numberofTrainingSample/len(fitzpatrick5_rest))
                        augment_imageMetadata = duplicate_images(fitzpatrick5_rest, imageColName, images_folder, numOfDuplicates, args.augmentation)
                        augSampleNum = numberofTrainingSample - len(fitzpatrick5_rest)
                        augment_image = augment_imageMetadata.sample(augSampleNum, random_state=None)
                    else:
                        augment_imageMetadata = duplicate_images(fitzpatrick5S, imageColName, images_folder, numOfDuplicates, args.augmentation)
                        augment_image = augment_imageMetadata
                    augmentSampleImage = pd.concat([augmentSampleImage, augment_image], ignore_index=True)
                    print (f"Length of augmentSampleImage F5: {len(augmentSampleImage)}")
                if (numberofTrainingSample > len(fitzpatrick4_rest) or args.augAll):
                    if not args.augAll:
                        numOfDuplicates = int(numberofTrainingSample/len(fitzpatrick4_rest))
                        augment_imageMetadata = duplicate_images(fitzpatrick4_rest, imageColName, images_folder, numOfDuplicates, args.augmentation)
                        augSampleNum = numberofTrainingSample - len(fitzpatrick4_rest)
                        augment_image = augment_imageMetadata.sample(augSampleNum, random_state=None)
                    else:
                        augment_imageMetadata = duplicate_images(fitzpatrick4S, imageColName, images_folder, numOfDuplicates, args.augmentation)
                        augment_image = augment_imageMetadata
                    augmentSampleImage = pd.concat([augmentSampleImage, augment_image], ignore_index=True)
                    print (f"Length of augmentSampleImage F4: {len(augmentSampleImage)}")
                if (numberofTrainingSample > len(fitzpatrick3_rest) or args.augAll):
                    if not args.augAll:
                        numOfDuplicates = int(numberofTrainingSample/len(fitzpatrick3_rest))
                        augment_imageMetadata = duplicate_images(fitzpatrick3_rest, imageColName, images_folder, numOfDuplicates, args.augmentation)
                        augSampleNum = numberofTrainingSample - len(fitzpatrick3_rest)
                        augment_image = augment_imageMetadata.sample(augSampleNum, random_state=None)
                    else:
                        augment_imageMetadata = duplicate_images(fitzpatrick3S, imageColName, images_folder, numOfDuplicates, args.augmentation)
                        augment_image = augment_imageMetadata
                    augmentSampleImage = pd.concat([augmentSampleImage, augment_image], ignore_index=True)
                    print (f"Length of augmentSampleImage F3: {len(augmentSampleImage)}")
                if (numberofTrainingSample > len(fitzpatrick2_rest) or args.augAll):
                    if not args.augAll:
                        numOfDuplicates = int(numberofTrainingSample/len(fitzpatrick2_rest))
                        augment_imageMetadata = duplicate_images(fitzpatrick2_rest, imageColName, images_folder, numOfDuplicates, args.augmentation)
                        augSampleNum = numberofTrainingSample - len(fitzpatrick2_rest)
                        augment_image = augment_imageMetadata.sample(augSampleNum, random_state=None)
                    else:
                        augment_imageMetadata = duplicate_images(fitzpatrick2S, imageColName, images_folder, numOfDuplicates, args.augmentation)
                        augment_image = augment_imageMetadata
                    augmentSampleImage = pd.concat([augmentSampleImage, augment_image], ignore_index=True)
                    print (f"Length of augmentSampleImage F2: {len(augmentSampleImage)}")
                if (numberofTrainingSample > len(fitzpatrick1_rest) or args.augAll):
                    if not args.augAll:
                        numOfDuplicates = int(numberofTrainingSample/len(fitzpatrick1_rest))
                        augment_imageMetadata = duplicate_images(fitzpatrick1_rest, imageColName, images_folder, numOfDuplicates, args.augmentation)
                        augSampleNum = numberofTrainingSample - len(fitzpatrick1_rest)
                        augment_image = augment_imageMetadata.sample(augSampleNum, random_state=None)
                    else:
                        augment_imageMetadata = duplicate_images(fitzpatrick1S, imageColName, images_folder, numOfDuplicates, args.augmentation)
                        augment_image = augment_imageMetadata
                    augmentSampleImage = pd.concat([augmentSampleImage, augment_image], ignore_index=True)
                    print (f"Length of augmentSampleImage F1: {len(augmentSampleImage)}")


            augmented_Sample = augmentSampleImage
            print (f"Length of augmented_Sample: {len(augmented_Sample)}")
            
            # combine the sample data of all the skin types. resultMetadata is the dataset we will use for training
            resultMetadata = pd.concat([fitzpatrick1S, fitzpatrick2S, fitzpatrick3S, fitzpatrick4S, fitzpatrick5S, fitzpatrick6S], ignore_index=True)

        elif (args.aug_class != "None"):  # if augment only F1 or F2,.....
            resultMetadata = pd.concat([fitzpatrick_rest, augmentSampleImage], ignore_index=True)

        else: # if only train F1 or F2,..., only get the training data from F1 or F2,.....
            if (args.trainOnlyCount > 0):
                if (args.trainOnly == "F1"):
                    trainOnlyCount = min(args.trainOnlyCount, len(fitzpatrick1_rest))
                    fitzpatrick1_rest = fitzpatrick1_rest.sample(n=trainOnlyCount, random_state=None)
                    resultMetadata = fitzpatrick1_rest
                if (args.trainOnly == "F2"):
                    trainOnlyCount = min(args.trainOnlyCount, len(fitzpatrick2_rest))
                    fitzpatrick2_rest = fitzpatrick2_rest.sample(n=trainOnlyCount, random_state=None)
                    resultMetadata = fitzpatrick2_rest
                if (args.trainOnly == "F3"):
                    trainOnlyCount = min(args.trainOnlyCount, len(fitzpatrick3_rest))
                    fitzpatrick3_rest = fitzpatrick3_rest.sample(n=trainOnlyCount, random_state=None)
                    resultMetadata = fitzpatrick3_rest
                if (args.trainOnly == "F4"):
                    trainOnlyCount = min(args.trainOnlyCount, len(fitzpatrick4_rest)) 
                    fitzpatrick4_rest = fitzpatrick4_rest.sample(n=trainOnlyCount, random_state=None)
                    resultMetadata = fitzpatrick4_rest
                if (args.trainOnly == "F5"):
                    trainOnlyCount = min(args.trainOnlyCount, len(fitzpatrick5_rest))
                    fitzpatrick5_rest = fitzpatrick5_rest.sample(n=trainOnlyCount, random_state=None)
                    resultMetadata = fitzpatrick5_rest
                if (args.trainOnly == "F6"):
                    trainOnlyCount = min(args.trainOnlyCount, len(fitzpatrick6_rest))
                    fitzpatrick6_rest = fitzpatrick6_rest.sample(n=trainOnlyCount, random_state=None)
                    resultMetadata = fitzpatrick6_rest
            else:  # training data by percent
                fraction = args.trainOnlyPercent/100
                #fraction = 0.625
                # only train F1 or F2,.....
                if (args.trainOnly == "F1"):
                    if (args.sample):
                        fitzpatrick1_rest = fitzpatrick1_rest.sample(numberofTrainingSample, random_state=None)
                    else:
                        fitzpatrick1_rest = fitzpatrick1_rest.sample(frac=fraction)
                    resultMetadata = fitzpatrick1_rest
                if (args.trainOnly == "F2"):
                    if (args.sample):
                        fitzpatrick2_rest = fitzpatrick2_rest.sample(numberofTrainingSample, random_state=None)
                    else:
                        fitzpatrick2_rest = fitzpatrick2_rest.sample(frac=fraction)
                    resultMetadata = fitzpatrick2_rest
                if (args.trainOnly == "F3"):
                    if (args.sample):
                        fitzpatrick3_rest = fitzpatrick3_rest.sample(numberofTrainingSample, random_state=None)
                    else:
                        fitzpatrick3_rest = fitzpatrick3_rest.sample(frac=fraction)
                    resultMetadata = fitzpatrick3_rest
                if (args.trainOnly == "F4"):
                    if (args.sample):
                        fitzpatrick4_rest = fitzpatrick4_rest.sample(numberofTrainingSample, random_state=None)
                    else:
                        fitzpatrick4_rest = fitzpatrick4_rest.sample(frac=fraction)
                    resultMetadata = fitzpatrick4_rest
                if (args.trainOnly == "F5"):
                    if (args.sample):
                        fitzpatrick5_rest = fitzpatrick5_rest.sample(numberofTrainingSample, random_state=None)
                    else:
                        fitzpatrick5_rest = fitzpatrick5_rest.sample(frac=fraction)
                    resultMetadata = fitzpatrick5_rest

                # if (args.trainOnly == "F6"):
                #     fitzpatrick6_rest = fitzpatrick6_rest.sample(frac=fraction)
                #     resultMetadata = fitzpatrick6_rest

        # create test data
        if (args.aug_class == "None"):
            testMetadata = pd.concat([fitzpatrick1_test, fitzpatrick2_test, fitzpatrick3_test, fitzpatrick4_test, fitzpatrick5_test, fitzpatrick6_test], ignore_index=True)
        else:
            if (args.aug_class == "F1"):
                testMetadata = fitzpatrick1_test
            if (args.aug_class == "F2"):
                testMetadata = fitzpatrick2_test
            if (args.aug_class == "F3"):
                testMetadata = fitzpatrick3_test
            if (args.aug_class == "F4"):
                testMetadata = fitzpatrick4_test
            if (args.aug_class == "F5"):
                testMetadata = fitzpatrick5_test
            if (args.aug_class == "F6"):
                testMetadata = fitzpatrick6_test

        # ------------ Preprocess images ------------ #
        # Sample code to add 'full_path' to the dataframe based on the file extension
        resultMetadata['full_path'] = resultMetadata[imageColName].apply(
            lambda x:
                os.path.join(images_folder, x) if x.endswith(".jpg") else
                os.path.join(images_folder, x + ".jpg")
        )
        resultMetadata['processed_image'] = resultMetadata['full_path'].apply(lambda x: load_and_preprocess_image(x))
        # combine resultmetadata with augmented sample


        if (len(augmented_Sample) > 0) :
            resultMetadata = pd.concat([resultMetadata, augmented_Sample], ignore_index=True)


        image_size = 224


        # Assuming 'processed_image' is a column containing image data,
        # 'categoryName' is the target category, and 'Fitzpatrick' is another attribute
        X = np.stack(resultMetadata['processed_image'].to_numpy())  # Convert images to a numpy array


        # Initialize LabelEncoder for the target category
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(resultMetadata[categoryName])
        class_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))


        # 'Fitzpatrick' is another attribute
        Fitzpatrick = resultMetadata[fitzpatrickColName].to_numpy()


        # Add other attributes to X
        other_attributes = resultMetadata.drop(['processed_image'], axis=1).to_numpy()



        # do the same for test data
        testMetadata['full_path'] = testMetadata[imageColName].apply(
            lambda x:
                os.path.join(images_folder, x) if x.endswith(".jpg") else
                os.path.join(images_folder, x + ".jpg")
        )
        testMetadata['processed_image'] = testMetadata['full_path'].apply(lambda x: load_and_preprocess_image(x))


        X_test = np.stack(testMetadata['processed_image'].to_numpy())  # Convert images to a numpy array


        # Initialize LabelEncoder for the target category
        label_encoder = LabelEncoder()
        y_test = label_encoder.fit_transform(testMetadata[categoryName])
        class_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

        # 'Fitzpatrick' is another attribute
        Fitzpatrick_test = testMetadata[fitzpatrickColName].to_numpy()

        # Add other attributes to X
        other_attributes_test = testMetadata.drop(['processed_image'], axis=1).to_numpy()

        # Use StratifiedShuffleSplit to split the data while preserving Fitzpatrick distribution
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state= None)
        train_indices, val_indices = next(sss.split(X, Fitzpatrick))

        # Split the data into training and testing sets based on the selected indices
        X_train, X_val, y_train, y_val, other_attributes_train, other_attributes_val = (
            X[train_indices],
            X[val_indices],
            y[train_indices],
            y[val_indices],
            other_attributes[train_indices],
            other_attributes[val_indices],
        )

        train_count, val_count, test_count = np.bincount(y_train), np.bincount(y_val), np.bincount(y_test)
        notifier.send_message("""Dataset split:
            Train set:      {} benign ({:.2%}), {} malignant ({:.2%})
            Validation set: {} benign ({:.2%}), {} malignant ({:.2%})
            Test set:       {} benign ({:.2%}), {} malignant ({:.2%})
        """.format(
            train_count[0], train_count[0]/sum(train_count), train_count[1], train_count[1]/sum(train_count),
            val_count[0],   val_count[0]/sum(val_count),     val_count[1],   val_count[1]/sum(val_count),
            test_count[0],  test_count[0]/sum(test_count),   test_count[1],  test_count[1]/sum(test_count)
        ))

        # ------------ Training ------------ #
        notifier.send_message("Starting training model")

        mirrored_strategy = tf.distribute.MirroredStrategy()

        with mirrored_strategy.scope():
            efficientnet = EfficientNet(weights='imagenet', include_top=False, input_shape=input_shape, classes=2)
            model = keras.models.Sequential()
            model.add(efficientnet)
            model.add(keras.layers.GlobalAveragePooling2D())
            model.add(keras.layers.Dense(1, activation='sigmoid'))
            model.summary()

            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

        # Early stopping to monitor the validation loss and avoid overfitting
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50, restore_best_weights=True)


        # Reducing learning rate on plateau
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', mode='min', patience=5, factor=0.5, verbose=1)

        # Checkpoint callback
        checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=args.checkpoints_folder + "/checkpoint.epoch{epoch:02d}-loss{val_loss:.2f}.hdf5",
            save_weights_only=False,
            monitor='val_binary_accuracy',
            mode='max',
            save_best_only=True
        )

        callbacks = [Notify(epochs, args.notifier_prefix), early_stop, reduce_lr, checkpoint]

        # train the data
        try:
            history = model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                verbose=2,
                callbacks=callbacks,
                shuffle=True,
                # class_weight={0: 1, 1: 8}
            )
        except:
            pass

        # print the results
        print ("===================================================================")
        print(class_mapping)


        # Evaluation & post-training analysis
        # total accuracy of all test data
        evlResult = model.evaluate(X_test, y_test)
        loss, overall_accuracy = evlResult[0], evlResult[1]
        print("Accuracy for test data : ", overall_accuracy)

        test_predictions = model.predict(X_test)


        # Convert binary predictions back to class names
        binary_predictions = (test_predictions >= 0.5).astype(int)
        # Assuming y_test_binary_predictions contains binary predictions (0 or 1)
        y_test_class_predictions = [next(class_name for class_name, class_value in class_mapping.items() if class_value == pred) for pred in binary_predictions]
        y_test_class_truth = [next(class_name for class_name, class_value in class_mapping.items() if class_value == val) for val in y_test]


        # Convert other_attributes_test to a DataFrame
        df_other_attributes_test = pd.DataFrame(other_attributes_test)
        # Add a new column to the DataFrame using y_test_class_predictions
        df_other_attributes_test[categoryName] = y_test_class_truth
        df_other_attributes_test['predicted'] = y_test_class_predictions
        df_other_attributes_test['Fitzpatrick'] = other_attributes_test[:, fitzpatrickColIndex]

        df_results = df_other_attributes_test.copy()

        test_preds_number = test_predictions.argmax(axis=1)
        classes = np.argmax(test_predictions, axis = 1)
        uniqueDiagnostics =  df_other_attributes_test[categoryName].unique()
        test_preds = uniqueDiagnostics[test_preds_number]

        # Group predictions by Fitzpatrick scale and compute accuracy
        df_results['correct'] = df_results[categoryName] == df_results['predicted']
        fitzpatrick_accuracies = {}
        for fitz_val in range(1, 7):
            subset = df_results[df_results['Fitzpatrick'] == fitz_val]
            print ("-------------------------------------------------------------------")
            print (subset[categoryName])
            print (subset['predicted'])
            accuracy = accuracy_score(subset[categoryName], subset['predicted'])
            fitzpatrick_accuracies[f'Fitzpatrick{fitz_val}'] = accuracy
            print(f"Fitzpatrick{fitz_val}: {accuracy:.2f}")

            print ("-------------------------------------------------------------------")

        # Append the accuracies to the list
        accuracies.append({'OverallAccuracy': overall_accuracy, **fitzpatrick_accuracies})
        numRuns += 1
        # Restore the original stdout and stderr
        sys.stdout = original_stdout


# After the while loop, convert the list of accuracies to a DataFrame
df_accuracies = pd.DataFrame(accuracies)


# Save the DataFrame to a CSV file
csv_file_path = args.output_folder + args.testName + "combined.csv"
df_accuracies.to_csv(csv_file_path, index=False)
