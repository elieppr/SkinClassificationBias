import argparse
import os
os.environ['PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT'] = '600.0'

import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_score, classification_report
from sklearn.model_selection import train_test_split

import keras
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0 as EfficientNet

from notifier import *
from utils import *

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder

# ------------ Some parameters ------------ #
# csv_file = "dataset/ISIC_2020_Training_GroundTruth_v2.csv"
# csv_file = "C:\Users\Eliana\Downloads\Fitzpatrick17kImages\FitzpatrickData.csv"
# duplicates_csv_file = "dataset/ISIC_2020_Training_Duplicates.csv"
#images_folder = "dataset/train_jpeg/"
#images_folder = "dataset/train/"
# categoryName = "benign_malignant"
# imageColName = "image_name"
# fitzpatrickColName = "Fitzpatrick"

csv_file = r"C:\Users\Eliana\Downloads\Fitzpatrick17kImages\FitzpatrickData.csv"
images_folder = r"C:\Users\Eliana\Downloads\Fitzpatrick17kImages"
categoryName = "three_partition_label"
imageColName = "imgName"
fitzpatrickColName = "fitspatrick_scale"
fitzpatrickColIndex = 1

#epochs = 300
epochs = 10
#batch_size = 256
batch_size = 32
input_shape = (224, 224, 3)
#numOfDuplicates = 6
numOfDuplicates = 2


# ------------ Arguments ------------ #
parser = argparse.ArgumentParser()
parser.add_argument("--remove-artifacts", action="store_true", help="Perform artifact removal using morphological closing")
parser.add_argument("--segmentation", action="store_true", help="Perform segmentation using the k-means algorithm")
parser.add_argument("--checkpoint-folder", default="checkpoints", type=str, dest="checkpoints_folder", help="Folder to save model checkpoints and final model")
parser.add_argument("--notifier-prefix", default=None, type=str)

parser.add_argument("--epochs", type=int, default=epochs, help="Maximum number of epochs")
parser.add_argument("--batch-size", type=int, default=batch_size, help="Batch size to use for training")
args = parser.parse_args()

args.checkpoints_folder = "out"
args.notifier_prefix = "test"
args.remove_artifacts = False
args.segmentation = True
args.epochs = 3
args.batch_size = 32

# ------------ Some setup ------------ #
os.makedirs(args.checkpoints_folder, exist_ok=True)
os.makedirs("preloaded_data", exist_ok=True)

notifier = TelegramNotifier(prefix=args.notifier_prefix)
notifier.send_message("Started")


# ------------ Load images and metadata ------------ #
metadata = pd.read_csv(csv_file)

# benignMetadata = metadata[metadata[categoryName] == 'benign']
# malignantMetadata = metadata[metadata[categoryName] == 'malignant']
# augment_imagesList, augment_imageMetadata = duplicate_images(malignantMetadata, imageColName, images_folder, numOfDuplicates)

# benignMetadataSampleNumber = len(augment_imageMetadata)

# benignMetadataSample = benignMetadata.sample(benignMetadataSampleNumber, random_state=42)

# resultMetadata = pd.concat([benignMetadataSample, augment_imageMetadata], ignore_index=True)

resultMetadata = metadata

# # ------------ Data augmentation ------------ #
# notifier.send_message("Augmenting data...")
# count = 0
# for img in malignant:
#     for augmented_img in augment_image(img):
#         augmented_images[count] = augmented_img
#         count += 1


image_size = 224 

resultMetadata['full_path'] = resultMetadata[imageColName].apply(lambda x: os.path.join(images_folder, x + ".jpg") if not x.endswith(".jpg") else os.path.join(images_folder, x))
resultMetadata['processed_image'] = resultMetadata['full_path'].apply(lambda x: load_and_preprocess_image(x))


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

from sklearn.model_selection import StratifiedShuffleSplit

# Use StratifiedShuffleSplit to split the data while preserving Fitzpatrick distribution
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=6)
train_indices, test_indices = next(sss.split(X, Fitzpatrick))

# Split the data into training and testing sets based on the selected indices
X_train, X_test, y_train, y_test, other_attributes_train, other_attributes_test = (
    X[train_indices],
    X[test_indices],
    y[train_indices],
    y[test_indices],
    other_attributes[train_indices],
    other_attributes[test_indices],
)

# Now, let's add X_val and y_val to the existing train-test split
# Perform an additional train-validation split on X_train and y_train
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=6)
train_indices, val_indices = next(sss.split(X_train, Fitzpatrick[train_indices]))

# Create X_val and y_val based on the selected validation indices
X_val, y_val, other_attributes_val = (
    X_train[val_indices],
    y_train[val_indices],
    other_attributes_train[val_indices],
)

# Update X_train, y_train, and other_attributes_train to exclude the validation data
X_train, y_train, other_attributes_train = (
    X_train[train_indices],
    y_train[train_indices],
    other_attributes_train[train_indices],
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
    #model.add(keras.layers.Dropout(0.1))
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

print("Saving model...")
model.save(f"{args.checkpoints_folder}/final_model.h5")

notifier.send_message("Training finished!")



 
print ("===================================================================")
print(class_mapping)

# Evaluation & post-training analysis
# total accuracy of all test data
evlResult = model.evaluate(X_test, y_test)
loss, accuracy = evlResult[0], evlResult[1]
#loss, accuracy = model.evaluate(test_gen)
print("Accuracy for test data : ",accuracy)

test_predictions = model.predict(X_test)
#test_predictions = model.predict(test_gen)


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
#print(pred_name)
# Group predictions by Fitzpatrick scale and compute accuracy


#df_results['predicted'] = test_preds

df_results['correct'] = df_results[categoryName] == df_results['predicted']

 

for fitz_val in range(1, 7):
   subset = df_results[df_results['Fitzpatrick'] == fitz_val]
   print ("-------------------------------------------------------------------")
   print (subset[categoryName])
   print (subset['predicted'])
   accuracy = accuracy_score(subset[categoryName], subset['predicted'])
   #accuracy = accuracy_score(subset['correct'], subset['predicted'])
   print(f"Fitzpatrick {fitz_val} accuracy: {accuracy:.2f}")

print ("-------------------------------------------------------------------")

# Plotting loss and accuracy history
# plt.figure()
# plt.plot(history.history['loss'], label="Training loss")
# plt.plot(history.history['val_loss'], label="Validation loss")
# plt.title("Training and validation loss during training")
# plt.xlabel("No. epoch")
# plt.ylabel("Loss")
# plt.legend()
# plt.savefig(args.checkpoints_folder + "/loss_history.png")

# plt.figure()
# plt.plot(history.history['binary_accuracy'], label="Training accuracy")
# plt.plot(history.history['val_binary_accuracy'], label="Validation accuracy")
# plt.title("Training and validation accuracy during training")
# plt.xlabel("No. epoch")
# plt.ylabel("Binary accuracy")
# plt.legend()
# plt.savefig(args.checkpoints_folder + "/accuracy_history.png")


# ------------ Evaluation ------------ #
# metrics = model.evaluate(X_test, y_test, return_dict=True)
# notifier.send_message(f"Model evaluation on test data:\n{metrics}")

# with open(args.checkpoints_folder + "/eval_metrics", 'w') as f:
#     f.write(str(metrics))
