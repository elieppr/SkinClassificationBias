import numpy as np
import os
import cv2
import skimage.morphology
import random
import pandas as pd
import matplotlib.pyplot as plt

os.environ['PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT'] = '600.0'

def pictures_grid(args: list, layout: tuple, fontsize:int=12, show_plot:bool=False, return_figure:bool=False):
    """
    Draw a picture grid with layout = (layout[0], layout[1])
    """
    fig, axes = plt.subplots(layout[0], layout[1], figsize=(8, 8))
    ax = axes.flatten()
    [a.set_axis_off() for a in ax]
    for i, img in enumerate(args):
        if type(img) in (list, tuple):
            kwargs = img[2] if len(img) >= 3 else {}
            ax[i].imshow(img[0], **kwargs)
            ax[i].set_title(img[1], fontsize=fontsize)
        else:
            ax[i].imshow(img)
    fig.tight_layout()
    if show_plot:
        fig.show()
    if return_figure:
        return fig


# Images loading
def load_image(filename):
    img = cv2.imread(filename)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# Morphological closing
def apply_morpho_closing(image, disk_size=4):
    disk = skimage.morphology.disk(disk_size)
    r = skimage.morphology.closing(image[..., 0], disk)
    g = skimage.morphology.closing(image[..., 1], disk)
    b = skimage.morphology.closing(image[..., 2], disk)
    return np.stack((r, g, b), axis=-1)


# KMeans segmentation
def kmeans_mask(image, return_rgb=False):
    K = 2
    attempts = 1
    _, labels, centers = cv2.kmeans(np.float32(image.reshape((-1, 3))), K, None, None, attempts, cv2.KMEANS_RANDOM_CENTERS) # or cv2.KMEANS_PP_CENTERS
    centers = np.uint8(centers)
    lesion_cluster = np.argmin(np.mean(centers, axis=1))
    lesion_mask = labels.flatten() == lesion_cluster
    if return_rgb:
        rgb_mask = np.zeros(image.shape)
        rgb_mask[~lesion_mask.reshape(image.shape[:2])] = 255
        return rgb_mask
    return lesion_mask

def kmeans_segmentation(image, force_copy=True, mask=None):
    lesion_mask = mask if mask else kmeans_mask(image)
    segmented_img = image.reshape((-1, 3))
    if force_copy and segmented_img.base is image:
        segmented_img = segmented_img.copy()
    segmented_img[~lesion_mask] = 255
    return segmented_img.reshape(image.shape)

def augment_imageWithMetadata(metadata, index, imgColName, folderpath):
    oridinal_metadata = metadata.iloc[index]
    baseImageName = oridinal_metadata[imgColName]
    image_path = os.path.join(folderpath, oridinal_metadata[imgColName]) + ".jpg"
    image = load_image(image_path)

    augmented_images = []
    new_metadata = []
    vertical_flip = cv2.flip(image, 0)
    horizontal_flip = cv2.flip(image, 1)
    randomNum = random.random()

    newImage = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    newImageName = baseImageName + str(randomNum) + 'ROTATE_90_CLOCKWISE.jpg'
    augmented_images.append(newImage)
    oridinal_metadata[imgColName] = newImageName
    save_augmented_image(newImage, newImageName, folderpath)
    new_metadata.append(oridinal_metadata)

    newImage = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    newImageName = baseImageName + str(randomNum) + 'ROTATE_90_COUNTERCLOCKWISE.jpg'
    augmented_images.append(newImage)
    oridinal_metadata[imgColName] = newImageName
    save_augmented_image(newImage, newImageName, folderpath)
    new_metadata.append(oridinal_metadata)

    newImage = cv2.rotate(image, cv2.ROTATE_180)
    newImageName = baseImageName + str(randomNum) + 'ROTATE_180.jpg'
    augmented_images.append(newImage)
    oridinal_metadata[imgColName] = newImageName
    save_augmented_image(newImage, newImageName, folderpath)
    new_metadata.append(oridinal_metadata)

    newImage = vertical_flip
    newImageName = baseImageName + str(randomNum) + 'vertical_flip.jpg'
    augmented_images.append(newImage)
    oridinal_metadata[imgColName] = newImageName
    save_augmented_image(newImage, newImageName, folderpath)
    new_metadata.append(oridinal_metadata)

    newImage = horizontal_flip
    newImageName = baseImageName + str(randomNum) + 'horizontal_flip.jpg'
    augmented_images.append(newImage)
    oridinal_metadata[imgColName] = newImageName
    save_augmented_image(newImage, newImageName, folderpath)
    new_metadata.append(oridinal_metadata)

    newImage = cv2.rotate(vertical_flip, cv2.ROTATE_90_CLOCKWISE)
    newImageName = baseImageName + str(randomNum) + 'vf_ROTATE_90_CLOCKWISE.jpg'
    augmented_images.append(newImage)
    oridinal_metadata[imgColName] = newImageName
    save_augmented_image(newImage, newImageName, folderpath)
    new_metadata.append(oridinal_metadata)

    newImage = cv2.rotate(horizontal_flip, cv2.ROTATE_90_CLOCKWISE)
    newImageName = baseImageName + str(randomNum) + 'hf_ROTATE_90_CLOCKWISE.jpg'
    augmented_images.append(newImage)
    oridinal_metadata[imgColName] = newImageName
    save_augmented_image(newImage, newImageName, folderpath)
    new_metadata.append(oridinal_metadata)

    return augmented_images, new_metadata

# Data augmentation
def augment_image(image):
    augmented_images = []
    vertical_flip = cv2.flip(image, 0)
    horizontal_flip = cv2.flip(image, 1)
    augmented_images.append(cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE))
    augmented_images.append(cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE))
    augmented_images.append(cv2.rotate(image, cv2.ROTATE_180))
    augmented_images.append(vertical_flip)
    augmented_images.append(horizontal_flip)
    augmented_images.append(cv2.rotate(vertical_flip, cv2.ROTATE_90_CLOCKWISE))
    augmented_images.append(cv2.rotate(horizontal_flip, cv2.ROTATE_90_CLOCKWISE))
    return augmented_images

# Function to save augmented images
def save_augmented_image(augmented_image, imageName, output_folder):
        output_path = os.path.join(output_folder, imageName)
        cv2.imwrite(output_path, augmented_image)

# Data augmentation
# def augment_imageWithMetadata(metadata, index, imgColName, folderpath, count_dup):
#     oridinal_metadata = metadata.iloc[index]
#     baseImageName = oridinal_metadata[imgColName]
#     image_path = os.path.join(folderpath, oridinal_metadata['image_name']) + ".jpg"
#     image = cv2.imread(image_path, cv2.IMREAD_COLOR)
#     augmented_images = []
#     new_metadata = []

#     # Probability of applying each augmentation
#     p_rotate_90 = 0.5
#     p_rotate_180 = 0.5
#     p_vertical_flip = 0.5
#     p_horizontal_flip = 0.5

#     cnt = 0
#     while(cnt<count_dup):
#         randomNum = random.random()
#         if randomNum < p_rotate_90 and cnt<count_dup:
#             newImage = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
#             #newImageName = oridinal_metadata[imgColName].replace('.jpg','_' + randomNum + '_90.jpg')
#             newImageName = baseImageName + str(randomNum) + '_90.jpg'
#             augmented_images.append(newImage)
#             oridinal_metadata[imgColName] = newImageName
#             save_augmented_image(newImage, newImageName, folderpath)
#             new_metadata.append(oridinal_metadata)
#             cnt += 1
#         if randomNum < p_rotate_180:
#             newImage = cv2.rotate(image, cv2.ROTATE_180)
#             #newImageName = oridinal_metadata[imgColName].replace('.jpg','_' + randomNum + '_180.jpg')
#             newImageName = baseImageName + str(randomNum) + '_180.jpg'
#             augmented_images.append(newImage)
#             oridinal_metadata[imgColName] = newImageName
#             save_augmented_image(newImage, newImageName, folderpath)
#             new_metadata.append(oridinal_metadata)
#             cnt += 1
#         if randomNum < p_vertical_flip:
#             newImage = cv2.flip(image, 0)
#             #newImageName = oridinal_metadata[imgColName].replace('.jpg','_' + randomNum + '_vf.jpg')
#             newImageName = baseImageName + str(randomNum) + '_vf.jpg'
#             augmented_images.append(newImage)
#             oridinal_metadata[imgColName] = newImageName
#             save_augmented_image(newImage, newImageName, folderpath)
#             new_metadata.append(oridinal_metadata)
#             cnt += 1
#         if randomNum < p_horizontal_flip:
#             newImage = cv2.flip(image, 1)
#             #newImageName = oridinal_metadata[imgColName].replace('.jpg','_' + randomNum + '_hf.jpg')
#             newImageName = baseImageName + str(randomNum) + '_hf.jpg'
#             augmented_images.append(newImage)
#             oridinal_metadata[imgColName] = newImageName
#             save_augmented_image(newImage, newImageName, folderpath)
#             new_metadata.append(oridinal_metadata)
#             cnt += 1

#     return augmented_images, new_metadata


def duplicate_images(df, imgColName, image_folder, count_dup):
    augmented_images = []
    newMetadataToAdd = []
    for _ in range(count_dup):
        for index in range(len(df)):
            augimgs, newmetadata = augment_imageWithMetadata(df, index, imgColName, image_folder)
            augmented_images.extend(augimgs)
            newMetadataToAdd.extend(newmetadata)

    # Convert the list of dictionaries to a DataFrame
    new_metadata_df = pd.DataFrame(newMetadataToAdd)
    # Add the new metadata to the existing DataFrame
    newdf = pd.concat([df, new_metadata_df], ignore_index=True)
    return augmented_images, newdf
    #return augmented_images, newMetadataToAdd

from PIL import Image
def load_and_preprocess_image(file_path, target_size=(224, 224)):
    img = cv2.resize(load_image(file_path), target_size)
    # img = Image.open(file_path + ".jpg")
    # img = img.resize(target_size)
    #imgarray = np.array(img) / 255.0  # Normalize pixel values
    imgarray = np.array(img) 
    return imgarray