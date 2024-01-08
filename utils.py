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
    return img
    #return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


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
    oridinal_metadata = metadata.iloc[index].copy()  # Create a copy of the DataFrame row
    baseImageName = oridinal_metadata[imgColName]
    #image_path = os.path.join(folderpath, oridinal_metadata[imgColName]) + ".jpg"
    image_path = os.path.join(folderpath, oridinal_metadata[imgColName])
    if not (image_path.endswith('.jpg') or image_path.endswith('.png')):
        image_path += ".jpg"
    image = load_image(image_path)

    augmented_images = []
    new_metadata = []
    vertical_flip = cv2.flip(image, 0)
    horizontal_flip = cv2.flip(image, 1)
    randomNum = random.random()

    # newImage = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    # newImageName = baseImageName + str(randomNum) + 'ROTATE_90_CLOCKWISE.jpg'
    # augmented_images.append(newImage)
    # oridinal_metadata[imgColName] = newImageName
    # save_augmented_image(newImage, newImageName, folderpath)
    # new_metadata.append(oridinal_metadata)

    # newImage = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # newImageName = baseImageName + str(randomNum) + 'ROTATE_90_COUNTERCLOCKWISE.jpg'
    # augmented_images.append(newImage)
    # oridinal_metadata[imgColName] = newImageName
    # save_augmented_image(newImage, newImageName, folderpath)
    # new_metadata.append(oridinal_metadata)

    # newImage = cv2.rotate(image, cv2.ROTATE_180)
    # newImageName = baseImageName + str(randomNum) + 'ROTATE_180.jpg'
    # augmented_images.append(newImage)
    # oridinal_metadata[imgColName] = newImageName
    # save_augmented_image(newImage, newImageName, folderpath)
    # new_metadata.append(oridinal_metadata)

    # newImage = vertical_flip
    # newImageName = baseImageName + str(randomNum) + 'vertical_flip.jpg'
    # augmented_images.append(newImage)
    # oridinal_metadata[imgColName] = newImageName
    # save_augmented_image(newImage, newImageName, folderpath)
    # new_metadata.append(oridinal_metadata)

    # newImage = horizontal_flip
    # newImageName = baseImageName + str(randomNum) + 'horizontal_flip.jpg'
    # augmented_images.append(newImage)
    # oridinal_metadata[imgColName] = newImageName
    # save_augmented_image(newImage, newImageName, folderpath)
    # new_metadata.append(oridinal_metadata)

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

    # Assuming 'image' is read in BGR format using cv2.imread()
    # Convert the image from BGR to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Split the LAB image into 'L', 'A', and 'B' channels
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    # Apply CLAHE to 'L' channel only
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel_clahe = clahe.apply(l_channel)

    # Merge the CLAHE enhanced L channel back with A and B channels
    lab_image_clahe = cv2.merge((l_channel_clahe, a_channel, b_channel))

    # Convert the LAB image back to BGR color space
    newImage = cv2.cvtColor(lab_image_clahe, cv2.COLOR_LAB2BGR)

    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # newImage = clahe.apply(image)
    newImageName = baseImageName + str(randomNum) + '_createCLAHE.jpg'
    augmented_images.append(newImage)
    oridinal_metadata[imgColName] = newImageName
    save_augmented_image(newImage, newImageName, folderpath)
    new_metadata.append(oridinal_metadata)

    # color jitter
    newImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    newImage = np.array(newImage, dtype = np.float64)
    random_brightness_coefficient = random.uniform(0.5, 1.5)
    random_saturation_coefficient = random.uniform(0.5, 1.5)
    random_hue_coefficient = random.uniform(0.5, 1.5)
    newImage[:,:,0] = newImage[:,:,0]*random_brightness_coefficient
    newImage[:,:,1] = newImage[:,:,1]*random_saturation_coefficient
    newImage[:,:,2] = newImage[:,:,2]*random_hue_coefficient
    newImage[:,:,0][newImage[:,:,0]>255]  = 255
    newImage[:,:,1][newImage[:,:,1]>255]  = 255
    newImage[:,:,2][newImage[:,:,2]>255]  = 255
    newImage = np.array(newImage, dtype = np.uint8)
    newImage = cv2.cvtColor(newImage, cv2.COLOR_HSV2BGR)
    newImageName = baseImageName + str(randomNum) + '_color_jitter.jpg'
    augmented_images.append(newImage)
    oridinal_metadata[imgColName] = newImageName
    save_augmented_image(newImage, newImageName, folderpath)
    new_metadata.append(oridinal_metadata)

    return augmented_images, new_metadata

def augment_image_removeBG_WithMetadata(metadata, index, imgColName, folderpath):
    oridinal_metadata = metadata.iloc[index].copy()  # Create a copy of the DataFrame row
    baseImageName = oridinal_metadata[imgColName]
    #image_path = os.path.join(folderpath, oridinal_metadata[imgColName]) + ".jpg"
    image_path = os.path.join(folderpath, oridinal_metadata[imgColName])
    if not (image_path.endswith('.jpg') or image_path.endswith('.png')):
        image_path += ".jpg"
    image = load_image(image_path)

    augmented_images = []
    new_metadata = []

    vertical_flip = cv2.flip(image, 0)
    horizontal_flip = cv2.flip(image, 1)
    randomNum = random.random()

    # Convert the image to a color space like HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Define range for skin lesion colors
    # These values would need to be adjusted based on your specific images
    lower_bound = np.array([0, 50, 50])
    upper_bound = np.array([30, 255, 255])
    # Threshold the HSV image to get only skin lesions colors
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    # Bitwise-AND mask and original image
    newImage = cv2.bitwise_and(image, image, mask=mask)
    newImageName = baseImageName + str(randomNum) + '_removeBG.jpg'
    augmented_images.append(newImage)
    oridinal_metadata[imgColName] = newImageName
    save_augmented_image(newImage, newImageName, folderpath)
    new_metadata.append(oridinal_metadata)
    
    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # Convert the image to a color space like HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Define range for skin lesion colors
    # These values would need to be adjusted based on your specific images
    lower_bound = np.array([0, 50, 50])
    upper_bound = np.array([30, 255, 255])
    # Threshold the HSV image to get only skin lesions colors
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    # Bitwise-AND mask and original image
    newImage = cv2.bitwise_and(image, image, mask=mask)
    newImageName = baseImageName + str(randomNum) + '_removeBG_rotate90cw.jpg'
    augmented_images.append(newImage)
    oridinal_metadata[imgColName] = newImageName
    save_augmented_image(newImage, newImageName, folderpath)
    new_metadata.append(oridinal_metadata)

    image = vertical_flip
    # Convert the image to a color space like HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Define range for skin lesion colors
    # These values would need to be adjusted based on your specific images
    lower_bound = np.array([0, 50, 50])
    upper_bound = np.array([30, 255, 255])
    # Threshold the HSV image to get only skin lesions colors
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    # Bitwise-AND mask and original image
    newImage = cv2.bitwise_and(image, image, mask=mask)
    newImageName = baseImageName + str(randomNum) + '_removeBG_verticalFlip.jpg'
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
    # #augmented_images.append(cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE))
    # #augmented_images.append(cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE))
    # #augmented_images.append(cv2.rotate(image, cv2.ROTATE_180))
    # #augmented_images.append(vertical_flip)
    # augmented_images.append(horizontal_flip)
    # augmented_images.append(cv2.rotate(vertical_flip, cv2.ROTATE_90_CLOCKWISE))
    augmented_images.append(cv2.rotate(horizontal_flip, cv2.ROTATE_90_CLOCKWISE))
    clahe = cv2.createCLAHE(clipLimit=random.randint(1.5,2.5), tileGridSize=(8,8))
    cl1 = clahe.apply(image)
    augmented_images.append(cl1)
    # clahe2 = cv2.createCLAHE(clipLimit=random.randint(1.5,2.5), tileGridSize=(8,8))
    # cl2 = clahe.apply(image)
    # augmented_images.append(cl2)
    return augmented_images

#################### new image augmentation functions ####################
def augment_image_FlipRotate_WithMetadata(metadata, index, imgColName, folderpath):
    oridinal_metadata = metadata.iloc[index].copy()  # Create a copy of the DataFrame row
    baseImageName = oridinal_metadata[imgColName]
    image_path = os.path.join(folderpath, baseImageName)
    if not (image_path.endswith('.jpg') or image_path.endswith('.png')):
        image_path += ".jpg"
    image = load_image(image_path)

    randomNum = random.random()

    if randomNum < 0.33:
        # flip and rotate
        vertical_flip = cv2.flip(image, 0)
        horizontal_flip = cv2.flip(image, 1)
        newImage = cv2.rotate(vertical_flip, cv2.ROTATE_90_CLOCKWISE)
        newImageName = baseImageName + str(randomNum) + '_aug_vf_ROTATE_90_CLOCKWISE.jpg'
    elif randomNum < 0.66:
        # flip and rotate
        vertical_flip = cv2.flip(image, 1)
        horizontal_flip = cv2.flip(image, 0)
        newImage = cv2.rotate(vertical_flip, cv2.ROTATE_90_CLOCKWISE)
        newImageName = baseImageName + str(randomNum) + '_aug_hf_ROTATE_90_CLOCKWISE.jpg'
    else :
        #rotate random
        rotation_options = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
        rotation_choice = random.choice(rotation_options)
        newImage = cv2.rotate(image, rotation_choice)
        newImageName = baseImageName + str(randomNum) + '_aug_rotate_random.jpg'

    oridinal_metadata[imgColName] = newImageName
    save_augmented_image(newImage, newImageName, folderpath)
    oridinal_metadata['full_path'] = os.path.join(folderpath, newImageName)
    oridinal_metadata['processed_image'] = preprocess_image(newImage)    
    return oridinal_metadata

def augment_image_jitter_WithMetadata(metadata, index, imgColName, folderpath):
    oridinal_metadata = metadata.iloc[index].copy()  # Create a copy of the DataFrame row
    baseImageName = oridinal_metadata[imgColName]
    image_path = os.path.join(folderpath, baseImageName)
    if not (image_path.endswith('.jpg') or image_path.endswith('.png')):
        image_path += ".jpg"
    image = load_image(image_path)
    randomNum = random.random()
    # color jitter
    newImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    newImage = np.array(newImage, dtype = np.float64)
    random_brightness_coefficient = random.uniform(0.5, 1.5)
    random_saturation_coefficient = random.uniform(0.5, 1.5)
    random_hue_coefficient = random.uniform(0.5, 1.5)
    newImage[:,:,0] = newImage[:,:,0]*random_brightness_coefficient
    newImage[:,:,1] = newImage[:,:,1]*random_saturation_coefficient
    newImage[:,:,2] = newImage[:,:,2]*random_hue_coefficient
    newImage[:,:,0][newImage[:,:,0]>255]  = 255
    newImage[:,:,1][newImage[:,:,1]>255]  = 255
    newImage[:,:,2][newImage[:,:,2]>255]  = 255
    newImage = np.array(newImage, dtype = np.uint8)
    newImage = cv2.cvtColor(newImage, cv2.COLOR_HSV2BGR)
    newImageName = baseImageName + str(randomNum) + '_aug_color_jitter.jpg'
    oridinal_metadata[imgColName] = newImageName
    save_augmented_image(newImage, newImageName, folderpath)
    oridinal_metadata['full_path'] = os.path.join(folderpath, newImageName)
    oridinal_metadata['processed_image'] = preprocess_image(newImage)    
    return oridinal_metadata

def augment_image_IncreaseBrightness_WithMetadata(metadata, index, imgColName, folderpath):
    oridinal_metadata = metadata.iloc[index].copy()  # Create a copy of the DataFrame row
    baseImageName = oridinal_metadata[imgColName]
    image_path = os.path.join(folderpath, baseImageName)
    if not (image_path.endswith('.jpg') or image_path.endswith('.png')):
        image_path += ".jpg"
    image = load_image(image_path)
    randomNum = random.random()    
    #increase brightness
    alpha = random.uniform(1.0, 2.0)  # For example, a scale factor between 1.0 and 2.0
    beta = random.randint(0, 50)  # For example, an added value between 0 and 50
    newImage = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    newImageName = baseImageName + str(randomNum) + '_aug_IncreaseBrightness.jpg'
    oridinal_metadata[imgColName] = newImageName
    save_augmented_image(newImage, newImageName, folderpath)
    oridinal_metadata['full_path'] = os.path.join(folderpath, newImageName)
    oridinal_metadata['processed_image'] = preprocess_image(newImage)    
    return oridinal_metadata

def augment_image_GBlur_WithMetadata(metadata, index, imgColName, folderpath):
    oridinal_metadata = metadata.iloc[index].copy()  # Create a copy of the DataFrame row
    baseImageName = oridinal_metadata[imgColName]
    image_path = os.path.join(folderpath, baseImageName)
    if not (image_path.endswith('.jpg') or image_path.endswith('.png')):
        image_path += ".jpg"
    image = load_image(image_path)
    randomNum = random.random()
    # random blur
    kernel_width = random.choice(range(3, 11, 2))  # Choose an odd number between 3 and 10
    kernel_height = random.choice(range(3, 11, 2))  # Choose an odd number between 3 and 10
    sigma = random.uniform(0.1, 2.0)  # For example, a sigma between 0.1 and 2.0
    newImage = cv2.GaussianBlur(image, (kernel_width, kernel_height), sigma)
    newImageName = baseImageName + str(randomNum) + '_aug_GBlur.jpg'
    oridinal_metadata[imgColName] = newImageName
    save_augmented_image(newImage, newImageName, folderpath)
    oridinal_metadata['full_path'] = os.path.join(folderpath, newImageName)
    oridinal_metadata['processed_image'] = preprocess_image(newImage)    
    return oridinal_metadata

def augment_image_CLAHE_WithMetadata(metadata, index, imgColName, folderpath):
    oridinal_metadata = metadata.iloc[index].copy()  # Create a copy of the DataFrame row
    baseImageName = oridinal_metadata[imgColName]
    #image_path = os.path.join(folderpath, oridinal_metadata[imgColName]) + ".jpg"
    image_path = os.path.join(folderpath, baseImageName)
    if not (image_path.endswith('.jpg') or image_path.endswith('.png')):
        image_path += ".jpg"
    image = load_image(image_path)
    randomNum = random.random()
    # Assuming 'image' is read in BGR format using cv2.imread()
    # Convert the image from BGR to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # Split the LAB image into 'L', 'A', and 'B' channels
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    # Apply CLAHE to 'L' channel only
    clahe = cv2.createCLAHE(clipLimit=random.uniform(1.5, 2.5), tileGridSize=(8, 8))
    l_channel_clahe = clahe.apply(l_channel)
    # Merge the CLAHE enhanced L channel back with A and B channels
    lab_image_clahe = cv2.merge((l_channel_clahe, a_channel, b_channel))
    # Convert the LAB image back to BGR color space
    newImage = cv2.cvtColor(lab_image_clahe, cv2.COLOR_LAB2BGR)
    newImageName = baseImageName + str(randomNum) + '_aug_createCLAHE.jpg'
    oridinal_metadata[imgColName] = newImageName
    save_augmented_image(newImage, newImageName, folderpath)
    oridinal_metadata['full_path'] = os.path.join(folderpath, newImageName)
    oridinal_metadata['processed_image'] = preprocess_image(newImage)
    return oridinal_metadata

def augment_image_GBlurAndCLAHE_WithMetadata(metadata, index, imgColName, folderpath):
    oridinal_metadata = metadata.iloc[index].copy()  # Create a copy of the DataFrame row
    baseImageName = oridinal_metadata[imgColName]
    image_path = os.path.join(folderpath, baseImageName)
    if not (image_path.endswith('.jpg') or image_path.endswith('.png')):
        image_path += ".jpg"
    image = load_image(image_path)
    # random blur
    kernel_width = random.choice(range(3, 11, 2))  # Choose an odd number between 3 and 10
    kernel_height = random.choice(range(3, 11, 2))  # Choose an odd number between 3 and 10
    sigma = random.uniform(0.1, 2.0)  # For example, a sigma between 0.1 and 2.0
    newImage0 = cv2.GaussianBlur(image, (kernel_width, kernel_height), sigma)
    randomNum = random.random()
    # Assuming 'image' is read in BGR format using cv2.imread()
    # Convert the image from BGR to LAB color space
    lab_image = cv2.cvtColor(newImage0, cv2.COLOR_BGR2LAB)
    # Split the LAB image into 'L', 'A', and 'B' channels
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    # Apply CLAHE to 'L' channel only
    clahe = cv2.createCLAHE(clipLimit=random.uniform(1.5, 2.5), tileGridSize=(8, 8))
    l_channel_clahe = clahe.apply(l_channel)
    # Merge the CLAHE enhanced L channel back with A and B channels
    lab_image_clahe = cv2.merge((l_channel_clahe, a_channel, b_channel))
    # Convert the LAB image back to BGR color space
    newImage = cv2.cvtColor(lab_image_clahe, cv2.COLOR_LAB2BGR)
    newImageName = baseImageName + str(randomNum) + '_aug_Glur_CLAHE.jpg'
    oridinal_metadata[imgColName] = newImageName
    save_augmented_image(newImage, newImageName, folderpath)
    oridinal_metadata['full_path'] = os.path.join(folderpath, newImageName)
    oridinal_metadata['processed_image'] = preprocess_image(newImage)    
    return oridinal_metadata

def augment_image_GBlur_CLAHE_jitter_WithMetadata(metadata, index, imgColName, folderpath):
    oridinal_metadata = metadata.iloc[index].copy()  # Create a copy of the DataFrame row
    baseImageName = oridinal_metadata[imgColName]
    image_path = os.path.join(folderpath, baseImageName)
    if not (image_path.endswith('.jpg') or image_path.endswith('.png')):
        image_path += ".jpg"
    image = load_image(image_path)
    # random blur
    kernel_width = random.choice(range(3, 11, 2))  # Choose an odd number between 3 and 10
    kernel_height = random.choice(range(3, 11, 2))  # Choose an odd number between 3 and 10
    sigma = random.uniform(0.1, 2.0)  # For example, a sigma between 0.1 and 2.0
    newImage0 = cv2.GaussianBlur(image, (kernel_width, kernel_height), sigma)
    randomNum = random.random()
    # Assuming 'image' is read in BGR format using cv2.imread()
    # Convert the image from BGR to LAB color space
    lab_image = cv2.cvtColor(newImage0, cv2.COLOR_BGR2LAB)
    # Split the LAB image into 'L', 'A', and 'B' channels
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    # Apply CLAHE to 'L' channel only
    clahe = cv2.createCLAHE(clipLimit=random.uniform(1.5, 2.5), tileGridSize=(8, 8))
    l_channel_clahe = clahe.apply(l_channel)
    # Merge the CLAHE enhanced L channel back with A and B channels
    lab_image_clahe = cv2.merge((l_channel_clahe, a_channel, b_channel))
    # Convert the LAB image back to BGR color space
    image = cv2.cvtColor(lab_image_clahe, cv2.COLOR_LAB2BGR)

    randomNum = random.random()
    # color jitter
    newImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    newImage = np.array(newImage, dtype = np.float64)
    random_brightness_coefficient = random.uniform(0.5, 1.5)
    random_saturation_coefficient = random.uniform(0.5, 1.5)
    random_hue_coefficient = random.uniform(0.5, 1.5)
    newImage[:,:,0] = newImage[:,:,0]*random_brightness_coefficient
    newImage[:,:,1] = newImage[:,:,1]*random_saturation_coefficient
    newImage[:,:,2] = newImage[:,:,2]*random_hue_coefficient
    newImage[:,:,0][newImage[:,:,0]>255]  = 255
    newImage[:,:,1][newImage[:,:,1]>255]  = 255
    newImage[:,:,2][newImage[:,:,2]>255]  = 255
    newImage = np.array(newImage, dtype = np.uint8)
    newImage = cv2.cvtColor(newImage, cv2.COLOR_HSV2BGR)
    newImageName = baseImageName + str(randomNum) + '_aug_Glur_CLAHE_jitter.jpg'
    oridinal_metadata[imgColName] = newImageName
    save_augmented_image(newImage, newImageName, folderpath)
    oridinal_metadata['full_path'] = os.path.join(folderpath, newImageName)
    oridinal_metadata['processed_image'] = preprocess_image(newImage)    
    return oridinal_metadata

# Function to save augmented images
def save_augmented_image(augmented_image, imageName, output_folder):
        # output_path = os.path.join(output_folder, imageName)
        # cv2.imwrite(output_path, augmented_image)
    pass

def duplicate_imageWithMetadata(metadata, index, imgColName, folderpath):
    oridinal_metadata = metadata.iloc[index].copy()  # Create a copy of the DataFrame row
    image_path = os.path.join(folderpath, oridinal_metadata[imgColName])
    oridinal_metadata['full_path'] = image_path
    oridinal_metadata['processed_image'] =  load_and_preprocess_image(image_path)
    return oridinal_metadata

def duplicate_images(df, imgColName, image_folder, count_dup, type_aug):
    newMetadataToAdd = []
    for _ in range(count_dup):
        for index in range(len(df)):
            if type_aug.lower() == 'jitter':
                newmetadata = augment_image_jitter_WithMetadata(df, index, imgColName, image_folder)
            elif type_aug.lower() == 'clahe':
                newmetadata = augment_image_CLAHE_WithMetadata(df, index, imgColName, image_folder)
            elif type_aug.lower() == 'blur':
                newmetadata = augment_image_GBlur_WithMetadata(df, index, imgColName, image_folder)
            elif type_aug.lower() == 'rotate':
                newmetadata = augment_image_FlipRotate_WithMetadata(df, index, imgColName, image_folder)
            elif type_aug.lower() == 'brightness':
                newmetadata = augment_image_IncreaseBrightness_WithMetadata(df, index, imgColName, image_folder)
            elif type_aug.lower() == 'blur_clahe':
                newmetadata = augment_image_GBlurAndCLAHE_WithMetadata(df, index, imgColName, image_folder)
            elif type_aug.lower() == 'blur_clahe_jitter':
                newmetadata = augment_image_GBlur_CLAHE_jitter_WithMetadata(df, index, imgColName, image_folder)
            else:
                newmetadata = duplicate_imageWithMetadata(df, index, imgColName, image_folder)

            newMetadataToAdd.append(newmetadata)

    # Convert the list of dictionaries to a DataFrame
    new_metadata_df = pd.DataFrame(newMetadataToAdd)
    return new_metadata_df

from PIL import Image
def load_and_preprocess_image(file_path, target_size=(224, 224)):
    img = cv2.resize(load_image(file_path), target_size)
    # img = Image.open(file_path + ".jpg")
    # img = img.resize(target_size)
    #imgarray = np.array(img) / 255.0  # Normalize pixel values
    imgarray = np.array(img) 
    return imgarray

def preprocess_image(image, target_size=(224, 224)):
    img = cv2.resize(image, target_size)
    # img = Image.open(file_path + ".jpg")
    # img = img.resize(target_size)
    #imgarray = np.array(img) / 255.0  # Normalize pixel values
    imgarray = np.array(img) 
    return imgarray
