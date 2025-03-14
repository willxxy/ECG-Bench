import numpy as np
import os
import random
from PIL import Image
import cv2
from imgaug import augmenters as iaa
import cv2
import glob
import heapq
from skimage import util

def L2OverlapDiff(patch, block_size, overlap, res, y, x):
    error = 0
    if x > 0:
        left = patch[:, :overlap] - res[y:y+block_size, x:x+overlap]
        error += np.sum(left**2)

    if y > 0:
        up   = patch[:overlap, :] - res[y:y+overlap, x:x+block_size]
        error += np.sum(up**2)

    if x > 0 and y > 0:
        corner = patch[:overlap, :overlap] - res[y:y+overlap, x:x+overlap]
        error -= np.sum(corner**2)

    return error

def randomBestPatch(texture, block_size, overlap, res, y, x):
    h, w, _ = texture.shape
    errors = np.zeros((h - block_size, w - block_size))

    #Traverse through each block an calculate the overlap difference
    for i in range(h - block_size):
        for j in range(w - block_size):
            patch = texture[i:i+block_size, j:j+block_size]
            e = L2OverlapDiff(patch, block_size, overlap, res, y, x)
            errors[i, j] = e

    #Unravel to return block with least error
    i, j = np.unravel_index(np.argmin(errors), errors.shape)
    return texture[i:i+block_size, j:j+block_size]


def minCutPath(errors):
    # dijkstra's algorithm vertical
    pq = [(error, [i]) for i, error in enumerate(errors[0])]
    heapq.heapify(pq)

    h, w = errors.shape
    seen = set()

    while pq:
        error, path = heapq.heappop(pq)
        curDepth = len(path)
        curIndex = path[-1]

        if curDepth == h:
            return path

        for delta in -1, 0, 1:
            nextIndex = curIndex + delta

            if 0 <= nextIndex < w:
                if (curDepth, nextIndex) not in seen:
                    cumError = error + errors[curDepth, nextIndex]
                    heapq.heappush(pq, (cumError, path + [nextIndex]))
                    seen.add((curDepth, nextIndex))

def minCutPatch(patch, block_size, overlap, res, y, x):
    patch = patch.copy()
    dy, dx, _ = patch.shape
    minCut = np.zeros_like(patch, dtype=bool)

    if x > 0:
        left = patch[:, :overlap] - res[y:y+dy, x:x+overlap]
        leftL2 = np.sum(left**2, axis=2)
        for i, j in enumerate(minCutPath(leftL2)):
            minCut[i, :j] = True

    if y > 0:
        up = patch[:overlap, :] - res[y:y+overlap, x:x+dx]
        upL2 = np.sum(up**2, axis=2)
        for j, i in enumerate(minCutPath(upL2.T)):
            minCut[:i, j] = True

    np.copyto(patch, res[y:y+dy, x:x+dx], where=minCut)

    return patch

def quilt(texture_input, block_size, num_block, mode, sequence=False):
    """
    Generate a quilted texture from an input image.
    
    Parameters:
    -----------
    texture_input : str or numpy.ndarray
        Either a file path to an image or a numpy array containing the image
    block_size : int
        Size of blocks to use for quilting
    num_block : tuple
        (num_blockHigh, num_blockWide) - number of blocks in height and width
    mode : str
        Mode for quilting ('Cut' for minimum cut)
    sequence : bool, optional
        Whether to use sequential quilting
        
    Returns:
    --------
    numpy.ndarray
        Quilted image as a numpy array
    """
    # Handle both file path and numpy array inputs
    if isinstance(texture_input, str):
        texture = Image.open(texture_input)
        texture = util.img_as_float(texture)
    else:
        # Assume it's already a numpy array
        texture = util.img_as_float(texture_input)
    
    overlap = block_size // 6
    num_blockHigh, num_blockWide = num_block

    h = (num_blockHigh * block_size) - (num_blockHigh - 1) * overlap
    w = (num_blockWide * block_size) - (num_blockWide - 1) * overlap

    res = np.zeros((h, w, texture.shape[2]))

    #Loop to find random best patch that has minimum cut
    for i in range(num_blockHigh):
        for j in range(num_blockWide):
            y = i * (block_size - overlap)
            x = j * (block_size - overlap)
            patch = randomBestPatch(texture, block_size, overlap, res, y, x)
            
            patch = minCutPatch(patch, block_size, overlap, res, y, x)
            res[y:y+block_size, x:x+block_size] = patch

    image = (res * 255).astype(np.uint8)
    return image

def getCoords(angle,n,hh,ww):
    #Case when number of creases is 0
    if(n==0):
        return [[]],[[]]
    #Gap needed between lines would be the gap when lines are split among heights and weight of the image
    gap = int((hh+ww)/(n+1))
    
    #Coords1 refer to x and y coordinates of first point. Coords2 refer to x and y coordinates of second point.
    coords1 = []
    coords2 = []

    #Nested if else to handle different cases of angle
    if(angle<90 and angle!=0):
        #Divide the height and width of the image among n segments
        yc = 0
        xc = 0
        flag = 0
        for i in range(0,n):
            #Once xc+gap exceeds width of the image the lines will begin from the y axis. We track this with a flag
            if((xc+gap)<ww):
                xc = xc+ gap
            else:
                if(flag==0):
                    yc = xc + gap - ww
                    xc = ww 
                    flag = 1
                else:
                    yc = yc+ gap
                    xc = ww
            coord = [int(xc),int(yc)]
            coords1.append(coord)
    #If angle is 90 we just divide segments across the x axis
    elif(angle==90):
        yc = 0
        xc = 0
        gap = ww/(n+1)
        for i in range(0,n):
            xc = xc+gap
            coord = [int(xc),int(yc)]
            coords1.append(coord)
    #If angle is 180 or 0 we just divide segments across the y axis
    elif(angle==180 or angle==0):
        gap = hh/(n+1)
        xc = 0
        yc = 0
        for i in range(0,n):
            yc = yc+gap
            coord = [int(xc),int(yc)]
            coords1.append(coord)
    #If angle is greater than 90 then first start drawing lines from the y axis and then compute the x coordinates
    else:
        xc = 0
        yc = hh
        flag = 0
        for i in range(0,n):
            if((xc+gap)<ww):
                xc = xc + gap
            else:
                if(flag==0):
                    yc = (yc-(xc + gap - hh))
                    xc = ww
                    flag = 1
                else:
                    yc = yc - gap
                    xc = ww
            coord = [int(xc),int(yc)]
            coords1.append(coord)
    #Once the x and y coordinates of the starting point of the line are computed, compute the end point from slope
    for i in range(len(coords1)):
        x = coords1[i][0]
        y = coords1[i][1]
        m= np.tan((180-angle)*np.pi/180)
        c = int(y - m*x)
        #Computing end point from slope intercept form
        if(angle>90):
            if(c<0):
                yc = 0
                xc = int(-c/m)
            else:
                xc = ww
                yc = int(m*xc + c)
        #If angle is 90 we cannot use slope intercept form. We directly compute the end point coordinates
        elif (angle==90):
            yc = hh
            xc = (i+1)*(ww/(n+1))
        #If angle is 180 or 0, we again cannot compute end points using slope intercept form. We compute coordinates directly by division
        elif (angle==180 or angle==0):
            yc = (i+1)*(hh/(n+1))
            xc = ww
        #If angle is greater than 90 use slope intercept form
        else:
            if(c>hh):
                yc = hh
                xc = (yc - c)/m
            else:
                xc = 0
                yc = c
        coord = [int(xc),int(yc)]
        coords2.append(coord)
    return coords1,coords2

def get_creased_from_array(img_array, ifWrinkles=False, ifCreases=False, 
                          crease_angle=0, num_creases_vertically=3, 
                          num_creases_horizontally=2, bbox=False):
    """
    Apply creases and wrinkles to an image array without using temporary files.
    
    Parameters:
    -----------
    img_array : numpy.ndarray
        Input image as a numpy array
    ifWrinkles : bool
        Whether to add wrinkles
    ifCreases : bool
        Whether to add creases
    crease_angle : int
        Angle for creases
    num_creases_vertically : int
        Number of vertical creases
    num_creases_horizontally : int
        Number of horizontal creases
    bbox : bool
        Whether to use bounding boxes
        
    Returns:
    --------
    numpy.ndarray
        Processed image as a numpy array
    """
    # Convert to float in range 0 to 1
    img = img_array.astype("float32") / 255.0
    
    hh, ww = img.shape[:2]
    
    if ifWrinkles:
        
        # Import the quilt function here to avoid circular imports
        wrinklesImg = quilt(img, 250, (1,1), 'Cut')
        wrinklesImg = cv2.cvtColor(wrinklesImg, cv2.COLOR_BGR2GRAY)
        wrinklesImg = wrinklesImg.astype("float32") / 255.0
        
        # Resize wrinkles to same size as ecg input image
        wrinkles = cv2.resize(wrinklesImg, (ww, hh), fx=0, fy=0)
        # Shift image brightness so mean is (near) mid gray
        mean = np.mean(wrinkles)
        shift = mean - 0.4
        wrinkles = cv2.subtract(wrinkles, shift)
    
    if ifCreases:
        
        coords1, coords2 = getCoords(crease_angle, num_creases_horizontally, hh, ww)
        coords3, coords4 = getCoords(90+crease_angle, num_creases_vertically, hh, ww)
        
        creases = np.full((hh, ww), 1, dtype=np.float32)
        if num_creases_horizontally != 0:
            for i in range(len(coords1)):
                x1 = coords1[i][0]
                x2 = coords2[i][0]
                y1 = coords1[i][1]
                y2 = coords2[i][1]
                # Drawing lines
                if (x1-10) < 0:
                    cv2.line(creases, (x1, y1), (x2, y2), 1.25, 5)
                    cv2.line(creases, (x1, y1-5), (x2, y2-5), 1.15, 5)
                    cv2.line(creases, (x1, y1+5), (x2, y2+5), 1.15, 5)
                    cv2.line(creases, (x1, y1+10), (x2, y2+10), 1.05, 5)
                    cv2.line(creases, (x1, y1-10), (x2, y2-10), 1.05, 5)
                else:
                    cv2.line(creases, (x1, y1), (x2, y2), 1.25, 5)
                    cv2.line(creases, (x1-5, y1), (x2-5, y2), 1.15, 5)
                    cv2.line(creases, (x1+5, y1), (x2+5, y2), 1.15, 5)
                    cv2.line(creases, (x1-10, y1), (x2-10, y2), 1.05, 5)
                    cv2.line(creases, (x1+10, y1), (x2+10, y2), 1.05, 5)
        
        if num_creases_vertically != 0:
            for i in range(len(coords3)):
                x1 = coords3[i][0]
                x2 = coords4[i][0]
                y1 = coords3[i][1]
                y2 = coords4[i][1]
                if (x1-10) < 0:
                    cv2.line(creases, (x1, y1), (x2, y2), 1.25, 5)
                    cv2.line(creases, (x1, y1-5), (x2, y2-5), 1.15, 5)
                    cv2.line(creases, (x1, y1+5), (x2, y2+5), 1.15, 5)
                    cv2.line(creases, (x1, y1+10), (x2, y2+10), 1.05, 5)
                    cv2.line(creases, (x1, y1-10), (x2, y2-10), 1.05, 5)
                else:
                    cv2.line(creases, (x1, y1), (x2, y2), 1.25, 5)
                    cv2.line(creases, (x1-5, y1), (x2-5, y2), 1.15, 5)
                    cv2.line(creases, (x1+5, y1), (x2+5, y2), 1.15, 5)
                    cv2.line(creases, (x1-10, y1), (x2-10, y2), 1.05, 5)
                    cv2.line(creases, (x1+10, y1), (x2+10, y2), 1.05, 5)
        
        # Blur folds and crease array
        folds_creases = cv2.GaussianBlur(creases, (3, 3), 0)
        folds_creases = cv2.cvtColor(folds_creases, cv2.COLOR_GRAY2BGR)
        # Apply folds and crease mask
        img = (img * folds_creases)
    
    # If wrinkles need to be added, add the wrinkles mask
    if ifWrinkles:
        transform = wrinkles
        # Threshold wrinkles and invert
        thresh = cv2.threshold(transform, 0.6, 1, cv2.THRESH_BINARY)[1]
        thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        thresh_inv = 1 - thresh
        transform = cv2.cvtColor(transform, cv2.COLOR_GRAY2BGR)

        low = 2.0 * img * transform
        high = 1 - 2.0 * (1 - img) * (1 - transform)
        img = low * thresh_inv + high * thresh
    
    # Convert back to uint8
    img = (255 * img).clip(0, 255).astype(np.uint8)
    
    return img

def get_augment_from_array(img_array, rotate=25, noise=25, crop=0.01, temperature=6500, 
                          bbox=False, store_text_bounding_box=False):
    """
    Apply augmentations to an image array without using temporary files.
    
    Parameters:
    -----------
    img_array : numpy.ndarray
        Input image as a numpy array
    rotate : int
        Maximum rotation angle in degrees
    noise : int
        Amount of noise to add
    crop : float
        Maximum crop percentage
    temperature : int
        Color temperature adjustment
    bbox : bool
        Whether to use bounding boxes
    store_text_bounding_box : bool
        Whether to store text bounding boxes
        
    Returns:
    --------
    numpy.ndarray
        Processed image as a numpy array
    """
    
    # Create a copy of the input image
    image = img_array.copy()
    
    # Make sure image has 3 channels
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Create an empty list for lead bounding boxes
    lead_bbs = []
    leadNames_bbs = []
    plotted_pixels = []
    
    # Create a sequential augmentation pipeline
    rot = random.randint(-rotate, rotate)
    crop_sample = random.uniform(0, crop)
    seq = iaa.Sequential([
        iaa.Affine(rotate=rot),
        iaa.AdditiveGaussianNoise(scale=(noise, noise)),
        iaa.Crop(percent=crop_sample),
        iaa.ChangeColorTemperature(temperature)
    ])
    
    # Apply augmentations
    images_aug = seq(images=[image])
    
    return images_aug[0]


def augment_ecg_image(image_input, output_path=None, 
                     wrinkles=None, augment=None,
                     crease_angle=None, num_creases_vertically=None, 
                     num_creases_horizontally=None, noise=None, 
                     rotate=None, crop=None, temperature=None, 
                     fully_random=True, seed=None):
    """
    Apply augmentations to an ECG image with random parameter selection.
    Works directly with numpy arrays without creating temporary files.
    
    Parameters:
    -----------
    image_input : str or numpy.ndarray
        Path to the input ECG image or numpy array of the image
    output_path : str or None
        Path to save the augmented image. If None, returns the image as numpy array.
    wrinkles : bool or None
        Add wrinkles and creases to the image. If None, randomly decided.
    augment : bool or None
        Apply image augmentations (noise, rotation, etc.). If None, randomly decided.
    crease_angle : int or None
        Angle for creases (0-90). If None, randomly chosen.
    num_creases_vertically : int or None
        Number of vertical creases. If None, randomly chosen.
    num_creases_horizontally : int or None
        Number of horizontal creases. If None, randomly chosen.
    noise : int or None
        Amount of noise to add (0-100). If None, randomly chosen.
    rotate : int or None
        Maximum rotation angle in degrees. If None, randomly chosen.
    crop : float or None
        Maximum crop percentage (0-1). If None, randomly chosen.
    temperature : int or None
        Color temperature adjustment. If None, randomly chosen.
    fully_random : bool
        If True, randomly decide which augmentations to apply regardless of wrinkles/augment params
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    str or numpy.ndarray
        Path to the saved image if output_path is provided, or numpy array of the image if output_path is None
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Handle input as either path or numpy array
    if isinstance(image_input, str):
        img = np.array(Image.open(image_input))
    elif isinstance(image_input, np.ndarray):
        img = image_input.copy()
    else:
        raise ValueError("image_input must be either a file path or a numpy array")
    
    # Randomly decide which augmentations to apply if fully_random is True
    if fully_random:
        wrinkles = random.choice([True, False])
        augment = random.choice([True, False])
    else:
        # Default to True if not specified
        wrinkles = True if wrinkles is None else wrinkles
        augment = True if augment is None else augment
    
    # Apply wrinkles and creases if requested
    if wrinkles:
        # Randomly choose crease parameters if not specified
        if crease_angle is None:
            crease_angle = random.randint(0, 90)
        
        if num_creases_vertically is None:
            num_creases_vertically = random.randint(1, 10)
            
        if num_creases_horizontally is None:
            num_creases_horizontally = random.randint(1, 10)
        
        # Apply creases and wrinkles directly to the numpy array
        img = get_creased_from_array(
            img,
            ifWrinkles=True,
            ifCreases=True,
            crease_angle=crease_angle,
            num_creases_vertically=num_creases_vertically,
            num_creases_horizontally=num_creases_horizontally,
            bbox=False
        )
    
    # Apply image augmentations if requested
    if augment:
        # Randomly choose augmentation parameters if not specified
        if noise is None:
            noise = random.randint(10, 50)
            
        if rotate is None:
            rotate = random.randint(0, 25)
            
        if crop is None:
            crop = random.uniform(0, 0.05)
            
        if temperature is None:
            # Choose between blue-ish or yellow-ish temperature
            if random.choice([True, False]):  # blue_temp
                temperature = random.randint(2000, 4000)
            else:
                temperature = random.randint(10000, 20000)
        
        # Apply augmentations directly to the numpy array
        img = get_augment_from_array(
            img,
            rotate=rotate,
            noise=noise,
            crop=crop,
            temperature=temperature,
            bbox=False
        )
    
    # For debugging/information, you can print the parameters used
    if wrinkles or augment:
        params = {
            "wrinkles": wrinkles,
            "augment": augment
        }
        if wrinkles:
            params.update({
                "crease_angle": crease_angle,
                "num_creases_vertically": num_creases_vertically,
                "num_creases_horizontally": num_creases_horizontally
            })
        if augment:
            params.update({
                "noise": noise,
                "rotate": rotate,
                "crop": crop,
                "temperature": temperature
            })
        print(f"Augmentation parameters used: {params}")
    
    # Save or return the final image
    if output_path:
        # If the output path is a directory, create a filename
        if os.path.isdir(output_path):
            output_path = os.path.join(output_path, f"ecg_augmented_{random.randint(1000, 9999)}.png")
        
        # Save the final image to the output path
        Image.fromarray(img).save(output_path)
        return output_path
    else:
        # Return the image as a numpy array
        return img

# Original code for loading and processing ECG data
path_to_npy = glob.glob('./data/mimic/preprocessed_1250_250/*.npy')[0]

test_file = np.load(path_to_npy, allow_pickle = True).item()
print(test_file.keys())
ecg = test_file['ecg']
print(ecg.shape)

lead_names = ['I', 'II', 'III', 'aVL', 'aVR', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

from utils.viz_utils import VizUtil
VizUtil.plot_2d_ecg(ecg, '', './pngs/test_img', 250)

image = VizUtil.get_plot_as_image(ecg, 250)
print(image.shape)

# Save the original image
img = Image.fromarray(image)
original_image_path = './pngs/test_from_array.png'
img.save(original_image_path)

# Example 1: Save augmented image to disk
augmented_image_path = augment_ecg_image(
    original_image_path, 
    output_path='./pngs',
    fully_random=True,
    seed=42
)
print(f"Augmented image saved to: {augmented_image_path}")

# Example 2: Get augmented image as numpy array
augmented_array = augment_ecg_image(
    original_image_path,  # Using file path
    output_path=None,     # No output path means return numpy array
    fully_random=True,
    seed=43
)
print(f"Augmented image array shape: {augmented_array.shape}")

# Example 3: Directly use numpy array as input
augmented_array2 = augment_ecg_image(
    image,               # Using numpy array directly
    output_path=None,    # No output path means return numpy array
    fully_random=True,
    seed=44
)
print(f"Augmented image array shape: {augmented_array2.shape}")
