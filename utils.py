import os
import cv2
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from scipy.ndimage import binary_fill_holes
from skimage.morphology import skeletonize, remove_small_objects, remove_small_holes, thin
from skimage.util import invert

img_threshold = 80
small_object_size = 100

def get_all_frame_paths(path, extension=".jpg"):
    print("There is this many files: ", len([f for f in os.listdir(path) if f.endswith(extension)]))
    return sorted([f for f in os.listdir(path) if f.endswith(extension)])#, key=lambda x: int(x.split('.')[0].split('_')[-1]), reverse=True)


def get_solidity(worm, picture):
    '''
    function to calculate the solidity of the object with equivalent diameter the most similar to the first frame's equivalent diameter of the animal
    :param worm: a Worm object
    :param picture: the specific worm picture name
    :return: solidity of the worm
    '''
    thresh, img = get_thresholded_img(worm.path + picture, worm.invert_color)
    if worm.invert_color:
        thresh = invert(thresh)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    assert len(contours) >= 1, "Not enough objects, check grayscale threshold" #usually 0 is entire image
    contours = sorted(contours, key=lambda x: cv2.contourArea(x))
    cnt = contours[-1]

    closest_diameter_index = -1

    if worm.equivalent_diameter == -1:  # means its not set yet
        worm.equivalent_diameter = eq_dia(cnt) # set it to the first area

    else: #need to check that the current one is the most similar area
        closest_diameter = eq_dia(cnt)
        for i, cnt_ in enumerate(contours[:-1]):
            diameter = eq_dia(cnt_)
            if abs(worm.equivalent_diameter-diameter)<abs(worm.equivalent_diameter-closest_diameter):
                closest_diameter_index = i
                closest_diameter = diameter
    cnt = contours[closest_diameter_index]
    area = cv2.contourArea(cnt)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    assert hull_area>0, "Hull area 0! oh oh!"
    solidity = float(area) / hull_area
    return solidity, thresh

def get_thresholded_img(path, is_inverted):
    if type(path)==str:
        img = cv2.imread(path) # skip loading for hdf5
    else:
        img = path
    #img = cv2.fastNlMeansDenoising(img, None)
    #print("converting and splitting")
    if len(img.shape) == 2:  # Grayscale image (single channel)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    plt.imshow(img)
    plt.show()
    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl, a, b))

    # Converting image from LAB Color model to BGR color spcae
    img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    # convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #print("REmoving bg")
    #bg = restoration.rolling_ball(gray)
    #gray = gray - bg
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (17, 17), 0) #(17,17), 0 for omegas

    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(13, 13)) #2.0, (13,13) for omegas
    enhanced = clahe.apply(blurred)
    #plt.imshow(enhanced)
    #plt.show()
    ret, thresh = cv2.threshold(enhanced, img_threshold, 255, cv2.THRESH_BINARY)  # + cv2.THRESH_OTSU)
    #thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 119, 7)
    plt.imshow(thresh)
    plt.show()
    return thresh, img

def eq_dia(cnt):
    area = cv2.contourArea(cnt)
    equi_diameter = np.sqrt(4 * area / np.pi)
    return equi_diameter

def get_first_frame_from_hdf5(hdf5_file):
    with h5py.File(hdf5_file, "r") as f:
        for name in f.keys():
            print(name)
        first_frame = f["/full_data"][0]  # Load only the first frame
    show = False
    if show:
        cv2.imshow("First Frame", first_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return first_frame




def calculate_worm_size(image_path, worm):
    """
    Calculate worm size from the first frame.
    Inputs:
        image_path: Path to the first frame (O*.jpg).
        worm: Worm object.
    Output:
        Worm length in mm.
    """


    thresh, img = get_thresholded_img(path=image_path, is_inverted=worm.invert_color)
    plt.imshow(thresh)
    plt.title("thresholded img")
    plt.show()
    # Skeletonize the binary image
    #thresh = remove_small_objects(thresh, min_size=150, connectivity=2)
    thinned = thin(invert(thresh))
    thinned = remove_small_objects(thinned, min_size=small_object_size, connectivity=2)
    skeleton = skeletonize(thinned)  # Convert binary to boolean for skimage
    skeleton = binary_fill_holes(skeleton)
    skeleton = skeletonize(skeleton)

    # Count the number of pixels in the skeleton
    plt.imshow(skeleton)
    plt.title("skeleton img")
    plt.show()
    pixel_count = np.sum(skeleton)
    print("there are this many pixels in length: ", pixel_count)
    # Convert to mm
    worm_length_mm = pixel_count * worm.pixel_to_mm_ratio

    return worm_length_mm


def calculate_reversals(data, animal_size_um, angle_threshold, scale=1.0):
    """
    Adaptation of Hardaker's method to detect reversal events.
    A single worm's centroid trajectory is re-sampled with a distance interval
    equivalent to the worm's length (animal_size_um) and reversals are calculated from turning angles.

    Inputs:
        data: DataFrame with 'X position' and 'Y position' columns (stepper units).
        animal_size_um: Worm size in micrometers (um).
        angle_threshold: Turning angle threshold (in degrees) to define reversals.
        scale: Scaling factor to convert stepper units to micrometers (default=1).

    Outputs:
        DataFrame with an additional 'reversals' column.
    """
    # Rescale positions to micrometers
    data['X_um'] = data['X position'] * scale
    data['Y_um'] = data['Y position'] * scale
    data['time_diff'] = data['Timestamp'].diff().fillna(1)  # Assume a default time difference for the first frame

    # Calculate velocities (micrometers per second)
    data['velocity'] = np.sqrt(
        (data['X_um'].diff() ** 2) + (data['Y_um'].diff() ** 2)
    ) / data['time_diff']
    data = data.fillna(0)
    print("upon inspection, the mean velocity is: ", data['velocity'].quantile(0.25))

    # Calculate cumulative distance traveled by the centroid
    data['distance'] = data['velocity'].cumsum()

    # Find maximum number of worm lengths traveled
    maxlen = data['distance'].max() / animal_size_um

    # Create resampling levels at intervals of animal_size_um
    levels = np.arange(animal_size_um, maxlen * animal_size_um, animal_size_um)

    # Find indices closest to resampling levels
    indices = []
    for level in levels:
        idx = (data['distance'] - level).abs().idxmin()
        indices.append(idx)

    # Resample trajectory
    resampled_data = data.loc[indices, ['X_um', 'Y_um']].diff()
    resampled_data[['X_prev', 'Y_prev']] = resampled_data.shift(1).fillna(0)

    # Calculate turning angles using dot product
    def calculate_angle(row):
        v1 = np.array([row['X_um'], row['Y_um']])
        v2 = np.array([row['X_prev'], row['Y_prev']])
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 0
        cos_theta = np.dot(v1, v2) / (norm_v1 * norm_v2)
        return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

    resampled_data['angle'] = resampled_data.apply(calculate_angle, axis=1)

    # Mark reversals based on angle threshold
    reversal_indices = resampled_data.index[resampled_data['angle'] >= angle_threshold]

    data['reversals'] = 0
    data.loc[reversal_indices, 'reversals'] = 1
    return data

def visualize_crawl_clusters(concordances, curvatures, labels, centers, n_clusters):
    plt.close()
    fig, axs = plt.subplots(1, n_clusters-1, figsize=(8*(n_clusters-1), 8))
    for i in range(n_clusters-1):
        axs[i].scatter(concordances, np.log(curvatures), c=labels[i])
        axs[i].scatter(centers[i][:, 0], centers[i][:, 1], c="black", s=100)
        axs[i].set_title(f'k={i}')
        axs[i].set_xlabel('Angular concordance')
        axs[i].set_ylabel('Curvature (ln(k))')
    plt.show()


def visualize_single_crawl_cluster(concordances, curvatures, labels, centers):
    plt.close()
    plt.scatter(concordances, np.log(curvatures), c=labels)
    plt.scatter(centers[:, 0], centers[:, 1], c="black", s=100)

    plt.xlabel('Angular concordance')
    plt.ylabel('Curvature (ln(k))')
    plt.show()

SQRT2 = 1.41421356237

def LNquantile(velocity, p):
    """
    Compute quantile function for log-normal distribution
    """
    μ = np.mean(velocity)
    σ = np.std(velocity)**2
    print(f'found mu {μ} and sigma {σ}')
    nq = SQRT2 * special.erfinv(2.0*p - 1.0) # N(0,1) normal quantile
    print(f'nq {nq}')
    t = μ +  σ * nq # N(μ, σ) quantile
    return np.exp(t) # LN(μ, σ) quantile

