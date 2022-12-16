import cv2
import numpy as np

def RecoverCLAHE(sceneRadiance):
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(4, 4))
    for i in range(3):

        # sceneRadiance[:, :, i] =  cv2.equalizeHist(sceneRadiance[:, :, i])
        sceneRadiance[:, :, i] = clahe.apply((sceneRadiance[:, :, i]))


    return sceneRadiance

def RecoverHE(sceneRadiance):
    # clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(2, 2))
    for i in range(3):
        sceneRadiance[:, :, i] =  cv2.equalizeHist(sceneRadiance[:, :, i])
        # sceneRadiance[:, :, i] = clahe.apply((sceneRadiance[:, :, i]))
    return sceneRadiance

def skimageHE(sceneRadiance):
    # https://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_equalize.html
    from skimage import exposure
    # Equalization
    # sceneRadiance = exposure.adjust_sigmoid(sceneRadiance)
    p2, p98 = np.percentile(sceneRadiance, (2, 99))
    sceneRadiance = exposure.rescale_intensity(sceneRadiance, in_range=(p2, p98))
    # for i in range(3):
    #     sceneRadiance[:, :, i] =  exposure.rescale_intensity(sceneRadiance[:, :, i], in_range=(p2, p98))
    return sceneRadiance

def skimageintens(sceneRadiance):
    # https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.rescale_intensity
    from skimage import exposure
    for i in range(3):
        sceneRadiance[:, :, i] =  exposure.rescale_intensity(sceneRadiance[:, :, i])
    return sceneRadiance



def RecoverGC(sceneRadiance):
    sceneRadiance = sceneRadiance/255.0
    # clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(2, 2))
    for i in range(3):
        sceneRadiance[:, :, i] =  np.power(sceneRadiance[:, :, i] / float(np.max(sceneRadiance[:, :, i])), 1.5)
    sceneRadiance = np.clip(sceneRadiance*255, 0, 255)
    sceneRadiance = np.uint8(sceneRadiance)
    return sceneRadiance

def RecoverSUTR(sceneRadiance):
    sceneRadiance = sceneRadiance/255.0
    # clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(2, 2))
    for i in range(3):
        sceneRadiance[:, :, i] =  (np.subtract(sceneRadiance[:, :, i] , sceneRadiance[:, :, i] / float(np.sum(sceneRadiance[:, :, i])))* float(np.min(sceneRadiance[:, :, i]))) + sceneRadiance[:, :, i]
    sceneRadiance = np.clip(sceneRadiance*255, 0, 255)
    sceneRadiance = np.uint8(sceneRadiance)
    return sceneRadiance

def RecoverCTR(sceneRadiance):
    sceneRadiance = sceneRadiance/255.0
    # clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(2, 2))
    for i in range(3):
        sceneRadiance[:, :, i] =  (np.subtract(sceneRadiance[:, :, i] , sceneRadiance[:, :, i] / 0.5)* float(np.min(sceneRadiance[:, :, i]))) + sceneRadiance[:, :, i]
    sceneRadiance = np.clip(sceneRadiance*255, 0, 255)
    sceneRadiance = np.uint8(sceneRadiance)
    return sceneRadiance


def pil_cotrst(img):
    from PIL import Image, ImageEnhance
    image_pil = Image.fromarray(img)
    enhancer = ImageEnhance.Contrast(image_pil)
    # enhancer = ImageEnhance.Color(image_pil)
    # enhancer = ImageEnhance.Brightness(image_pil)
    # enhancer = ImageEnhance.Sharpness(image_pil)
    factor = 4  # increase contrast
    im_output = enhancer.enhance(factor)
    im_output = np.asarray(im_output).astype(np.uint16)
    return im_output

def cv2_cotrst(img):
    import cv2
    alpha = 1  # Contrast control (1.0-3.0)
    beta = 1  # Brightness control (0-100)
    im_output = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return im_output



def singleScaleRetinex(img, variance):
    retinex = np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), variance))
    return retinex


def multiScaleRetinex(img, variance_list):
    retinex = np.zeros_like(img)
    for variance in variance_list:
        retinex += singleScaleRetinex(img, variance)
    retinex = retinex / len(variance_list)
    return retinex


def MSR(img, variance_list):
    img = np.float64(img) + 1.0
    img_retinex = multiScaleRetinex(img, variance_list)

    for i in range(img_retinex.shape[2]):
        unique, count = np.unique(np.int32(img_retinex[:, :, i] * 100), return_counts=True)
        for u, c in zip(unique, count):
            if u == 0:
                zero_count = c
                break
        low_val = unique[0] / 100.0
        high_val = unique[-1] / 100.0
        for u, c in zip(unique, count):
            if u < 0 and c < zero_count * 0.1:
                low_val = u / 100.0
            if u > 0 and c < zero_count * 0.1:
                high_val = u / 100.0
                break
        img_retinex[:, :, i] = np.maximum(np.minimum(img_retinex[:, :, i], high_val), low_val)

        img_retinex[:, :, i] = (img_retinex[:, :, i] - np.min(img_retinex[:, :, i])) / \
                               (np.max(img_retinex[:, :, i]) - np.min(img_retinex[:, :, i])) \
                               * 255
    img_retinex = np.uint8(img_retinex)
    return img_retinex


def SSR(img, variance):
    img = np.float64(img) + 1.0
    img_retinex = singleScaleRetinex(img, variance)
    for i in range(img_retinex.shape[2]):
        unique, count = np.unique(np.int32(img_retinex[:, :, i] * 100), return_counts=True)
        for u, c in zip(unique, count):
            if u == 0:
                zero_count = c
                break
        low_val = unique[0] / 100.0
        high_val = unique[-1] / 100.0
        for u, c in zip(unique, count):
            if u < 0 and c < zero_count * 0.1:
                low_val = u / 100.0
            if u > 0 and c < zero_count * 0.1:
                high_val = u / 100.0
                break
        img_retinex[:, :, i] = np.maximum(np.minimum(img_retinex[:, :, i], high_val), low_val)

        img_retinex[:, :, i] = (img_retinex[:, :, i] - np.min(img_retinex[:, :, i])) / \
                               (np.max(img_retinex[:, :, i]) - np.min(img_retinex[:, :, i])) \
                               * 255
    img_retinex = np.uint8(img_retinex)
    return img_retinex
