import rawpy
import imageio
import json
import numpy as np
import os
from pathlib import Path
from p_tqdm import p_umap
import skimage as ski
import errno
import random


def unpack(raw):
    """
    Unpack raw data to white-balanced RGB image and relevant color metadata to
    transform camera RGB to sRGB

    Parameters
    ----------
    raw : ndarray
        rawpy object representation of raw image file
    """

    black = raw.black_level_per_channel.copy()
    wb = np.diagflat(raw.camera_whitebalance.copy()[:-1])
    cfa = raw.raw_image.copy().astype(float)
    mask = raw.raw_colors.copy()
    cam2rgb = raw.color_matrix.copy()

    # scale cfa to valid range
    scale(cfa, mask, black)
    rgb = deinterleave(cfa, mask)

    # white balance
    rgb = np.clip(rgb @ wb, 0, 1)

    return rgb, wb, cam2rgb


def scale(img, mask, black):
    """
    Scale 10b image from [black, 1023] -> [0, 1]

    Parameters
    ----------
    img : ndarray
        2D representation of CFA data
    mask : ndarray
        2D matrix same size as img, determines RGBG color
        R = 0, G1 = 1, B = 2, G2 = 3
    black: ndarray
        Contains black level for each color in CFA
    """

    masks = [mask == 0,
             mask == 1,
             mask == 3,
             mask == 2]

    for i in range(4):
        m = masks[i]
        b = black[i]
        a = 1 / (1023.0 - b)
        img[m] = a * (img[m] - b)


def deinterleave(img, mask):
    """
    Extract RGBG values into three separate matrices R, G, B.
    Performs first binning step in size reduction

    Parameters
    ----------
    img : ndarray
        2D representation of CFA data
    mask : ndarray
        2D matrix same size as img, determines RGBG color
        R = 0, G1 = 1, B = 2, G2 = 3
    """

    # generate shape for first 2x2 bin
    s = (int(img.shape[0] / 2), int(img.shape[1] / 2))

    # there are two green channels in CFA
    # average them for first binning step
    g = np.mean([img[mask == 1], img[mask == 3]], axis=0).reshape(s)

    # extract remaining channels
    r = np.reshape(img[mask == 0], s)
    b = np.reshape(img[mask == 2], s)

    return np.dstack((r, g, b))


def extract_patches(im, window_shape=(192, 192, 3), stride=192):
    if im.ndim < 3:
        im = np.dstack([im] * 3)
    patches = ski.util.view_as_windows(im, window_shape, stride)
    nr, nc, t, h, w, c = patches.shape
    n = nr * nc
    patches = np.reshape(patches, (n, h, w, c))
    return patches


def write_img(img, path):
    scaled = np.uint16(np.clip(img * 65535, 0, 65535))
    n = np.uint16((scaled.shape[0] * scaled.shape[1]) / (192 ** 2))
    patches = extract_patches(scaled)
    for i, p in enumerate(patches):
        imageio.imwrite(path + f'_p_' + str(i).zfill(5) + '.tif', p)


def process(raw_file):
    """
    Performs entire camera pipe on image with all highlight recovery methods
    and writes them in respective directories

    Parameters
    ----------
    raw_file : string
        path the the directory the raw file is in
    """

    try:
        with rawpy.imread(raw_file) as raw:
            # unpack raw data
            rgb, wb, cam2rgb = unpack(raw)

            file = Path(raw_file).stem
            path = '/media/eddie/DATA/patches/train/' + file

            with open(path + '.json', 'w') as of:
                of.write(json.dumps({'wb': wb.tolist(), 'cam2rgb': cam2rgb[:, :-1].tolist()}))

            # try:
            #     os.makedirs(path)
            # except OSError as e:
            #     if e.errno != errno.EEXIST:
            #         raise

            # After white balancing, the image is normally just clipped again
            write_img(rgb, path)
    except Exception as e:
        print("error converting file: " + raw_file)
        print(e)


def get_patch_hdr(target):
    if target.max() == 255:
        target = target / 255.0 * 65535

    # shot and read noise
    if random.random() > 0.5:
        scale = np.random.uniform(1, 2 ** 9)
        read = np.random.uniform(0, 2 ** -4)
        print('Read noise: ' + str(read))
        print('Shot noise scale: ' + str(scale))
        shot = np.random.poisson(target / scale) * scale
        noisy = shot + np.sqrt(read) * np.random.standard_normal(target.shape)
    else:
        noisy = target

    print('Noise mean: ' + str((noisy - target).mean()))

    # mosaic
    mask = np.zeros_like(target)

    # red
    mask[::2, ::2, 0] = 1

    # green
    mask[::2, 1::2, 1] = 1
    mask[1::2, ::2, 1] = 1

    # blue
    mask[1::2, 1::2, 2] = 1

    mosaiced = np.clip(noisy * mask, 0, 65535)

    print('Mosaiced max: ' + str(mosaiced.max()))
    print('Target max: ' + str(target.max()))

    # saturate
    sat_point = np.random.uniform(1, 2 ** 2)
    scaled = mosaiced / mosaiced.max() * sat_point
    saturated = scaled

    clip = [0, 0, 0]
    if random.random() > 0.5:
        clip[0] = 1
        saturated[:, :, 0] = np.clip(saturated[:, :, 0], 0, 1)
    if random.random() > 0.33:
        clip[1] = 1
        saturated[:, :, 1] = np.clip(saturated[:, :, 1], 0, 1)
    if random.random() > 0.5:
        clip[2] = 1
        saturated[:, :, 2] = np.clip(saturated[:, :, 2], 0, 1)

    target = target / target.max() * sat_point

    # if all three channels saturate, avoid reconstruction to prevent artifacts
    if np.sum(clip) == 3:
        target = np.clip(target, 0, 1)

    print('Saturation point: ' + str(sat_point))

    saturated = np.uint16(np.round(np.clip(hlg(saturated) * 65535, 0, 65535)))
    target = np.uint16(np.round(np.clip(hlg(target) * 65535, 0, 65535)))

    return saturated, target


def damage(image):
    """
    Damages an image with noise, a mosaic, and saturation

    Parameters
    ----------
    image : string
        path to the directory the image file is in
    """

    # unpack raw data
    try:
        im = imageio.imread(image)
    except:
        print("error reading file: " + image)
        return
    damaged, target = get_patch_hdr(im)

    file = Path(image).stem
    path = '/Volumes/Extreme Pro/damaged/test/'
    path = './'

    # imageio.imwrite(path + '/damaged/' + file + '_damaged.tif', damaged)
    # imageio.imwrite(path + '/target/' + file + '_target.tif', target)

    imageio.imwrite(file + '_damaged.tif', damaged)
    imageio.imwrite(file + '_target.tif', target)

    mosaic = np.sum(damaged, axis=2)
    mask = np.zeros_like(mosaic)
    mask[::2, 1::2] = 1
    mask[1::2, ::2] = 3
    mask[1::2, 1::2] = 2

    half = np.uint16(deinterleave(mosaic, mask))
    # imageio.imwrite(path + '/half/' + file + '_half.tif', half)
    imageio.imwrite(file + '_half.tif', half)


def hlg(rgb):
    rgb[rgb < 0] = 0
    mask = rgb > 1
    rgb[mask] = 0.17883277 * np.log(rgb[mask] - 0.28466892) + 0.55991073
    rgb[~mask] = 0.5 * np.sqrt(rgb[~mask])
    return rgb


if __name__ == "__main__":
    # Make the output dirs if they don't exist
    # out_dirs = ['test', 'train', 'val']
    #
    # for out_dir in out_dirs:
    #     try:
    #         os.makedirs(f'/Volumes/Extreme Pro/patches/{out_dir}')
    #     except OSError as e:
    #         if e.errno != errno.EEXIST:
    #             raise
    #
    # # Path to top level dir containing dataset folders
    #path = '/media/eddie/DATA/dngs/train'
    # # Get a list of full paths to each of the subdirs
    #raw_files = [f.path for f in os.scandir(path) if (not Path(f.path).stem.startswith('.') and f.path.endswith('.dng'))]
    # # Process each subdir
    #p_umap(process, raw_files)

    # path = '/Volumes/Extreme Pro/patches/test'
    # images = [f.path for f in os.scandir(path) if (not Path(f.path).stem.startswith('.') and f.path.endswith('.tif'))]
    # p_umap(damage, images)
    # damage('Covid19Lung.jpg')
    write_img(imageio.imread('samples/mehanik_SDR-circle-v2.tif')/65535.0, 'samples/patches/mehanik_SDR-circle-v2')
