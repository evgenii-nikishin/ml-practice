import numpy as np


def normalize_data(images):
    N, D = images.shape
    
    shapes = (int(np.sqrt(D/3)), int(np.sqrt(D/3)), 3)
    images = images.reshape(N, *shapes)

    means = np.mean(images, axis=(1, 2))
    stds = np.std(images, axis=(1, 2))
    
    # cut anomalies
    images = np.maximum(images, (means - 3*stds)[:, np.newaxis, np.newaxis, :])
    images = np.minimum(images, (means + 3*stds)[:, np.newaxis, np.newaxis, :])

    # make images belong [0.1, 0.9]
    max_channel = np.max(images, axis=(0, 1, 2))#[:, np.newaxis, np.newaxis, :]
    min_channel = np.min(images, axis=(0, 1, 2))#[:, np.newaxis, np.newaxis, :]
    images = (images - min_channel) / (max_channel - min_channel) * 0.8 + 0.1
    
    return images.reshape(N, -1)


def sample_patches_raw(images, num_patches=10000, patch_size=8):
    N, D = images.shape
    d = int(np.sqrt(D/3))
    pics = images.reshape(N, d, d, 3)
    
    source_pics = np.random.choice(N, size=num_patches)
    start_points = np.random.randint(0, d-patch_size+1, size=num_patches)
    res = np.zeros((num_patches, patch_size, patch_size, 3), dtype='uint')
    for i in range(num_patches):
        res[i] = pics[source_pics[i], start_points[i] : start_points[i]+patch_size, 
                   start_points[i] : start_points[i]+patch_size, :]
    return res.reshape(num_patches, -1)


def sample_patches(images, num_patches=10000, patch_size=8):
    raw = sample_patches_raw(images, num_patches, patch_size)
    res = normalize_data(raw)
    return res