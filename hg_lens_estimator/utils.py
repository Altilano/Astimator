import numpy as np
from scipy.special import comb
import cv2

def calculate_ax_ay(waist, lamb, z0, f, theta, z, flip_lens=False):

    zr = np.pi*(waist)**2/lamb
    fx = f*(np.cos(theta))**3
    fy = f*(np.cos(theta))
    if flip_lens == True:
        fx, fy = fy, fx
    zbar = zr*(1 + (z0/zr)**2)
    R =  (z0**2 + zr**2)/z0
    ax = np.pi/2 - np.arctan2(zbar*(fx*R -z*R+z*fx), z*fx*R) + np.arctan(z0/zr)
    ay = np.pi/2 - np.arctan2(zbar*(fy*R -z*R+z*fy), z*fy*R) + np.arctan(z0/zr)
    return ax, ay

def calculate_scale_factor(waist, lamb, z0, f, theta, z, flip_lens=False):
    zr = np.pi*(waist)**2/lamb
    fx = f*np.cos(theta)**3
    fy = f*np.cos(theta)
    if flip_lens == True:
        fx, fy = fy, fx
    zbar = zr*(1 + (z0/zr)**2)
    w_z0 = waist*np.sqrt(1+(z0/zr)**2)
    R =  (z0**2 + zr**2)/z0
    k = 2*np.pi/lamb
    factor_x = (zbar/z)*waist/(np.sqrt(1+zbar**2 * (1/z -1/fx + 1/R)**2))
    factor_y = (zbar/z)*waist/(np.sqrt(1+zbar**2 * (1/z -1/fy + 1/R)**2))
    return factor_x, factor_y


def normalize(img):
    img = img - img.min()
    return img / img.max()

def center_of_mass(img):
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = img
    y, x = np.indices(gray.shape)
    total = gray.sum()

    if total == 0:
        # fallback se a imagem for toda zero
        cy = img.shape[0] // 2
        cx = img.shape[1] // 2
    else:
        cx = (x * gray).sum() / total
        cy = (y * gray).sum() / total

    return int(cx), int(cy)

def stretch_zoom_and_resize(img, stretch_x, stretch_y, zoom, output_size=512):

    h, w = img.shape

    
    # Step 1: Apply stretching
    new_w = int(w * stretch_x)
    new_h = int(h * stretch_y)
    
    stretched = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Step 2: Apply zoom
    # Zoom > 1.0 means zoom in (crop), zoom < 1.0 means zoom out (will pad more)
    zoomed_w = int(new_w / zoom)
    zoomed_h = int(new_h / zoom)
    
    if zoom > 1.0:
        # Zoom in: crop from center

        start_x = (new_w - zoomed_w) // 2
        start_y = (new_h - zoomed_h) // 2
        zoomed = stretched[start_y:start_y+zoomed_h, start_x:start_x+zoomed_w]
    elif zoom < 1.0:
        # Zoom out: pad

        zoomed = np.zeros((zoomed_h, zoomed_w), dtype=img.dtype)
        start_x = (zoomed_w - new_w) // 2
        start_y = (zoomed_h - new_h) // 2
        zoomed[start_y:start_y+new_h, start_x:start_x+new_w] = stretched
    else:
        # No zoom
        zoomed = stretched
    
    # Step 3: Pad to square (use the larger dimension)
    final_h, final_w = zoomed.shape
    max_dim = max(final_w, final_h)
    padded = np.zeros((max_dim, max_dim), dtype=img.dtype)
    
    # Center the image
    start_y = (max_dim - final_h) // 2
    start_x = (max_dim - final_w) // 2
    padded[start_y:start_y+final_h, start_x:start_x+final_w] = zoomed
    
    
    # Step 4: Resize to final output size
    result = cv2.resize(padded, (output_size, output_size), interpolation=cv2.INTER_LINEAR)
    
    return result



def process_three_beams_manual(img1, img2, img3, 
                                stretch2_x=1.0, stretch2_y=1.0, zoom2=1.0,
                                stretch3_x=1.0, stretch3_y=1.0, zoom3=1.0,
                                output_size=512):

    h1, w1 = img1.shape
    max_dim1 = max(h1, w1)
    padded1 = np.zeros((max_dim1, max_dim1), dtype=img1.dtype)
    start_y1 = (max_dim1 - h1) // 2
    start_x1 = (max_dim1 - w1) // 2
    padded1[start_y1:start_y1+h1, start_x1:start_x1+w1] = img1
    result1 = cv2.resize(padded1, (output_size, output_size), interpolation=cv2.INTER_LINEAR)
    result2 = stretch_zoom_and_resize(img2, stretch2_x, stretch2_y, zoom2, output_size)
    result3 = stretch_zoom_and_resize(img3, stretch3_x, stretch3_y, zoom3, output_size)
    
    return result1, result2, result3


def recenter_image(img, cx, cy):
    h, w = img.shape[:2]
    img = img.astype(np.float32)

    target_x = w // 2
    target_y = h // 2
    shift_x = target_x - cx
    shift_y = target_y - cy

    M = np.float32([[1, 0, shift_x],
                    [0, 1, shift_y]])

    # Shift the image with smooth interpolation
    img_shifted = cv2.warpAffine(img, M, (w, h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Shift the original binary mask the same way (nearest to avoid bleeding)
    mask = (img > 0).astype(np.float32)
    mask_shifted = cv2.warpAffine(mask, M, (w, h),
                                  flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Zero out any pixels that fall outside the original mask
    img_shifted *= mask_shifted

    return img_shifted

def background_from_edges(img, border=20, factor=1):
    edges = np.concatenate([
        img[:border, :].ravel(),
        img[-border:, :].ravel(),
        img[:, :border].ravel(),
        img[:, -border:].ravel()
    ])
    B = factor * np.median(edges)
    corrected = img - B
    corrected[corrected < 0] = 0
    return corrected