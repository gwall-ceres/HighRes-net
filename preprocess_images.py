import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform, img_as_float, exposure
from skimage.metrics import structural_similarity as ssim
from skimage.measure import ransac
from skimage.transform import SimilarityTransform
from skimage.feature import ORB, match_descriptors
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift as ndi_shift

import torch
import torch.nn.functional as F
import os
import random
import json
import time


def compute_perceptual_loss(model, ref_image, mov_image, ref_mask, mov_mask, device='cpu'):
    # Input validation
    assert ref_image.shape[:2] == mov_image.shape[:2], "Images must have same dimensions"
    assert ref_mask.shape == mov_mask.shape, "Masks must have same dimensions"
    
    # combine the two masks in order to only consider valid pixels
    combined_mask = ref_mask.astype(float) * mov_mask.astype(float)
    # Apply masks to input images
    ref_masked = ref_image * combined_mask
    mov_masked = mov_image * combined_mask
    
    # Extract features using the model (which handles grayscale conversion)
    with torch.no_grad():
        ref_features = model(ref_masked)
        mov_features = model(mov_masked)
    
    # Ensure consistent feature ordering
    if isinstance(ref_features, dict):
        feature_names = sorted(ref_features.keys())
        ref_features = [ref_features[name] for name in feature_names]
        mov_features = [mov_features[name] for name in feature_names]
    
    # Initialize loss tracking
    total_loss = 0.0
    diff_features = {}
    # Weights for VGG19 layers ['0', '5', '10', '19', '28']
    # Layer 0: First conv layer - highest detail
    # Layer 5: End of first conv block
    # Layer 10: End of second conv block
    # Layer 19: Deep in third conv block
    # Layer 28: Deep features
    layer_weights = [1.0, 0.8, 0.6, 0.4, 0.2]  # Decreasing weights for deeper layers
    
    # Process each layer with proper mask handling
    for idx, (ref_feat, mov_feat) in enumerate(zip(ref_features, mov_features)):
        _, C, Hf, Wf = ref_feat.shape
        
        # Resize mask to match feature map size
        mask_resized = transform.resize(
            combined_mask,
            (Hf, Wf),
            order=1,  # Use bilinear interpolation for mask
            mode='constant',
            cval=0,
            anti_aliasing=True,
            preserve_range=True
        ).astype(np.float32)
        
        # Convert mask to tensor and expand to match feature dimensions
        mask_tensor = torch.from_numpy(mask_resized).unsqueeze(0).unsqueeze(0).to(device)
        mask_expanded = mask_tensor.expand_as(ref_feat)
        
        # Apply mask to features
        ref_feat_masked = ref_feat * mask_expanded
        mov_feat_masked = mov_feat * mask_expanded
        
        # Calculate valid pixels for this layer
        num_valid = torch.sum(mask_expanded)
        if num_valid > 0:
            # Compute masked L1 loss directly on feature values
            layer_loss = F.l1_loss(ref_feat_masked, mov_feat_masked, reduction='sum') / num_valid
            
            # Store feature differences for visualization
            l1_diff = torch.abs(ref_feat_masked - mov_feat_masked)
            l1_diff_summed = l1_diff.sum(dim=1).squeeze(0) / num_valid
            diff_features[feature_names[idx]] = l1_diff_summed.detach().cpu().numpy()
            
            # Apply layer weight
            weight = layer_weights[idx] if idx < len(layer_weights) else layer_weights[-1]
            total_loss += weight * layer_loss.item()
    
    # Normalize by sum of weights
    total_loss /= sum(layer_weights[:len(ref_features)])
    
    return total_loss, diff_features

def compute_masked_ncc(ref_image, template_image, ref_mask, template_mask):
    """
    Compute the Weighted Normalized Cross-Correlation (NCC) between two images.

    Parameters:
        ref_image (np.ndarray): Reference image array.
        template_image (np.ndarray): Shifted template image array.
        ref_mask (np.ndarray): Reference mask array with values between 0 and 1.
        template_mask (np.ndarray): Template mask array with values between 0 and 1.

    Returns:
        float: Weighted NCC value.
    """
    # Compute combined weights
    weights = ref_mask.astype(float) * template_mask.astype(float)

    # Sum of weights
    weight_sum = np.sum(weights)
    if weight_sum == 0:
        return np.nan  # or handle as appropriate

    # Compute weighted means
    mu_ref = np.sum(ref_image * weights) / weight_sum
    mu_template = np.sum(template_image * weights) / weight_sum

    # Compute weighted standard deviations
    sigma_ref = np.sqrt(np.sum(weights * (ref_image - mu_ref) ** 2) / weight_sum)
    sigma_template = np.sqrt(np.sum(weights * (template_image - mu_template) ** 2) / weight_sum)

    # Avoid division by zero
    if sigma_ref == 0 or sigma_template == 0:
        return np.nan  # or handle as appropriate

    # Compute weighted covariance
    covariance = np.sum(weights * (ref_image - mu_ref) * (template_image - mu_template)) / weight_sum

    # Compute weighted NCC
    ncc = covariance / (sigma_ref * sigma_template)

    return ncc

def compute_sum_of_layers(diff_features, normalize=True):
    """
    Compute the sum of all layers in diff_features dictionary.
    This is used to visualize the sum of all layers in VGG feature space.
    If normalize is True, normalize each layer to [0,1] range before summing.
    """
    
    if diff_features is None:
        return np.zeros((10,10), dtype=float)
    
    
    # Find the largest dimensions among all layers
    max_height = 0
    max_width = 0
    for layer in diff_features:
        activations = diff_features[layer]
        height, width = activations.shape
        max_height = max(max_height, height)
        max_width = max(max_width, width)
    
    # Create array to store the sum of all layers
    summed_activations = np.zeros((max_height, max_width))
    
    # Add each normalized layer to the sum
    for layer in diff_features:
        activations = diff_features[layer]
        
        # Normalize the activations to [0,1] range
        layer_max = np.max(np.abs(activations))
        if layer_max > 0 and normalize:  # Avoid division by zero
            normalized_activations = activations / layer_max
        else:
            normalized_activations = activations
            
        # Scale to largest dimensions if necessary
        if normalized_activations.shape != (max_height, max_width):
            scaled_activations = transform.resize(
                normalized_activations,
                (max_height, max_width),
                order=3,  # Cubic interpolation
                mode='reflect',
                anti_aliasing=True,
                preserve_range=True
            )
        else:
            scaled_activations = normalized_activations
            
        # Add to sum
        summed_activations += scaled_activations
    
    if normalize:
        # Optionally normalize the final sum to [0,1] range
        final_max = np.max(np.abs(summed_activations))
        if final_max > 0:
            summed_activations /= final_max
        
    return summed_activations

def contrast_stretch_8bit(image, mask=None):
    """
    Perform contrast stretching on a NumPy array to map its values to the 0-255 range.
    
    Parameters:
        image (np.ndarray): Input image to be contrast stretched
        mask (np.ndarray, optional): Mask where True/1 indicates valid pixels to include
                                   in histogram calculation. If None, all pixels are used.
    
    Returns:
        np.ndarray: Contrast-stretched image as uint8
    """
    if mask is None:
        # If no mask provided, use all pixels
        p1 = np.percentile(image, 1)
        p99 = np.percentile(image, 99)
    else:
        if mask.dtype != bool:
            mask = (mask > 0.5).astype(bool)
        # Calculate percentiles only for masked pixels
        valid_pixels = image[mask]
        if len(valid_pixels) == 0:
            return np.zeros_like(image, dtype=np.uint8)
            
        p1 = np.percentile(valid_pixels, 1)
        p99 = np.percentile(valid_pixels, 99)

    # Apply contrast stretching to the entire image using the computed intensity range
    stretched = exposure.rescale_intensity(
        image,
        in_range=(p1, p99),
        out_range=(0, 255)
    ).astype(np.uint8)

    return stretched

def min_max_scale(image):
    min_val = np.min(image)
    max_val = np.max(image)
    scaled_image = (image - min_val) / (max_val - min_val + 1e-8)  # Avoid division by zero
    return scaled_image


# Normalize image to zero mean and unit std_dev
def normalize_masked_array(masked_array):
    epsilon = 1e-8
    mean = np.mean(masked_array)
    std = np.std(masked_array)
    if std == 0:
        raise ValueError("Standard deviation is zero. Cannot normalize an array with constant values.")
    normalized_array = (masked_array - mean) / (std + epsilon)
    return normalized_array


def compute_mse(image1, mask1, image2, mask2, use_masks=True, normalize=True):
    #only consider pixels that are valid in both images, so we "and" the masks

    if use_masks:
        combined_mask = np.logical_and(mask1, mask2)
        # Apply the combined mask to both images
        masked_image1 = image1[combined_mask]
        masked_image2 = image2[combined_mask]
    else:
        masked_image1 = image1
        masked_image2 = image2

    # Normalize both masked images
    if normalize:
        normalized_image1 = normalize_masked_array(masked_image1)
        normalized_image2 = normalize_masked_array(masked_image2)
        return np.mean((normalized_image1 - normalized_image2) ** 2)
    else:
        return np.mean((masked_image1 - masked_image2) ** 2)

def compute_ml1e(image1, mask1, image2, mask2, use_masks=True, normalize=True):
    #only consider pixels that are valid in both images, so we "and" the masks

    if use_masks:
        combined_mask = np.logical_and(mask1, mask2)
        # Apply the combined mask to both images
        masked_image1 = image1[combined_mask]
        masked_image2 = image2[combined_mask]
    else:
        masked_image1 = image1
        masked_image2 = image2

    # Normalize both masked images
    if normalize:
        normalized_image1 = normalize_masked_array(masked_image1)
        normalized_image2 = normalize_masked_array(masked_image2)
        return np.mean((normalized_image1 - normalized_image2) ** 2)
    else:
        return np.mean(np.abs(masked_image1 - masked_image2))
    
def compute_ssim(image1, mask1, image2, mask2, use_masks=True, normalize=True):
    """
    Compute the Structural Similarity Index Measure (SSIM) between two images,
    optionally considering only the overlapping valid regions defined by masks.

    Parameters:
    - image1 (np.ndarray): First image array, shape (H, W) or (H, W, C).
    - mask1 (np.ndarray): Boolean mask for the first image, shape (H, W).
    - image2 (np.ndarray): Second image array, shape (H, W) or (H, W, C).
    - mask2 (np.ndarray): Boolean mask for the second image, shape (H, W).
    - use_masks (bool): If True, only consider pixels where both masks are True.
    - normalize (bool): If True, normalize the masked images before computing SSIM.

    Returns:
    - ssim_value (float): The computed SSIM.

    Raises:
    - ValueError: If there are no overlapping valid pixels when `use_masks` is True.
    - ValueError: If input images and masks have incompatible shapes.
    """
    # Ensure both images have the same spatial dimensions
    if image1.shape[:2] != image2.shape[:2]:
        raise ValueError("Reference and moving images must have the same height and width.")
    if mask1.shape != mask2.shape:
        raise ValueError("Reference and moving masks must have the same height and width.")

    # Combine masks to focus on overlapping valid regions
    if use_masks:
        combined_mask = np.logical_and(mask1, mask2)  # Shape: (H, W)
        if not np.any(combined_mask):
            raise ValueError("No overlapping valid pixels found between the masks.")
        # Apply the combined mask to both images
        masked_image1 = image1.copy()
        masked_image2 = image2.copy()
        masked_image1[~combined_mask] = 0
        masked_image2[~combined_mask] = 0
    else:
        masked_image1 = image1
        masked_image2 = image2

    # Normalize images if required
    if normalize:
        masked_image1 = normalize_masked_array(masked_image1)
        masked_image2 = normalize_masked_array(masked_image2)

    # Determine the data range
    # Option A: If images are known to be in [0, 1]
    # data_range = 1.0

    # Option B: If images are in [0, 255]
    # data_range = 255.0

    # Option C: Compute based on image data
    # Choose one based on your data

    # Here, we choose Option C to make the function flexible
    data_min = min(masked_image1.min(), masked_image2.min())
    data_max = max(masked_image1.max(), masked_image2.max())
    data_range = data_max - data_min

    if data_range <= 0:
        raise ValueError("Data range must be positive.")

    # Handle grayscale and multichannel images
    if masked_image1.ndim == 2:
        # Grayscale images
        ssim_value = ssim(masked_image1, masked_image2, data_range=data_range)
    elif masked_image1.ndim == 3 and masked_image1.shape[2] in [1, 3, 4]:
        # Multichannel images (e.g., RGB)
        ssim_value = ssim(masked_image1, masked_image2, data_range=data_range, multichannel=True)
    else:
        raise ValueError("Input images must have shape (H, W), (H, W, 1), (H, W, 3), or (H, W, 4)")

    return ssim_value


def process_image_for_display(image, p2=1, p98=99):
    """
    Normalize and contrast-stretch an image for better visualization.
    
    Parameters:
        image (np.ndarray): Input image array (grayscale or RGB).
        p2 (float): Lower percentile for contrast stretching.
        p98 (float): Upper percentile for contrast stretching.
        
    Returns:
        processed_image (np.ndarray): Image normalized to [0, 1] after contrast stretching.
    """
    # If the image has multiple channels (e.g., RGB), process each channel separately
    if image.ndim == 3:
        processed_channels = []
        for c in range(image.shape[2]):
            channel = image[:, :, c]
            # Normalize the channel
            channel_normalized = exposure.rescale_intensity(
                channel,
                in_range=(np.percentile(channel, p2), np.percentile(channel, p98)),
                out_range=(0, 1)
            )
            processed_channels.append(channel_normalized)
        processed_image = np.stack(processed_channels, axis=2)
    else:
        # Grayscale image
        processed_image = exposure.rescale_intensity(
            image,
            in_range=(np.percentile(image, p2), np.percentile(image, p98)),
            out_range=(0, 1)
        )
    
    return processed_image



def compute_grid_mncc(norm_ref, ref_mask, template_image, template_mask, bounds_y, bounds_x, points_per_dim):
    """
    Compute MNCC scores for a grid of points within given bounds.
    
    Parameters:
        norm_ref (np.ndarray): Normalized reference image
        ref_mask (np.ndarray): Reference mask as float [0,1]
        template_image (np.ndarray): Template image to be aligned
        template_mask (np.ndarray): Template mask as bool
        bounds_y (tuple): (min_y, max_y) bounds for vertical shifts
        bounds_x (tuple): (min_x, max_x) bounds for horizontal shifts
        points_per_dim (int): Number of points to evaluate per dimension
        
    Returns:
        tuple: (best_shift_y, best_shift_x, best_score)
    """
    min_y, max_y = bounds_y
    min_x, max_x = bounds_x
    
    # Generate grid points
    y_points = np.linspace(min_y, max_y, points_per_dim)
    x_points = np.linspace(min_x, max_x, points_per_dim)
    
    best_score = float('-inf')
    best_shift = (0.0, 0.0)
    
    # Evaluate MNCC at each grid point
    for dy in y_points:
        for dx in x_points:
            # Shift template image and mask
            shifted_template = ndi_shift(template_image, (dy, dx), mode='constant', order=3)
            shifted_mask = ndi_shift(template_mask.astype(float), (dy, dx), 
                                  mode='constant', order=1)
            shifted_mask = (shifted_mask > 0.5).astype(float)
            
            # Compute combined mask
            combined_mask = ref_mask * shifted_mask
            weight_sum = np.sum(combined_mask)
            
            if weight_sum > 0:
                # Compute normalized template statistics
                mu_template = np.sum(shifted_template * combined_mask) / weight_sum
                sigma_template = np.sqrt(np.sum(combined_mask * 
                                              (shifted_template - mu_template) ** 2) / 
                                      weight_sum)
                
                if sigma_template > 0:
                    # Compute correlation using pre-computed normalized reference
                    norm_template = (shifted_template - mu_template) / sigma_template
                    mncc = np.sum(combined_mask * norm_ref * norm_template) / weight_sum
                    
                    if mncc > best_score:
                        best_score = mncc
                        best_shift = (dy, dx)
    
    return best_shift[0], best_shift[1], best_score


def recursive_mncc_search(norm_ref, ref_mask, template_image, template_mask, 
                         points_per_dim, scale_factor, max_recursions, current_recursion=0,
                         prev_best_dy=0.0, prev_best_dx=0.0):
    """
    Recursive function to search for the best alignment using MNCC.
    
    Parameters:
        norm_ref (np.ndarray): Normalized reference image
        ref_mask (np.ndarray): Reference mask as float [0,1]
        template_image (np.ndarray): Template image to be aligned
        template_mask (np.ndarray): Template mask as bool
        points_per_dim (int): Number of points to evaluate per dimension
        scale_factor (float): Factor to scale bounds by for next recursion
        max_recursions (int): Maximum recursion depth
        current_recursion (int): Current recursion depth
        prev_best_dy (float): Best dy from previous recursion
        prev_best_dx (float): Best dx from previous recursion
        
    Returns:
        tuple: Best shift (delta_y, delta_x)
    """
    # Compute current bounds based on recursion level
    bound_width = 2.0 * (scale_factor ** current_recursion)
    bounds_y = (prev_best_dy - bound_width/2, prev_best_dy + bound_width/2)
    bounds_x = (prev_best_dx - bound_width/2, prev_best_dx + bound_width/2)
    print(f"recursive_mncc_search::bound_width {bound_width}, current_recursion {current_recursion}")
    print(f"recursive_mncc_search::prev_best_dy {prev_best_dy} prev_best_dx {prev_best_dx}")
    print(f"recursive_mncc_search::bounds_y {bounds_y} bounds_x {bounds_x}")

    # Compute best shift for current grid
    best_dy, best_dx, best_score = compute_grid_mncc(
        norm_ref, ref_mask, template_image, template_mask,
        bounds_y, bounds_x, points_per_dim
    )
    print(f"recursive_mncc_search::best_score {best_score}")
    
    # Base case: reached maximum recursions
    if current_recursion >= max_recursions - 1:
        return best_dy, best_dx
    
    # Recursive case: continue search with refined bounds
    return recursive_mncc_search(
        norm_ref, ref_mask, template_image, template_mask,
        points_per_dim, scale_factor, max_recursions,
        current_recursion + 1, best_dy, best_dx
    )

def compute_shift_ncc(ref_image, template_image, ref_mask, template_mask, 
                      points_per_dim=5, max_recursions=8):
    """
    Main function to compute the best shift between reference and template images using MNCC.
    
    Parameters:
        ref_image (np.ndarray): Reference image
        template_image (np.ndarray): Template image to be aligned
        ref_mask (np.ndarray): Reference mask (binary)
        template_mask (np.ndarray): Template mask (binary)
        points_per_dim (int): Number of points to evaluate per dimension
        max_recursions (int): Maximum number of recursive calls
    
    Returns:
        tuple: Best shift (delta_y, delta_x)
    """
    if points_per_dim < 3:
        raise ValueError("points_per_dim must be at least 3")
    if max_recursions < 3:
        raise ValueError("max_recursions must be at least 3")
    
    # Convert masks to float
    ref_mask_float = ref_mask.astype(float)
    
    # Pre-compute normalized reference image statistics
    weight_sum = np.sum(ref_mask_float)
    if weight_sum == 0:
        return 0.0, 0.0
        
    mu_ref = np.sum(ref_image * ref_mask_float) / weight_sum
    sigma_ref = np.sqrt(np.sum(ref_mask_float * (ref_image - mu_ref) ** 2) / weight_sum)
    
    if sigma_ref == 0:
        return 0.0, 0.0
        
    norm_ref = (ref_image - mu_ref) / sigma_ref
    
    # Calculate scale factor based on points_per_dim
    # For example, if points_per_dim = 4, scale_factor ≈ 0.5
    # if points_per_dim = 5, scale_factor ≈ 0.25
    scale_factor = 1.0 / (points_per_dim - 2)
    if scale_factor < 0.25:
        scale_factor = 0.25
    if scale_factor >= 1.0:
        scale_factor = 0.9
    
    # Start recursive search
    return recursive_mncc_search(
        norm_ref, ref_mask_float, template_image, template_mask,
        points_per_dim, scale_factor, max_recursions
    )

def compute_shift_pl(model, ref_image, template_image, ref_mask, template_mask, 
                    points_per_dim=3, max_recursions=3):
    """
    Main function to compute the best shift between reference and template images using
    perceptual loss with VGG features.
    
    Parameters:
        model: VGG-based perceptual loss network
        ref_image (np.ndarray): Reference image
        template_image (np.ndarray): Template image to be aligned
        ref_mask (np.ndarray): Reference mask (binary)
        template_mask (np.ndarray): Template mask (binary)
        points_per_dim (int): Number of points to evaluate per dimension
        max_recursions (int): Maximum number of recursive calls
    
    Returns:
        tuple: Best shift (delta_y, delta_x)
    """
    # Calculate scale factor based on points_per_dim
    scale_factor = 1.0 / (points_per_dim - 1)
    if scale_factor < 0.25:
        scale_factor = 0.25
    if scale_factor >= 1.0:
        scale_factor = 0.9
    
    # Start recursive search
    return recursive_pl_search(
        model, ref_image, ref_mask, template_image, template_mask,
        points_per_dim, scale_factor, max_recursions
    )

def recursive_pl_search(model, ref_image, ref_mask, template_image, template_mask, 
                       points_per_dim, scale_factor, max_recursions, current_recursion=0,
                       prev_best_dy=0.0, prev_best_dx=0.0):
    """
    Recursive function to search for the best alignment using perceptual loss.
    
    Parameters:
        model: VGG-based perceptual loss network
        ref_image (np.ndarray): Reference image
        ref_mask (np.ndarray): Reference mask
        template_image (np.ndarray): Template image to be aligned
        template_mask (np.ndarray): Template mask
        points_per_dim (int): Number of points to evaluate per dimension
        scale_factor (float): Factor to scale bounds by for next recursion
        max_recursions (int): Maximum recursion depth
        current_recursion (int): Current recursion depth
        prev_best_dy (float): Best dy from previous recursion
        prev_best_dx (float): Best dx from previous recursion
        
    Returns:
        tuple: Best shift (delta_y, delta_x)
    """
    # Compute current bounds based on recursion level
    bound_width = 2.0 * (scale_factor ** current_recursion)
    bounds_y = (prev_best_dy - bound_width/2, prev_best_dy + bound_width/2)
    bounds_x = (prev_best_dx - bound_width/2, prev_best_dx + bound_width/2)

    print(f"recursive_pl_search::bound_width {bound_width}, current_recursion {current_recursion}")
    print(f"recursive_pl_search::prev_best_dy {prev_best_dy} prev_best_dx {prev_best_dx}")
    print(f"recursive_pl_search::bounds_y {bounds_y} bounds_x {bounds_x}")
    
    # Compute best shift for current grid
    best_dy, best_dx, best_score = compute_grid_pl(
        model, ref_image, ref_mask, template_image, template_mask,
        bounds_y, bounds_x, points_per_dim
    )
    print(f"recursive_pl_search::best_score {best_score}, best_dy {best_dy}, best_dx {best_dx}")
    # Base case: reached maximum recursions
    if current_recursion >= max_recursions - 1:
        return best_dy, best_dx
    
    # Recursive case: continue search with refined bounds
    return recursive_pl_search(
        model, ref_image, ref_mask, template_image, template_mask,
        points_per_dim, scale_factor, max_recursions,
        current_recursion + 1, best_dy, best_dx
    )

def compute_grid_pl(model, ref_image, ref_mask, template_image, template_mask, 
                   bounds_y, bounds_x, points_per_dim):
    """
    Compute perceptual loss scores for a grid of points within given bounds.
    
    Parameters:
        model: VGG-based perceptual loss network
        ref_image (np.ndarray): Reference image
        ref_mask (np.ndarray): Reference mask
        template_image (np.ndarray): Template image to be aligned
        template_mask (np.ndarray): Template mask
        bounds_y (tuple): (min_y, max_y) bounds for vertical shifts
        bounds_x (tuple): (min_x, max_x) bounds for horizontal shifts
        points_per_dim (int): Number of points to evaluate per dimension
        
    Returns:
        tuple: (best_shift_y, best_shift_x, best_score)
    """
    min_y, max_y = bounds_y
    min_x, max_x = bounds_x
    
    # Generate grid points
    y_points = np.linspace(min_y, max_y, points_per_dim)
    x_points = np.linspace(min_x, max_x, points_per_dim)
    
    best_score = float('inf')  # We want to minimize perceptual loss
    best_shift = (0.0, 0.0)
    
    # Evaluate perceptual loss at each grid point
    for dy in y_points:
        for dx in x_points:
            # Shift template image and mask
            shifted_template = ndi_shift(template_image, (dy, dx), mode='constant', order=3)
            shifted_mask = ndi_shift(template_mask.astype(float), (dy, dx), 
                                   mode='constant', order=1)
            shifted_mask = (shifted_mask > 0.5).astype(float)
            
            # Compute perceptual loss between reference and shifted template
            pl_score, _ = compute_perceptual_loss(
                model=model,
                ref_image=ref_image,
                mov_image=shifted_template,
                ref_mask=ref_mask,
                mov_mask=shifted_mask,
                device=model.hardware
            )
            
            if pl_score < best_score:
                best_score = pl_score
                best_shift = (dy, dx)
    
    return best_shift[0], best_shift[1], best_score

def compute_shift_pcc(ref_image, shifted_image, ref_mask, shifted_mask):
    #i have different way of applying the mask to the image before computing the phase cross correlation
    #for some corner cases where there are a lot of invalid pixels, fully blacking out the masked pixels 
    #results in very wrong results. Weighting them by half seems like it is an ok compromise. Needs more testing though.
    #ref_masked = ref_image * (ref_mask.astype(ref_image.dtype) * 0.1 + 0.9)
    #mov_masked = shifted_image * (shifted_mask.astype(shifted_image.dtype) * 0.1 + 0.9)
    #completely black out the masked pixels
    #ref_masked = ref_image * ref_mask.astype(ref_image.dtype)
    #mov_masked = shifted_image * shifted_mask.astype(shifted_image.dtype)
    #do not mask the pixels at all
    ref_masked = ref_image 
    mov_masked = shifted_image

    # Compute the shift (delta_x, delta_y) between reference and template image
    shift_yx, error, diffphase = phase_cross_correlation(
            ref_image,
            mov_masked,
            upsample_factor=1000
        )
    return shift_yx
      

def compute_shift_point_matching(ref_image, tmplt_image, n_keypoints=500, match_threshold=0.75, ransac_threshold=2, scale=4):
    """
    Aligns two images using point matching to estimate a subpixel shift.

    Parameters:
    - ref_image: Reference image (numpy array).
    - tmplt_image: Image to be aligned (numpy array).
    - n_keypoints: Number of keypoints to detect.
    - match_threshold: Threshold for Lowe's ratio test (not directly used here).
    - ransac_threshold: RANSAC inlier threshold in pixels.
    - scale: Scale factor for the images.

    Returns:
    - shift_yx: Estimated (y, x) shift.
    """
    shape = ref_image.shape
    # Convert images to floating point
    image1 = transform.resize(
                img_as_float(ref_image),
                (shape[0]*scale, shape[1]*scale),
                order=3,  # Cubic interpolation
                mode='reflect',
                anti_aliasing=True,
                preserve_range=True)
    

    image2 = transform.resize(
                img_as_float(tmplt_image),
                (shape[0]*scale, shape[1]*scale),
                order=3,  # Cubic interpolation
                mode='reflect',
                anti_aliasing=True,
                preserve_range=True)

    # Initialize ORB detector
    orb = ORB(n_keypoints=n_keypoints, fast_threshold=0.05)

    # Detect and extract features from image1
    orb.detect_and_extract(image1)
    keypoints1 = orb.keypoints
    descriptors1 = orb.descriptors

    # Detect and extract features from image2
    orb.detect_and_extract(image2)
    keypoints2 = orb.keypoints
    descriptors2 = orb.descriptors

    # Match descriptors using mutual nearest neighbors
    matches12 = match_descriptors(descriptors1, descriptors2, cross_check=True)

    if len(matches12) < 4:
        raise ValueError("Not enough matches found for reliable alignment.")

    # Extract matched keypoints
    src = keypoints1[matches12[:, 0]][:, ::-1]  # (x, y)
    dst = keypoints2[matches12[:, 1]][:, ::-1]  # (x, y)

    # Estimate translation using RANSAC
    # Define a transformation model: translation only
    model = SimilarityTransform(scale=1, rotation=0, translation=(0, 0))
    try:
        model_robust, inliers = ransac((dst, src),
                                       SimilarityTransform,
                                       min_samples=2,
                                       residual_threshold=ransac_threshold,
                                       max_trials=1000)
    except Exception as e:
        raise ValueError(f"RANSAC failed to find a robust model: {e}")

    if model_robust is None:
        raise ValueError("RANSAC failed to find a robust model.")

    # Extract translation
    estimated_translation = model_robust.translation  # (tx, ty)

    # Since SimilarityTransform includes both x and y translations,
    # and we set rotation=0 and scale=1, we can directly use the translation components.
    shift_x, shift_y = estimated_translation  # (x_shift, y_shift)

    # Apply the estimated shift using scipy.ndimage.shift
    # Note: shift expects (y, x) order
    #aligned_image = ndi_shift(image2, shift=(-shift_y, -shift_x), mode='nearest', order=3)

    # The total estimated shift is (y, x)
    shift_yx = (shift_y/scale, shift_x/scale)
    print(f"point matching shift_yx {shift_yx}")
    return shift_yx



def iterative_align_refinement_with_perceptual_loss(
    model,
    ref_image,
    ref_mask,
    original_moving_image,
    original_moving_mask,
    max_iterations=10,
    shift_dampening=1.0,
    debug=False
):
    total_shift_y = 0.0
    total_shift_x = 0.0
    shifts_history = [(0.0, 0.0)]
    pl_history = []
    mse_history = []
    ssim_history = []

    for iteration in range(max_iterations):
        #print(f"\n--- Iteration {iteration} ---")

        if iteration == 0:
            shifted_image = original_moving_image
            shifted_mask = original_moving_mask.astype(float)
        else:
            # Apply the accumulated shift to the original moving image
            shifted_image = ndi_shift(
                original_moving_image,
                shift=(total_shift_y, total_shift_x),
                mode='reflect',
                order=3
            )
            shifted_mask = ndi_shift(
                original_moving_mask.astype(float),
                shift=(total_shift_y, total_shift_x),
                mode='constant',
                order=3
            )
        shifted_mask = shifted_mask > 0.5  # Re-binarize the mask

        # Compute perceptual loss between reference and shifted moving image
        perceptual_loss, diff_features = compute_perceptual_loss(model, ref_image, shifted_image, ref_mask, shifted_mask, model.hardware, False)
        mse = compute_mse(ref_image, ref_mask, shifted_image, shifted_mask, use_masks=True, normalize=True)
        ssim_score = compute_ssim(ref_image, ref_mask, shifted_image, shifted_mask, use_masks=True)

        print(
            f"Iteration {iteration}: Shift=({float(total_shift_y):.4f}, {float(total_shift_x):.4f}), Perceptual Loss={perceptual_loss:.6f}, MSE={mse:.6f}, SSIM={ssim_score:.8f}")
        pl_history.append(perceptual_loss)
        mse_history.append(mse)
        ssim_history.append(ssim_score)

        #i have different way of applying the mask to the image before computing the phase cross correlation
        #for some corner cases where there are a lot of invalid pixels, fully blacking out the masked pixels 
        #results in very wrong results. Weighting them by half seems like it is an ok compromise. Needs more testing though.
        ref_masked = ref_image * (ref_mask.astype(ref_image.dtype) * 0.5 + 0.5)
        mov_masked = shifted_image * (shifted_mask.astype(shifted_image.dtype) * 0.5 + 0.5)
        #completely black out the masked pixels
        #ref_masked = ref_image * ref_mask.astype(ref_image.dtype)
        #mov_masked = shifted_image * shifted_mask.astype(shifted_image.dtype)
        #do not mask the pixels at all
        #ref_masked = ref_image 
        #mov_masked = shifted_image

        #don't bother estimating the shift on the final iterations of the loop because we won't use it
        if iteration < max_iterations - 1:
            #find the best alignment
            shift_yx, error, diffphase = phase_cross_correlation(
                ref_masked,
                mov_masked,
                upsample_factor=2**(iteration+2)
            )
            delta_y, delta_x = shift_yx


            # Apply shift dampening to prevent overshooting
            dampened_shift_y = delta_y * shift_dampening
            dampened_shift_x = delta_x * shift_dampening

            # Accumulate shifts
            total_shift_y += dampened_shift_y
            total_shift_x += dampened_shift_x
            shifts_history.append((total_shift_y, total_shift_x))

    return shifts_history, pl_history, mse_history, ssim_history


# Define a function to create the 3x2 grid of plots
def plot_metrics_vs_shifts(shift_x, shift_y, ssim, mse, pl):
    """
    Plot SSIM, MSE, and Perceptual Loss against Shift X and Shift Y in a 3x2 grid.

    Parameters:
    - shift_x (list or array): Horizontal shifts.
    - shift_y (list or array): Vertical shifts.
    - ssim (list or array): SSIM values.
    - mse (list or array): MSE values.
    - pl (list or array): Perceptual Loss values.
    """
    fig, axs = plt.subplots(3, 2, figsize=(14, 18))

    # Metrics Names and Data
    metrics = {
        'SSIM': ssim,
        'MSE': mse,
        'Perceptual Loss': pl
    }

    # Iterate over metrics and create subplots
    for idx, (metric_name, metric_values) in enumerate(metrics.items()):
        # Plot vs Shift X
        axs[idx, 0].plot(shift_x, metric_values, marker='o', linestyle='-', color='b')
        axs[idx, 0].set_xlabel('Shift X (pixels)', fontsize=12)
        axs[idx, 0].set_ylabel(metric_name, fontsize=12)
        axs[idx, 0].set_title(f'{metric_name} vs. Shift X', fontsize=14)
        axs[idx, 0].grid(True, linestyle='--', alpha=0.6)

        # Plot vs Shift Y
        axs[idx, 1].plot(shift_y, metric_values, marker='s', linestyle='--', color='r')
        axs[idx, 1].set_xlabel('Shift Y (pixels)', fontsize=12)
        axs[idx, 1].set_ylabel(metric_name, fontsize=12)
        axs[idx, 1].set_title(f'{metric_name} vs. Shift Y', fontsize=14)
        axs[idx, 1].grid(True, linestyle='--', alpha=0.6)

    # Adjust layout
    plt.tight_layout()
    plt.show()


# Helper Functions
def extract_suffix(filename, prefix):
    """
    Extracts the numerical suffix from a filename.
    
    Parameters:
        filename (str): The filename (e.g., 'LR001.png').
        prefix (str): The prefix to remove (e.g., 'LR').
    
    Returns:
        str: The numerical suffix (e.g., '001').
    """
    return filename.replace(prefix, '').replace('.png', '')

def save_image(image, path, dtype=np.float32, easy_display=False):
    """
    Saves an image to the specified path with the given data type.
    
    Parameters:
        image (np.ndarray): The image array to save.
        path (str): The file path where the image will be saved.
        dtype (data-type, optional): Desired data type of the saved image.
            Defaults to np.float32 for continuous images and np.uint8 for binary masks.
        easy_display (bool): If True, upscales and scales the image for easier visualization.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    if easy_display:
        # Step 1: Normalize and contrast-stretch the image
        image = process_image_for_display(image, p2=1, p98=99)
        
        # Step 2: Upscale the image by a factor of 12
        # For RGB images, ensure the channel dimension is not scaled
        if image.ndim == 3:
            scale_factors = (12, 12, 1)  # Upscale height and width by 12, keep channels unchanged
        else:
            scale_factors = (12, 12)  # Upscale height and width by 12
        
        image = transform.rescale(
            image,
            scale=scale_factors,
            order=0,  # Lanczos interpolation for high-quality upscaling
            mode='reflect',
            anti_aliasing=True,
            preserve_range=True  # Preserve the intensity range [0, 1]
        )
        
        # Step 3: Scale intensities to [0, 255] for uint8 or [0, 65535] for uint16
        # Here, we'll map to [0, 255] and convert to uint8 for easy viewing
        image = exposure.rescale_intensity(
            image,
            in_range='image',
            out_range=(0, 255)
        ).astype(np.uint8)
        
        # Override dtype to uint8 when easy_display is True
        dtype = np.uint8
    
    if dtype == np.bool_:
        # Convert boolean mask to uint8 (0 and 255)
        io.imsave(path, (image.astype(np.uint8) * 255))
    else:
        if dtype in [np.float32, np.float64]:
            # For float images, scale to [0, 65535] and convert to uint16
            image_to_save = np.clip(image, 0, 65535).astype(np.uint16)
        elif dtype == np.uint8:
            # For uint8 images, ensure pixel values are within [0, 255]
            image_to_save = np.clip(image, 0, 255).astype(np.uint8)
        elif dtype == np.uint16:
            # For uint16 images, ensure pixel values are within [0, 65535]
            image_to_save = np.clip(image, 0, 65535).astype(np.uint16)
        else:
            # For other data types, save as-is
            image_to_save = image
        
        io.imsave(path, image_to_save)

def save_shift(shift, path):
    """
    Saves the shift vector to a JSON file.

    Parameters:
        shift (list, tuple, or np.ndarray): The shift vector [delta_y, delta_x].
        path (str): The file path where the JSON will be saved.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Ensure that shift elements are standard Python floats
    shift_python = [float(shift[0]), float(shift[1])]
    with open(path, 'w') as f:
        json.dump({'delta_y': shift_python[0], 'delta_x': shift_python[1]}, f)


def read_image(path_to_image):
    image = io.imread(path_to_image)
    #print(f"{path_to_image} Data Type: {image.dtype}")
    return image


def plot_metrics_vs_shifts_highlight(shift_x, shift_y, ssim, mse, pl):
    fig, axs = plt.subplots(3, 2, figsize=(14, 18))

    metrics = {
        'SSIM': ssim,
        'MSE': mse,
        'Perceptual Loss': pl
    }

    for idx, (metric_name, metric_values) in enumerate(metrics.items()):
        # Plot vs Shift X
        axs[idx, 0].plot(shift_x, metric_values, marker='o', linestyle='-', color='b', label=metric_name)
        axs[idx, 0].scatter(shift_x[-1], metric_values[-1], color='k', zorder=5, label='Final Value')
        axs[idx, 0].set_xlabel('Shift X (pixels)', fontsize=12)
        axs[idx, 0].set_ylabel(metric_name, fontsize=12)
        axs[idx, 0].set_title(f'{metric_name} vs. Shift X', fontsize=14)
        axs[idx, 0].legend(fontsize=12)
        axs[idx, 0].grid(True, linestyle='--', alpha=0.6)

        # Plot vs Shift Y
        axs[idx, 1].plot(shift_y, metric_values, marker='s', linestyle='--', color='r', label=metric_name)
        axs[idx, 1].scatter(shift_y[-1], metric_values[-1], color='k', zorder=5, label='Final Value')
        axs[idx, 1].set_xlabel('Shift Y (pixels)', fontsize=12)
        axs[idx, 1].set_ylabel(metric_name, fontsize=12)
        axs[idx, 1].set_title(f'{metric_name} vs. Shift Y', fontsize=14)
        axs[idx, 1].legend(fontsize=12)
        axs[idx, 1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

# Main Preprocessing Function
def preprocess_imgset(base_dir, feature_extractor):
    """
    Processes a single imgset directory: downscales HR image and mask,
    aligns LR images to the downscaled HR reference, and saves the results.
    
    Parameters:
        base_dir (str): Path to the imgset directory.
        feature_extractor: The model or feature extractor used for alignment.
        device: The computation device (e.g., 'cpu' or 'cuda').
    """
    # List all LR image files
    lr_files = sorted([f for f in os.listdir(base_dir) if f.startswith('LR') and f.endswith('.png')])
    
    # Corresponding QM mask files
    qm_files = sorted([f for f in os.listdir(base_dir) if f.startswith('QM') and f.endswith('.png')])
    
    # HR image and mask
    hr_file = os.path.join(base_dir, 'HR.png')
    sm_file = os.path.join(base_dir, 'SM.png')
    
    # Load HR image (assuming 16-bit) and convert to float
    hr = io.imread(hr_file).astype(np.float32)
    
    # Load HR mask
    sm_raw = io.imread(sm_file)
    
    # Interpret the mask
    if sm_raw.dtype == bool:
        sm_binary = sm_raw
    elif sm_raw.dtype == np.uint8:
        if np.array_equal(np.unique(sm_raw), [0, 255]):
            sm_binary = sm_raw == 255
        elif np.array_equal(np.unique(sm_raw), [0, 1]):
            sm_binary = sm_raw.astype(bool)
        else:
            sm_binary = (sm_raw > (sm_raw.max() / 2)).astype(bool)
    else:
        sm_binary = sm_raw.astype(bool)
    
    print(f"HR mask binary unique values after processing: {np.unique(sm_binary)}")
    
    # Load one LR image to get target size
    lr_sample = io.imread(os.path.join(base_dir, lr_files[0]))
    lr_height, lr_width = lr_sample.shape
    
    # Downscale HR image using Lanczos (order=4)
    hr_downscaled = transform.resize(
        hr,
        (lr_height, lr_width),
        order=4,  # Lanczos interpolation
        mode='reflect',
        anti_aliasing=True,
        preserve_range=True
    ).astype(np.float32)
    
    # Downscale HR mask using nearest-neighbor (order=0)
    sm_downscaled = transform.resize(
        sm_binary.astype(float),
        (lr_height, lr_width),
        order=0,  # Nearest-neighbor interpolation
        mode='edge',
        anti_aliasing=False,
        preserve_range=True
    ).astype(bool)
    
    # Verify downscaled mask
    print(f"Downscaled HR mask unique values: {np.unique(sm_downscaled)}")
    
    # Save the downscaled HR image and mask
    hr_downscaled_path = os.path.join(base_dir, 'hr_downscaled.png')
    sm_downscaled_path = os.path.join(base_dir, 'sm_downscaled.png')
    save_image(hr_downscaled, hr_downscaled_path, easy_display=False)
    save_image(sm_downscaled, sm_downscaled_path, dtype=np.bool_)
    
    # Iterate over LR and QM files together
    for lr_file, qm_file in zip(lr_files, qm_files):
        # Extract numerical suffix
        lr_suffix = extract_suffix(lr_file, 'LR')
        qm_suffix = extract_suffix(qm_file, 'QM')
        
        # Verify that the suffixes match
        if lr_suffix != qm_suffix:
            print(f"Warning: LR file {lr_file} and QM file {qm_file} have different suffixes.")
        
        suffix = lr_suffix  # or qm_suffix, since they should be same

        if suffix != "009":
            continue
        
        # Load LR image and QM mask
        lr_path = os.path.join(base_dir, lr_file)
        qm_path = os.path.join(base_dir, qm_file)
        
        lr = io.imread(lr_path).astype(np.float32)
        qm = io.imread(qm_path)
        qm_score = np.sum(qm)
        print(f"qm_score = {qm_score}, percent = {100*qm_score / qm.shape[0] / qm.shape[1]}, min/max qm = {np.min(qm), np.max(qm)}")
        
        # Interpret QM mask
        if qm.dtype == bool:
            qm_binary = qm
        elif qm.dtype == np.uint8:
            if np.array_equal(np.unique(qm), [0, 255]):
                qm_binary = qm == 255
            elif np.array_equal(np.unique(qm), [0, 1]):
                qm_binary = qm.astype(bool)
            else:
                qm_binary = (qm > (qm.max() / 2)).astype(bool)
        else:
            qm_binary = qm.astype(bool)

        #unmasked = np.ones(hr_downscaled.shape).astype(np.bool)
        # Align LR to the downscaled HR reference
        shifts_history, loss_history, mse_history, ssim_history = iterative_align_refinement_with_perceptual_loss(
            model=feature_extractor,
            ref_image=hr_downscaled,
            ref_mask=sm_downscaled,
            original_moving_image=lr,
            original_moving_mask=qm_binary,
            max_iterations=5,
            shift_dampening=1.0,  # Apply only half of the computed shift each iteration
            debug=False  # Enable visualization
        )
        #throw out the zeroth item in the list
        #for a in [shifts_history, loss_history, mse_history, ssim_history]:
        #    a.pop(0)
        shift_y = [float(shift[0]) for shift in shifts_history]
        shift_x = [float(shift[1]) for shift in shifts_history]

        for i in range(len(shifts_history)):
            print(f"shifts[{i}] = {shift_y[i], shift_x[i]}, pl[{i}] = {loss_history[i]}, mse[{i}] = {mse_history[i]}, ssim[{i}] = {ssim_history[i]}")


    

        # Plotting functions
        plot_metrics_vs_shifts(shift_x, shift_y, ssim_history, mse_history, loss_history)
        plot_metrics_vs_shifts_highlight(shift_x, shift_y, ssim_history, mse_history, loss_history)

        # Define output paths with preserved suffixes
        aligned_image_filename = f'aligned_LR{suffix}.png'
        aligned_mask_filename = f'aligned_QM{suffix}.png'
        shift_filename = f'shift_{suffix}.json'
        
        aligned_image_path = os.path.join(base_dir, aligned_image_filename)
        aligned_mask_path = os.path.join(base_dir, aligned_mask_filename)
        shift_path = os.path.join(base_dir, shift_filename)
        
        # Save aligned images and masks
        #save_image(aligned_image, aligned_image_path, easy_display=False)
        #save_image(aligned_mask, aligned_mask_path, dtype=np.bool_)
        
        # Save shift vectors
        #save_shift(total_shift, shift_path)
        
        # Print the results for verification
        print(f"\nProcessed {lr_file} and {qm_file}:")
        #print(f"Final cumulative shift: (y: {total_shift[0]:.5f}, x: {total_shift[1]:.5f})")
        #print(f"Shift History: {shifts_history}")
        #print(f"Perceptual Loss History: {loss_history}")





