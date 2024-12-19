import numpy as np
import torch
import torch.nn.functional as F
from skimage import transform, exposure
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import normalized_mutual_information as nmi

# Normalize image to zero mean and unit std_dev
def normalize_masked_array(masked_array):
    epsilon = 1e-8
    mean = np.mean(masked_array)
    std = np.std(masked_array)
    if std == 0:
        raise ValueError("Standard deviation is zero. Cannot normalize an array with constant values.")
    normalized_array = (masked_array - mean) / (std + epsilon)
    return normalized_array

def _validate_and_convert_masks(ref_mask, mov_mask):
    """Helper function to validate and convert masks to boolean arrays"""
    # Convert to boolean if not already
    if ref_mask.dtype != bool:
        ref_mask = ref_mask > 0.5
    if mov_mask.dtype != bool:
        mov_mask = mov_mask > 0.5
    return ref_mask, mov_mask

def histogram_contrast_stretch(image):
    """
    Apply histogram contrast stretching to an image.

    Args:
        image (np.ndarray): Input image in [0, 1] range.

    Returns:
        np.ndarray: Contrast-stretched image in [0, 1] range.
    """
    # Apply contrast stretching
    p2, p98 = np.percentile(image, (2, 98))
    img_rescaled = exposure.rescale_intensity(image, in_range=(p2, p98), out_range=(0, 1))
    return img_rescaled

def compute_perceptual_loss(ref_image, mov_image,
                            ref_mask, mov_mask, model, **metric_kwargs):
    # Input validation
    assert ref_image.shape[:2] == mov_image.shape[:2], "Images must have same dimensions"
    assert ref_mask.shape == mov_mask.shape, "Masks must have same dimensions"
    
    # combine the two masks in order to only consider valid pixels
    combined_mask = ref_mask.astype(float) * mov_mask.astype(float)
    combined_mask = combined_mask > 0
    #valid_ratio = np.sum(combined_mask).astype(float) / combined_mask.shape[0] * combined_mask.shape[1]

    # Apply masks to input images
    ref_masked = histogram_contrast_stretch(ref_image)
    mov_masked = histogram_contrast_stretch(mov_image)

    ref_masked[~combined_mask] = 0#ref_image * combined_mask
    mov_masked[~combined_mask] = 0#mov_image * combined_mask

    ref_tensor = model.convert_grayscale_to_input_tensor(ref_masked).to(model.hardware)
    mov_tensor = model.convert_grayscale_to_input_tensor(mov_masked).to(model.hardware)
    # Extract features using the model (which handles grayscale conversion)
    with torch.no_grad():
        ref_features = model(ref_tensor)
        mov_features = model(mov_tensor)
    
    # Ensure consistent feature ordering
    if isinstance(ref_features, dict):
        feature_names = list(ref_features.keys())#sorted(ref_features.keys())
        #print(feature_names)
        ref_features = [ref_features[name] for name in feature_names]
        mov_features = [mov_features[name] for name in feature_names]
    
    # Initialize loss tracking
    total_loss = 0.0
    diff_features = {}
    # Weights for VGG19 layers ['1', '6', '11', '20', '29']
    # Layer 0: First conv layer - highest detail
    # Layer 5: End of first conv block
    # Layer 10: End of second conv block
    # Layer 19: Deep in third conv block
    # Layer 28: Deep features
    layer_weights = [1.0, 0.8, 0.6, 0.4, 0.2]  # Decreasing weights for deeper layers
    #if the number of features is different than the number of layers, keep track of the weights that were applied
    applied_weights = []
    
    # Process each layer with proper mask handling
    for idx, (ref_feat, mov_feat) in enumerate(zip(ref_features, mov_features)):
        _, C, Hf, Wf = ref_feat.shape
              
        # Compute masked L1 loss directly on feature values
        layer_loss = F.l1_loss(ref_feat, mov_feat, reduction='mean')
        diff_features[f"{feature_names[idx]}_loss"] = layer_loss.item()
        # Store feature differences for visualization
        l1_diff = torch.abs(ref_feat - mov_feat)
        l1_diff_summed = l1_diff.sum(dim=1).squeeze(0).detach().cpu().numpy()
        #store the feature differences so we can plot them later
        diff_features[f"{feature_names[idx]}_diff"] = l1_diff_summed
        #store the mask so we can plot it later
        diff_features[f"{feature_names[idx]}_mask"] = combined_mask
        
        
        # Apply layer weight
        weight = layer_weights[idx] if idx < len(layer_weights) else layer_weights[-1]
        applied_weights.append(weight)
        total_loss += weight * layer_loss.item()

    # Normalize by sum of weights
    total_loss /= sum(applied_weights)
    
    return total_loss, diff_features

#need to change these names
def compute_masked_ncc(ref_image, mov_image, ref_mask, mov_mask, **metric_kwargs):
    """
    Compute the Weighted Normalized Cross-Correlation (NCC) between two images.

    Parameters:
        ref_image (np.ndarray): Reference image array.
        mov_image (np.ndarray): Moving/template image array.
        ref_mask (np.ndarray): Reference mask array with values between 0 and 1.
        mov_mask (np.ndarray): Moving/template mask array with values between 0 and 1.
        **metric_kwargs: Additional keyword arguments (unused)

    Returns:
        float: Weighted NCC value.
    """
    # Compute combined weights
    weights = ref_mask.astype(float) * mov_mask.astype(float)

    # Rest of function remains the same...
    weight_sum = np.sum(weights)
    if weight_sum == 0:
        return np.nan

    # Compute weighted means
    mu_ref = np.sum(ref_image * weights) / weight_sum
    mu_mov = np.sum(mov_image * weights) / weight_sum

    # Compute weighted standard deviations
    sigma_ref = np.sqrt(np.sum(weights * (ref_image - mu_ref) ** 2) / weight_sum)
    sigma_mov = np.sqrt(np.sum(weights * (mov_image - mu_mov) ** 2) / weight_sum)

    if sigma_ref == 0 or sigma_mov == 0:
        return np.nan

    # Compute weighted covariance
    covariance = np.sum(weights * (ref_image - mu_ref) * (mov_image - mu_mov)) / weight_sum

    # Compute weighted NCC
    ncc = covariance / (sigma_ref * sigma_mov)

    return ncc

def compute_mse(ref_image, mov_image, ref_mask, mov_mask, use_masks=True, normalize=True, **metric_kwargs):
    """
    Compute Mean Squared Error between two images with optional masking and normalization.
    
    Parameters:
        ref_image (np.ndarray): Reference image
        mov_image (np.ndarray): Moving/template image
        ref_mask (np.ndarray): Reference mask
        mov_mask (np.ndarray): Moving/template mask
        use_masks (bool): Whether to apply masks
        normalize (bool): Whether to normalize images before comparison
        **metric_kwargs: Additional keyword arguments (unused)
        
    Returns:
        float: Mean squared error between the images
    """
    if use_masks:
        # Convert masks to boolean
        ref_mask, mov_mask = _validate_and_convert_masks(ref_mask, mov_mask)
        combined_mask = np.logical_and(ref_mask, mov_mask)
        # Apply the combined mask to both images
        masked_ref = ref_image[combined_mask]
        masked_mov = mov_image[combined_mask]
    else:
        masked_ref = ref_image
        masked_mov = mov_image

    # Normalize both masked images
    if normalize:
        normalized_ref = normalize_masked_array(masked_ref)
        normalized_mov = normalize_masked_array(masked_mov)
        return np.mean((normalized_ref - normalized_mov) ** 2)
    else:
        return np.mean((masked_ref - masked_mov) ** 2)

def compute_ml1e(ref_image, mov_image, ref_mask, mov_mask, use_masks=True, normalize=True, **metric_kwargs):
    """
    Compute Mean L1 Error between two images with optional masking and normalization.
    
    Parameters:
        ref_image (np.ndarray): Reference image
        mov_image (np.ndarray): Moving/template image
        ref_mask (np.ndarray): Reference mask
        mov_mask (np.ndarray): Moving/template mask
        use_masks (bool): Whether to apply masks
        normalize (bool): Whether to normalize images before comparison
        **metric_kwargs: Additional keyword arguments (unused)
        
    Returns:
        float: Mean L1 error between the images
    """
    if use_masks:
        # Convert masks to boolean
        ref_mask, mov_mask = _validate_and_convert_masks(ref_mask, mov_mask)
        combined_mask = np.logical_and(ref_mask, mov_mask)
        # Apply the combined mask to both images
        masked_ref = ref_image[combined_mask]
        masked_mov = mov_image[combined_mask]
    else:
        masked_ref = ref_image
        masked_mov = mov_image

    # Normalize both masked images
    if normalize:
        normalized_ref = normalize_masked_array(masked_ref)
        normalized_mov = normalize_masked_array(masked_mov)
        return np.mean(np.abs(normalized_ref - normalized_mov))
    else:
        return np.mean(np.abs(masked_ref - masked_mov))

def compute_ssim(ref_image, mov_image, ref_mask, mov_mask, use_masks=True, **metric_kwargs):
    """
    Compute the Structural Similarity Index Measure (SSIM) between two images,
    optionally considering only the overlapping valid regions defined by masks.

    Parameters:
    - ref_image (np.ndarray): Reference image array, shape (H, W) or (H, W, C).
    - mov_image (np.ndarray): Moving image array, shape (H, W) or (H, W, C).
    - ref_mask (np.ndarray): Boolean mask for the reference image, shape (H, W).
    - mov_mask (np.ndarray): Boolean mask for the moving image, shape (H, W).
    - use_masks (bool): If True, only consider pixels where both masks are True.

    Returns:
    - ssim_value (float): The computed SSIM.
    """
    # Ensure both images have the same spatial dimensions
    if ref_image.shape[:2] != mov_image.shape[:2]:
        raise ValueError("Reference and moving images must have the same height and width.")
    if ref_mask.shape != mov_mask.shape:
        raise ValueError("Reference and moving masks must have the same height and width.")

    # Convert masks to boolean if use_masks is True
    if use_masks:
        ref_mask, mov_mask = _validate_and_convert_masks(ref_mask, mov_mask)
        combined_mask = np.logical_and(ref_mask, mov_mask)  # Shape: (H, W)
        if not np.any(combined_mask):
            raise ValueError("No overlapping valid pixels found between the masks.")
        # Apply the combined mask to both images
        masked_ref = ref_image.copy()
        masked_mov = mov_image.copy()
        masked_ref[~combined_mask] = 0
        masked_mov[~combined_mask] = 0
    else:
        masked_ref = ref_image
        masked_mov = mov_image

    # Compute data range from the images
    data_min = min(masked_ref.min(), masked_mov.min())
    data_max = max(masked_ref.max(), masked_mov.max())
    data_range = data_max - data_min

    if data_range <= 0:
        raise ValueError("Data range must be positive.")

    # Handle grayscale and multichannel images
    if masked_ref.ndim == 2:
        # Grayscale images
        ssim_value = ssim(masked_ref, masked_mov, data_range=data_range)
    elif masked_ref.ndim == 3 and masked_ref.shape[2] in [1, 3, 4]:
        # Multichannel images (e.g., RGB)
        ssim_value = ssim(masked_ref, masked_mov, data_range=data_range, multichannel=True)
    else:
        raise ValueError("Input images must have shape (H, W), (H, W, 1), (H, W, 3), or (H, W, 4)")

    return ssim_value

def compute_mi(ref_image, mov_image, ref_mask, mov_mask, use_masks=True, 
                             bins=100, **metric_kwargs):
    """
    Compute Mutual Information between two images using scikit-image.
    
    Parameters:
        ref_image (np.ndarray): Reference image
        mov_image (np.ndarray): Moving/template image
        ref_mask (np.ndarray): Reference mask
        mov_mask (np.ndarray): Moving/template mask
        use_masks (bool): Whether to apply masks
        bins (int): Number of bins for histogram computation
        **metric_kwargs: Additional keyword arguments (unused)
        
    Returns:
        float: Mutual information between the images
    """
    if use_masks:
        # Convert masks to boolean
        ref_mask, mov_mask = _validate_and_convert_masks(ref_mask, mov_mask)
        combined_mask = np.logical_and(ref_mask, mov_mask)
        # Apply the combined mask to both images
        masked_ref = ref_image.copy()
        masked_mov = mov_image.copy()
        masked_ref[~combined_mask] = 0
        masked_mov[~combined_mask] = 0
    else:
        masked_ref = ref_image
        masked_mov = mov_image
    
    return nmi(masked_ref, masked_mov, bins=bins)

