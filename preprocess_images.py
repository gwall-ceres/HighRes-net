#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform
from skimage.registration import phase_cross_correlation
from skimage import exposure
from scipy.ndimage import shift as ndi_shift
from skimage.transform import resize
from torchvision import transforms, models
import torch
import torch.nn.functional as F
import os
import random
import json
import time


# In[2]:

'''
import torch
import torchvision.models as models
from torchvision.models import VGG19_Weights
# Define the VGGFeatureExtractor with updated weights parameter
class VGGFeatureExtractor(torch.nn.Module):
    def __init__(self, layers=['0', '5', '10', '19', '28']):
        super(VGGFeatureExtractor, self).__init__()
        # Use the updated 'weights' parameter instead of 'pretrained'
        self.vgg = models.vgg19(weights=VGG19_Weights.DEFAULT).features[:int(layers[-1])+1]
        self.layers = layers

    def forward(self, x):
        outputs = {}
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.layers:
                outputs[name] = x
        return outputs

# Initialize the feature extractor
feature_extractor = VGGFeatureExtractor(layers=['0', '5', '10', '19', '28']).to('cuda' if torch.cuda.is_available() else 'cpu')
feature_extractor.eval()
# Disable gradient computations
for param in feature_extractor.parameters():
    param.requires_grad = False
'''


def compute_perceptual_loss(model, ref_image, mov_image, ref_mask, mov_mask, device='cpu', debug=False):
    """
    Compute the perceptual loss between two images given their masks.
    Optionally visualize selected activations and masks for debugging.
    
    Parameters:
    - model: A PyTorch model (e.g., VGGFeatureExtractor) returning feature maps from multiple layers.
             The model should return a list or dictionary of feature maps when called.
    - ref_image: NumPy array, shape (H, W) or (H, W, C), reference image.
    - mov_image: NumPy array, shape (H, W) or (H, W, C), moving image to align.
    - ref_mask: NumPy boolean array, shape (H, W), valid pixels for ref_image.
    - mov_mask: NumPy boolean array, shape (H, W), valid pixels for mov_image.
    - device: 'cpu' or 'cuda', the device to run computations on.
    - debug: Boolean, if True, visualize activations and masks.
    
    Returns:
    - loss_value: float, the perceptual loss computed only over valid overlapping pixels.
    """
    
    # Ensure both images have the same spatial dimensions
    assert ref_image.shape[:2] == mov_image.shape[:2], "Reference and moving images must have the same height and width."
    assert ref_mask.shape == mov_mask.shape, "Reference and moving masks must have the same height and width."
    
    # Combine masks to focus on overlapping valid regions
    combined_mask = ref_mask & mov_mask  # Shape: (H, W)
    
    # Function to convert grayscale to RGB by replicating channels
    def ensure_rgb(img):
        if img.ndim == 2:
            # Grayscale image, replicate channels to make (H, W, 3)
            return np.stack([img, img, img], axis=-1)
        elif img.ndim == 3 and img.shape[2] == 1:
            # Single-channel image with shape (H, W, 1), replicate to (H, W, 3)
            return np.concatenate([img, img, img], axis=2)
        elif img.ndim == 3 and img.shape[2] == 3:
            # Already RGB
            return img
        else:
            raise ValueError("Input image must have shape (H, W), (H, W, 1), or (H, W, 3)")
    
    # Convert images to RGB if necessary
    ref_image_rgb = ensure_rgb(ref_image)
    mov_image_rgb = ensure_rgb(mov_image)
    
    # Zero out invalid pixels according to the combined mask
    ref_image_rgb[~combined_mask] = 0
    mov_image_rgb[~combined_mask] = 0
    
    # Define the preprocessing pipeline: PIL conversion, tensor conversion, normalization
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet means
                             std=[0.229, 0.224, 0.225])   # ImageNet stds
    ])
    torch.cuda.synchronize()
    start = time.time()
    # Apply preprocessing to both images
    ref_tensor = preprocess(ref_image_rgb).unsqueeze(0).to(device)  # Shape: [1, 3, H, W]
    mov_tensor = preprocess(mov_image_rgb).unsqueeze(0).to(device)  # Shape: [1, 3, H, W]
    
    # Convert combined mask to a torch tensor
    combined_mask_tensor = torch.from_numpy(combined_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)  # Shape: [1, 1, H, W]
    
    # Pass both images through the model to extract feature maps
    with torch.no_grad():
        ref_features = model(ref_tensor)  # Assume this returns a list or dict of feature maps
        mov_features = model(mov_tensor)  # Same structure as ref_features
    
    # If the model returns a dictionary, ensure consistent ordering
    if isinstance(ref_features, dict):
        feature_names = sorted(ref_features.keys())
        ref_features = [ref_features[name] for name in feature_names]
        mov_features = [mov_features[name] for name in feature_names]
    elif isinstance(ref_features, list) or isinstance(ref_features, tuple):
        pass  # Already a list or tuple
    else:
        raise TypeError("Model output must be a list, tuple, or dictionary of feature maps.")
    
    # Initialize total loss
    total_loss = 0.0
    
    # For debugging: prepare plots
    if debug:
        num_layers = len(ref_features)
        fig, axes = plt.subplots(num_layers, 5, figsize=(20, 4 * num_layers))
        if num_layers == 1:
            axes = [axes]  # Ensure axes is iterable
    else:
        fig = None
        axes = None
    
    # Iterate over each pair of feature maps
    for idx, (ref_feat, mov_feat) in enumerate(zip(ref_features, mov_features)):
        layer_name = f"Layer_{idx+1}"
        # Get spatial dimensions of the feature map
        _, C, Hf, Wf = ref_feat.shape
        
        # Resize the combined mask to match the feature map size using nearest neighbor
        mask_resized = F.interpolate(combined_mask_tensor, size=(Hf, Wf), mode='bilinear', antialias=True)  # Shape: [1, 1, Hf, Wf]
        
        # Expand mask to match the number of channels in the feature map
        mask_expanded = mask_resized.expand_as(ref_feat)  # Shape: [1, C, Hf, Wf]
        
        # Apply the mask to both feature maps
        ref_feat_masked = ref_feat * mask_expanded
        mov_feat_masked = mov_feat * mask_expanded
        
        # Compute the Mean Squared Error (MSE) between the masked feature maps
        mse_loss = F.mse_loss(ref_feat_masked, mov_feat_masked, reduction='sum') / torch.sum(mask_expanded)
        
        # Accumulate the loss
        total_loss += mse_loss.item()
        
        if debug:
            # Select 2 random channels
            selected_channels = random.sample(range(C), 2) if C >=2 else [0]
            
            for i, channel in enumerate(selected_channels):
                # Get the reference activation
                ref_activation = ref_feat[0, channel, :, :].cpu().numpy()
                ref_activation_norm = (ref_activation - ref_activation.min()) / (ref_activation.max() - ref_activation.min() + 1e-8)
                
                # Get the moving activation
                mov_activation = mov_feat[0, channel, :, :].cpu().numpy()
                mov_activation_norm = (mov_activation - mov_activation.min()) / (mov_activation.max() - mov_activation.min() + 1e-8)
                
                # Get the masked activation
                ref_feat_masked_np = ref_feat_masked[0, channel, :, :].cpu().numpy()
                mov_feat_masked_np = mov_feat_masked[0, channel, :, :].cpu().numpy()
                ref_feat_masked_norm = (ref_feat_masked_np - ref_feat_masked_np.min()) / (ref_feat_masked_np.max() - ref_feat_masked_np.min() + 1e-8)
                mov_feat_masked_norm = (mov_feat_masked_np - mov_feat_masked_np.min()) / (mov_feat_masked_np.max() - mov_feat_masked_np.min() + 1e-8)
                
                # Get the resized mask
                mask_np = mask_resized[0, 0, :, :].cpu().numpy()
                
                # Plot original activation
                axes[idx][0].imshow(ref_activation_norm, cmap='viridis')
                axes[idx][0].set_title(f"{layer_name} Ref Channel {channel}")
                axes[idx][0].axis('off')
                
                axes[idx][1].imshow(mov_activation_norm, cmap='viridis')
                axes[idx][1].set_title(f"{layer_name} Mov Channel {channel}")
                axes[idx][1].axis('off')
                
                # Plot masked activations
                axes[idx][2].imshow(ref_feat_masked_norm, cmap='viridis')
                axes[idx][2].set_title(f"{layer_name} Ref Masked Channel {channel}")
                axes[idx][2].axis('off')
                
                axes[idx][3].imshow(mov_feat_masked_norm, cmap='viridis')
                axes[idx][3].set_title(f"{layer_name} Mov Masked Channel {channel}")
                axes[idx][3].axis('off')
                
                # Plot the resized mask
                axes[idx][4].imshow(mask_np, cmap='gray',vmin=0, vmax=1)
                axes[idx][4].set_title(f"{layer_name} Resized Mask")
                axes[idx][4].axis('off')
    
    if debug and fig is not None:
        plt.tight_layout()
        plt.show()
    torch.cuda.synchronize()
    end = time.time()
    print(f"***total time to compute the perceptual loss = {end - start}***")
    return total_loss


def contrast_stretch_8bit(array):
    """
    Perform contrast stretching on a NumPy array to map its values to the 0-255 range.
    """
    # Convert to float to prevent overflow/underflow
    array = array.astype(float)

    # Compute minimum and maximum pixel values
    min_val = np.min(array)
    max_val = np.max(array)

    print(f"contrast_stretch_8bit min = {min_val}, max = {max_val}")

    # Avoid division by zero
    if max_val - min_val == 0:
        return np.zeros_like(array, dtype=np.uint8)

    # Perform contrast stretching
    stretched = (array - min_val) / (max_val - min_val) * 255.0

    # Clip values to the 0-255 range and convert to uint8
    stretched = np.clip(stretched, 0, 255).astype(np.uint8)

    return stretched

def min_max_scale(image):
    min_val = np.min(image)
    max_val = np.max(image)
    scaled_image = (image - min_val) / (max_val - min_val + 1e-8)  # Avoid division by zero
    return scaled_image
    
def compute_combined_perceptual_mse_loss(model, ref_image, mov_image, ref_mask, mov_mask, device='cpu', debug=False, alpha=0.5, beta=0.5):
    """
    Compute a combined perceptual and MSE loss between two images given their masks.
    
    Parameters:
        model: Pretrained feature extractor model.
        ref_image: NumPy array, reference image.
        mov_image: NumPy array, moving image to align.
        ref_mask: NumPy boolean array, mask for reference image.
        mov_mask: NumPy boolean array, mask for moving image.
        device: Computation device.
        debug: Boolean, if True, visualize activations and masks.
        alpha: Weight for perceptual loss.
        beta: Weight for MSE loss.
    
    Returns:
        combined_loss: Weighted sum of perceptual and MSE losses.
    """
    # Compute Perceptual Loss
    perceptual_loss = compute_perceptual_loss(model, ref_image, mov_image, ref_mask, mov_mask, device, debug)
    
    # Normalize images for MSE computation
    ref_normalized = min_max_scale(ref_image)
    mov_normalized = min_max_scale(mov_image)
    
    # Convert to tensors
    ref_tensor = torch.from_numpy(ref_normalized).float().to(device)
    mov_tensor = torch.from_numpy(mov_normalized).float().to(device)
    
    # Apply combined mask
    combined_mask = ref_mask & mov_mask
    combined_mask_tensor = torch.from_numpy(combined_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    
    # Mask the tensors
    ref_tensor_masked = ref_tensor * combined_mask_tensor
    mov_tensor_masked = mov_tensor * combined_mask_tensor
    
    # Compute MSE Loss
    mse_loss = F.mse_loss(ref_tensor_masked, mov_tensor_masked, reduction='mean')
    
    # Compute Combined Loss
    combined_loss = alpha * perceptual_loss + beta * mse_loss
    print(f"combined_loss {alpha * perceptual_loss} + {beta * mse_loss}")
    
    return combined_loss


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


# In[33]:


def visualize_alignment2(ref_image, aligned_image, ref_mask, aligned_mask, shift_vector, suffix=''):
    """
    Visualize the alignment between the reference and aligned images along with their masks.
    
    Parameters:
        ref_image (np.ndarray): Reference image array (grayscale or RGB), already processed for display.
        aligned_image (np.ndarray): Aligned image array (grayscale or RGB), already processed for display.
        ref_mask (np.ndarray): Binary mask for the reference image.
        aligned_mask (np.ndarray): Binary mask for the aligned image.
        shift_vector (tuple or list): Total shift applied (delta_y, delta_x).
        suffix (str): Optional identifier for labeling.
    """
    # Ensure masks are binary and have the same shape as images
    if ref_mask.ndim == 2:
        ref_mask_display = ref_mask
    else:
        ref_mask_display = ref_mask[:, :, 0]  # Assuming single-channel masks
    
    if aligned_mask.ndim == 2:
        aligned_mask_display = aligned_mask
    else:
        aligned_mask_display = aligned_mask[:, :, 0]
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Display Reference Image
    axes[0, 0].imshow(ref_image, cmap='gray' if ref_image.ndim == 2 else None)
    axes[0, 0].set_title('Reference Image')
    axes[0, 0].axis('off')
    
    # Display Aligned Image
    axes[0, 1].imshow(aligned_image, cmap='gray' if aligned_image.ndim == 2 else None)
    axes[0, 1].set_title('Aligned Image')
    axes[0, 1].axis('off')
    
    # Display Difference Image
    if ref_image.ndim == 3:
        # For RGB images, compute the mean difference across channels
        difference = np.mean(np.abs(ref_image - aligned_image), axis=2)
    else:
        difference = np.abs(ref_image - aligned_image)
    difference = process_image_for_display(difference)
    axes[0, 2].imshow(difference, cmap='gray')
    axes[0, 2].set_title('Difference Image')
    axes[0, 2].axis('off')
    
    # Overlay Reference Mask on Reference Image
    axes[1, 0].imshow(ref_image, cmap='gray' if ref_image.ndim == 2 else None)
    axes[1, 0].imshow(ref_mask_display, cmap='jet', alpha=0.3)
    axes[1, 0].set_title('Reference Image with Mask')
    axes[1, 0].axis('off')
    
    # Overlay Aligned Mask on Aligned Image
    axes[1, 1].imshow(aligned_image, cmap='gray' if aligned_image.ndim == 2 else None)
    axes[1, 1].imshow(aligned_mask_display, cmap='jet', alpha=0.3)
    axes[1, 1].set_title('Aligned Image with Mask')
    axes[1, 1].axis('off')
    
    # Display Shift Information
    axes[1, 2].axis('off')  # No image, just text
    text = f"""
    Alignment Summary
    -----------------
    Shift Applied:
      Δy: {shift_vector[0]:.4f}
      Δx: {shift_vector[1]:.4f}
    Image Set: {suffix}
    """
    axes[1, 2].text(0.5, 0.5, text, fontsize=14, ha='center', va='center', wrap=True)
    axes[1, 2].set_title('Shift Information')
    
    plt.tight_layout()
    plt.show()


# In[35]:


def visualize_registration(ref_image, initial_moving_image, aligned_image, ref_mask=None, mov_mask=None):
    """
    Visualize the registration results by displaying reference, initial moving, and aligned images.
    Also displays difference images and overlays for qualitative assessment.
    
    Parameters:
    - ref_image: NumPy array, reference image.
    - initial_moving_image: NumPy array, moving image before alignment.
    - aligned_image: NumPy array, moving image after alignment.
    - ref_mask: NumPy boolean array, mask for reference image (optional).
    - mov_mask: NumPy boolean array, mask for moving image (optional).
    """

    # Compute difference images
    difference_initial = np.abs(min_max_scale(ref_image) - min_max_scale(initial_moving_image))
    difference_aligned = np.abs(min_max_scale(ref_image) - min_max_scale(aligned_image))
    ratio = np.sum(difference_aligned) / np.sum(difference_initial)
    difference_initial = transform.rescale(difference_initial, (3, 3), order=4, mode='reflect', anti_aliasing=True)
    difference_aligned = transform.rescale(difference_aligned, (3, 3), order=4, mode='reflect', anti_aliasing=True)
    print(f"diff unaligned {np.sum(difference_initial)}, diff aligned {np.sum(difference_aligned)} ratio = {ratio}")
    
    # Normalize difference images for better visualization
    difference_initial_norm = (difference_initial - difference_initial.min()) / (difference_initial.max() - difference_initial.min() + 1e-8)
    difference_aligned_norm = (difference_aligned - difference_aligned.min()) / (difference_aligned.max() - difference_aligned.min() + 1e-8)
    #print(np.sum(difference_initial_norm), np.sum(difference_aligned_norm))
    print(f"difference_initial_norm {np.sum(difference_initial_norm)}, difference_aligned_norm {np.sum(difference_aligned_norm)} ratio = {np.sum(difference_aligned_norm) / np.sum(difference_initial_norm)}")
    if ratio < 1.0:
        return             
    # Create overlays
    overlay_initial = np.stack([ref_image, initial_moving_image, np.zeros_like(ref_image)], axis=-1).astype(np.float32)
    overlay_aligned = np.stack([ref_image, aligned_image, np.zeros_like(ref_image)], axis=-1).astype(np.float32)
    
    # Normalize overlays to [0, 1]
    overlay_initial_norm = (overlay_initial - overlay_initial.min()) / (overlay_initial.max() - overlay_initial.min() + 1e-8)
    overlay_aligned_norm = (overlay_aligned - overlay_aligned.min()) / (overlay_aligned.max() - overlay_aligned.min() + 1e-8)
    
    # Plotting
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Reference Image
    axes[0, 0].imshow(ref_image, cmap='gray')
    axes[0, 0].set_title('Reference Image')
    axes[0, 0].axis('off')
    
    # Initial Moving Image
    axes[0, 1].imshow(initial_moving_image, cmap='gray')
    axes[0, 1].set_title('Initial Moving Image')
    axes[0, 1].axis('off')
    
    # Aligned Moving Image
    axes[0, 2].imshow(aligned_image, cmap='gray')
    axes[0, 2].set_title('Aligned Moving Image')
    axes[0, 2].axis('off')
    
    # Difference Images
    axes[0, 3].imshow(difference_initial_norm, cmap='hot')
    axes[0, 3].set_title('Difference: Initial')
    axes[0, 3].axis('off')
    
    # Overlays
    axes[1, 0].imshow(overlay_initial_norm)
    axes[1, 0].set_title('Overlay: Reference & Initial')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(overlay_aligned_norm)
    axes[1, 1].set_title('Overlay: Reference & Aligned')
    axes[1, 1].axis('off')
    
    # Difference after alignment
    axes[1, 2].imshow(difference_aligned_norm, cmap='hot')
    axes[1, 2].set_title('Difference: Aligned')
    axes[1, 2].axis('off')
    
    # Optionally, display masks if provided
    if ref_mask is not None and mov_mask is not None:
        combined_mask = ref_mask & mov_mask
        axes[1, 3].imshow(combined_mask, cmap='gray')
        axes[1, 3].set_title('Combined Mask')
        axes[1, 3].axis('off')
    else:
        # Hide the subplot if masks are not provided
        axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.show()


# In[36]:


def iterative_align_refinement_with_perceptual_loss(
    model,
    ref_image,
    ref_mask,
    original_moving_image,
    original_moving_mask,
    max_iterations=10,
    convergence_threshold=0.01,
    upsample_factor=100,
    device='cpu',
    shift_dampening=0.5,
    debug=False
):
    """
    Iteratively align original_moving_image to reference_image using phase_cross_correlation.
    Applies accumulated shifts directly to the original image to preserve fidelity.
    Computes and records perceptual loss at each step.
    Optionally visualizes activations and masks for debugging.
    
    Parameters:
    - model: An instance of VGGFeatureExtractor or similar, pre-trained CNN for feature extraction.
    - ref_image: NumPy array, float32, shape (H, W) or (H, W, C), reference image.
    - ref_mask: NumPy boolean array, shape (H, W), mask for reference image.
    - original_moving_image: NumPy array, float32, shape (H, W) or (H, W, C), moving image to align.
    - original_moving_mask: NumPy boolean array, shape (H, W), mask for moving image.
    - max_iterations: int, maximum number of iterations.
    - convergence_threshold: float, shift magnitude below which to stop.
    - upsample_factor: int, for subpixel accuracy in phase_cross_correlation.
    - device: 'cpu' or 'cuda', device for computations.
    - shift_dampening: float between 0 and 1, fraction of shift to apply each iteration to prevent overshooting.
    - debug: Boolean, if True, visualize activations and masks.
    
    Returns:
    - aligned_image_final: NumPy array, float32, aligned moving image.
    - aligned_mask_final: NumPy boolean array, aligned mask.
    - total_shift: tuple, (shift_y, shift_x).
    - shifts_history: list of tuples, [(shift_y1, shift_x1), ...].
    - loss_history: list of floats, [loss1, loss2, ...].
    - initial_moving_image: NumPy array, original moving image before alignment.
    """
    
    total_shift_y = 0.0
    total_shift_x = 0.0
    shifts_history = []
    loss_history = []
    
    initial_moving_image = original_moving_image.copy()
    initial_moving_mask = original_moving_mask.copy()
    
    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1} ---")
        
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
            mode='reflect',
            order=0
        )
        #shifted_mask = shifted_mask > 0.5  # Re-binarize the mask

        # Compute perceptual loss between reference and shifted moving image
        loss = compute_combined_perceptual_mse_loss(
            model=model,
            ref_image=ref_image,
            mov_image=shifted_image,
            ref_mask=ref_mask,
            mov_mask=shifted_mask>0.5,
            device=device,
            debug=False#debug and iteration == 0  # Only visualize in the first iteration
        )
        loss_history.append(loss)
        
        print(f"Iteration {iteration + 1}: Perceptual Loss = {loss:.6f}")

        # Compute shift between reference and currently aligned moving image
        
        ref_masked = ref_image * (ref_mask.astype(ref_image.dtype) * 0.5 + 0.5)
        mov_masked = shifted_image * (shifted_mask.astype(shifted_image.dtype) * 0.5 + 0.5)

        #ref_masked = ref_image * ref_mask.astype(ref_image.dtype)
        #mov_masked = shifted_image * shifted_mask.astype(shifted_image.dtype)

        #ref_masked = ref_image 
        #mov_masked = shifted_image
        
        shift_yx, error, diffphase = phase_cross_correlation(
            ref_masked,
            mov_masked,
            upsample_factor=2**(iteration+3)
        )
        delta_y, delta_x = shift_yx
        shifts_history.append((delta_y, delta_x))
        
        print(f"Iteration {iteration + 1}: Computed Shift = (y: {delta_y:.5f}, x: {delta_x:.5f})")
        
        # Apply shift dampening to prevent overshooting
        dampened_shift_y = delta_y * shift_dampening
        dampened_shift_x = delta_x * shift_dampening
        
        print(f"Iteration {iteration + 1}: Applied Shift = (y: {dampened_shift_y:.5f}, x: {dampened_shift_x:.5f})")
        
        # Accumulate shifts
        total_shift_y += dampened_shift_y
        total_shift_x += dampened_shift_x
        
        # Compute perceptual loss after applying the shift
        shifted_image_new = ndi_shift(
            original_moving_image,
            shift=(total_shift_y, total_shift_x),
            mode='reflect',
            order=3
        )
        shifted_mask_new = ndi_shift(
            original_moving_mask.astype(float),
            shift=(total_shift_y, total_shift_x),
            mode='reflect',
            order=0
        )
        shifted_mask_new = shifted_mask_new > 0.5  # Re-binarize the mask
        
        loss_new = compute_combined_perceptual_mse_loss(
            model=model,
            ref_image=ref_image,
            mov_image=shifted_image_new,
            ref_mask=ref_mask,
            mov_mask=shifted_mask_new,
            device=device,
            debug=False  # Disable visualization for subsequent iterations
        )
        loss_history.append(loss_new)
        
        print(f"Iteration {iteration + 1}: Updated Perceptual Loss = {loss_new:.6f}")

        if False:
            ref_image1 = process_image_for_display(ref_image)
            aligned_image1 = process_image_for_display(shifted_image_new)
        else:
            ref_image1 = ref_image
            aligned_image1 = shifted_image_new
            
        aligned_mask = shifted_mask_new  
        
        # Define the shift vector (replace with your actual shift values)
        shift_vector = (total_shift_y, total_shift_x)  # e.g., (0.1234, -0.5678)
        
        # Visualize the alignment
        #ref_ref_image1 = ref_image1 * (ref_mask.astype(ref_image.dtype) * 0.5 + 0.5)
        #aligned_image1 = aligned_image1 * (shifted_mask_new.astype(shifted_image.dtype) * 0.5 + 0.5)
        #visualize_alignment(ref_image1, aligned_image1, ref_mask, aligned_mask, shift_vector, suffix='000')
        visualize_registration(ref_image1, initial_moving_image, aligned_image1, ref_mask=None, mov_mask=None)
        
        # Check for convergence based on shift magnitude
        '''
        if abs(dampened_shift_y) < convergence_threshold and abs(dampened_shift_x) < convergence_threshold:
            print("Converged based on shift threshold.")
            break
        '''
        
    
    # After iterations, apply the total accumulated shift to the original moving image
    aligned_image_final = ndi_shift(
        original_moving_image,
        shift=(total_shift_y, total_shift_x),
        mode='reflect',
        order=3
    )
    aligned_mask_final = ndi_shift(
        original_moving_mask.astype(float),
        shift=(total_shift_y, total_shift_x),
        mode='reflect',
        order=0
    )
    aligned_mask_final = aligned_mask_final > 0.5  # Re-binarize the mask
    
    total_shift = (total_shift_y, total_shift_x)
    
    return aligned_image_final, aligned_mask_final, total_shift, shifts_history, loss_history, initial_moving_image


# In[59]:


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

# Main Preprocessing Function
def preprocess_imgset(base_dir, feature_extractor, device):
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

        #if suffix != "009":
        #    continue
        
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
        aligned_image, aligned_mask, total_shift, shifts_history, loss_history, initial_moving_image = iterative_align_refinement_with_perceptual_loss(
            model=feature_extractor,
            ref_image=hr_downscaled,
            ref_mask=sm_downscaled,
            original_moving_image=lr,
            original_moving_mask=qm_binary,
            max_iterations=10,
            convergence_threshold=0.01,
            upsample_factor=100,
            device=device,
            shift_dampening=1.0,  # Apply only half of the computed shift each iteration
            debug=False  # Enable visualization
        )
        
        # Define output paths with preserved suffixes
        aligned_image_filename = f'aligned_LR{suffix}.png'
        aligned_mask_filename = f'aligned_QM{suffix}.png'
        shift_filename = f'shift_{suffix}.json'
        
        aligned_image_path = os.path.join(base_dir, aligned_image_filename)
        aligned_mask_path = os.path.join(base_dir, aligned_mask_filename)
        shift_path = os.path.join(base_dir, shift_filename)
        
        # Save aligned images and masks
        save_image(aligned_image, aligned_image_path, easy_display=False)
        save_image(aligned_mask, aligned_mask_path, dtype=np.bool_)
        
        # Save shift vectors
        save_shift(total_shift, shift_path)
        
        # Print the results for verification
        print(f"\nProcessed {lr_file} and {qm_file}:")
        print(f"Final cumulative shift: (y: {total_shift[0]:.5f}, x: {total_shift[1]:.5f})")
        print(f"Shift History: {shifts_history}")
        print(f"Perceptual Loss History: {loss_history}")





