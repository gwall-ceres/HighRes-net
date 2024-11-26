import torch
import torch.nn.functional as F
import numpy as np

def lanczos_kernel(dx, a=3, N=7, dtype=None, device=None):
    '''
    Generates 1D Lanczos kernels.
    Args:
        dx : tensor (batch_size * channels, 1), shifts
        a : int, number of lobes (default=3)
        N : int, kernel width (default=7)
    Returns:
        k: tensor (batch_size * channels, N)
    '''
    if not torch.is_tensor(dx):
        dx = torch.tensor(dx, dtype=dtype, device=device)

    if device is None:
        device = dx.device

    if dtype is None:
        dtype = dx.dtype

    # Create kernel positions
    n_lobes = (N - 1) // 2
    x = torch.linspace(-n_lobes, n_lobes, N, dtype=dtype, device=device).view(1, -1) - dx

    # Compute the pi * x values
    pi_x = np.pi * x

    # Avoid division by zero
    eps = 1e-6
    pi_x = torch.where(pi_x == 0, torch.tensor(eps, device=device, dtype=dtype), pi_x)

    # Compute Lanczos kernel
    sinc = torch.sin(pi_x) / pi_x
    sinc_a = torch.sin(pi_x / a) / (pi_x / a)
    k = sinc * sinc_a

    # Normalize the kernel
    k = k / k.sum(dim=1, keepdim=True)

    return k



def lanczos_shift(img, shift, p=3, a=3, N=7):
    '''
    Shifts images by convolving them with Lanczos kernels individually per channel.
    Args:
        img : tensor (batch_size, channels, height, width)
        shift : tensor (channels, 2), shifts per channel
        p : int, padding width (default=3)
        a : int, number of lobes in the Lanczos kernel (default=3)
        N : int, kernel width (should be odd, e.g., 7)
    Returns:
        I_s: tensor (batch_size, channels, height, width)
    '''
    batch_size, channels, height, width = img.shape
    I_s_list = []

    for c in range(channels):
        #print(f"\nProcessing channel {c+1}/{channels}")
        # Get the c-th channel across all batch images
        img_c = img[:, c:c+1, :, :]  # Shape: [batch_size, 1, H, W]
        shift_c = shift[c]  # Shape: [2]
        #print(f"img_c shape: {img_c.shape}")
        #print(f"shift_c: {shift_c}")

        # Apply padding
        padder = torch.nn.ReflectionPad2d(p)
        I_padded = padder(img_c)  # Shape: [batch_size, 1, H_padded, W_padded]
        #print(f"I_padded shape: {I_padded.shape}")

        # Generate kernels
        y_shift = shift_c[0:1].view(1, 1)  # Shape: [1, 1]
        x_shift = shift_c[1:2].view(1, 1)  # Shape: [1, 1]
        #print(f"y_shift: {y_shift}, x_shift: {x_shift}")

        k_y = lanczos_kernel(y_shift, a=a, N=N, dtype=img.dtype, device=img.device)  # Shape: [1, N]
        k_x = lanczos_kernel(x_shift, a=a, N=N, dtype=img.dtype, device=img.device)  # Shape: [1, N]
        #print(f"k_y shape: {k_y.shape}, k_x shape: {k_x.shape}")

        # Reshape kernels
        k_y = k_y.view(1, 1, N, 1)  # Shape: [1, 1, N, 1]
        k_x = k_x.view(1, 1, 1, N)  # Shape: [1, 1, 1, N]
        #print(f"k_y reshaped: {k_y.shape}, k_x reshaped: {k_x.shape}")

        # Perform convolution along y-axis
        I_s = F.conv2d(I_padded, weight=k_y, bias=None, groups=1, padding=(N//2, 0))
        #print(f"After y-conv, I_s shape: {I_s.shape}")

        # Perform convolution along x-axis
        I_s = F.conv2d(I_s, weight=k_x, bias=None, groups=1, padding=(0, N//2))
        #print(f"After x-conv, I_s shape: {I_s.shape}")

        # Remove padding
        I_s = I_s[..., p:-p, p:-p]
        #print(f"After removing padding, I_s shape: {I_s.shape}")

        I_s_list.append(I_s)

    # Stack the results along channel dimension
    I_s = torch.cat(I_s_list, dim=1)  # Shape: [batch_size, channels, H, W]
    #print(f"\nFinal shifted images shape: {I_s.shape}")

    return I_s