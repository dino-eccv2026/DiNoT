
import torch
import torch.nn.functional as F
import math

def get_gaussian_kernel(kernel_size: int, sigma: float, channels: int = 3) -> torch.Tensor:
    """
    Create a Gaussian kernel for blurring.
    """
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-D gaussian kernel
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    return gaussian_kernel

def differentiable_blur(image: torch.Tensor, kernel_size: int = 21, sigma: float = 5.0) -> torch.Tensor:
    """
    Apply Gaussian blur to an image tensor in a differentiable way.
    
    Args:
        image: (B, C, H, W) tensor
        kernel_size: Size of Gaussian kernel (odd)
        sigma: Standard deviation of Gaussian
        
    Returns:
        Blurred image tensor (B, C, H, W)
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
        
    channels = image.shape[1]
    kernel = get_gaussian_kernel(kernel_size, sigma, channels).to(device=image.device, dtype=image.dtype)
    
    # Pad image to maintain size
    padding = kernel_size // 2
    blurred = F.conv2d(image, kernel, padding=padding, groups=channels)
    
    return blurred

def apply_blur_mask(image: torch.Tensor, mask: torch.Tensor, blur_radius: int = 21) -> torch.Tensor:
    """
    Apply blur to background regions defined by (1-mask).
    
    Args:
        image: (B, C, H, W) tensor
        mask: (B, 1, H, W) mask tensor (1 = foreground/keep, 0 = background/blur)
        blur_radius: Kernel size for blur
    
    Returns:
        (B, C, H, W) tensor: image * mask + blurred * (1-mask)
    """
    blurred = differentiable_blur(image, kernel_size=blur_radius)
    return image * mask + blurred * (1 - mask)
