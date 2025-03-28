import torch
import torch.nn.functional as F

def binary_erosion(image, kernel_size=3):
    # Ensure the image is a PyTorch tensor and has the correct shape (1, 1, H, W) for 2D convolution
    # if len(image.shape) == 2:
    image = image.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    # elif len(image.shape) == 3:
    #     image = image.unsqueeze(1)  # Add channel dimension
    
    # Create a binary kernel
    kernel = torch.ones((1, 1, kernel_size, kernel_size), dtype=torch.float32)
    
    # Perform the convolution with padding to keep the same spatial dimensions
    # and use thresholding to simulate binary erosion
    padding = kernel_size // 2
    conv_result = F.conv2d(image.float(), kernel, padding=padding)

    # 核中有9个元素，因此我们检查卷积结果是否等于9来确定是否所有元素都匹配
    erosion = (conv_result == kernel.sum()).squeeze(0).squeeze(0)
    
    return erosion.bool()