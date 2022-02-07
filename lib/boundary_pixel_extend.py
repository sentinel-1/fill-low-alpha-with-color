import torch
from typing import Tuple



class FillLowAlphaWithColor(torch.nn.Module):
    """Fill low alpha areas of image with color from high alpha areas based on the provided alpha mask.
    Image is expected to have [C, H, W] shape, where C means channels (C=3 in case of RGB).
    Alpha mask is expected to have [H, W] shape.

    Args:
        alpha_high       int:  Colors are taken from area where alpha value is greather or equal to the `alpha_high`.
        alpha_low        int:  Colors are changed in area where the value of alpha is in (0, `alpha_low`) interval (exclusive).
        gpu             bool:  If True then uses GPU if the GPU is available, otherwise will work without GPU.
        pixel_chunk_size int:  How many pixels to process at one step, higher value will be faster but requires more memory.
        kernel_size      int:  Size of erosion kernel used to reduce the area from which are the fill colors taken, value must be a positive odd number,
                               if `kernel_size` value is 1 then kernel is not used, if `kernel_size` > 1, then kernel of the given size is used.
    """

    def __init__(self,
                 alpha_high: int = 255,
                 alpha_low: int = 255,
                 gpu: bool = False,
                 pixel_chunk_size: int = 72000,
                 kernel_size: int = 1
                ):
        super().__init__()
        self.alpha_high = alpha_high
        self.alpha_low = alpha_low
        self.gpu = gpu
        self.pixel_chunk_size = pixel_chunk_size
        assert kernel_size%2==1, "Kernel size should be odd."
        self.kernel_size = kernel_size
        
        if self.gpu and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{torch.cuda.current_device()}')
        else:
            self.device = torch.device("cpu")
        

    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert image.shape[-2:] == mask.shape, (f"Image size {image.shape[-2]}x{image.shape[-1]} and"
                                                f" mask size {mask.shape[0]}x{mask.shape[1]} do not match!")
        
        
        # Fill low alpha areas with colors from high alpha areas:
        raise Exception("DIY")
        
        
        return image, mask
    
    
    def __repr__(self):
        return self.__class__.__name__ + '(alpha_high={}, alpha_low={}, gpu={}, pixel_chunk_size={}, kernel_size={})'.format(
            self.alpha_high, self.alpha_low, self.gpu, self.pixel_chunk_size, self.kernel_size
        )

