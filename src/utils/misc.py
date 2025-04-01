import numpy as np

def make_grid(height, width, stride=1):
    """
    Create a grid of points with the given stride.
    
    Args:
        height (int): Grid height
        width (int): Grid width
        stride (int): Spacing between points
        
    Returns:
        np.ndarray: Grid points of shape ((height//stride)*(width//stride), 2)
    """
    x = np.arange(0, width, stride)
    y = np.arange(0, height, stride)
    X, Y = np.meshgrid(x, y)
    return np.stack([X.flatten(), Y.flatten()], axis=1)