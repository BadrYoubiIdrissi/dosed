import numpy as np


def clip(max_value):
    """returns a function to clip data"""

    def clipper(signal_data, max_value=max_value):
        """returns input signal clipped between +/- max_value.
        """
        return np.clip(signal_data, -max_value, max_value)

    return clipper


def clip_and_normalize(min_value, max_value):
    """returns a function to clip and normalize data"""

    def clipper(x, min_value=min_value, max_value=max_value):
        """returns input signal clipped between min_value and max_value
        and then normalized between -0.5 and 0.5.
        """
        x = np.clip(x, min_value, max_value)
        x = ((x - min_value) /
             (max_value - min_value)) - 0.5
        return x

    return clipper

def max_min_normalize():
    """returns a function to clip and normalize data"""

    def clipper(x):
        """returns input signal clipped between min_value and max_value
        and then normalized between -0.5 and 0.5.
        """
        max_val = x.max()
        min_val = x.min()
    
        x = ((x - min_val) /
             (max_val - min_val)) - 0.5
        return x

    return clipper

def quantile_normalize(min_quantile, max_quantile):
    """returns a function to clip and normalize data"""

    def clipper(x, min_quantile=min_quantile, max_quantile=max_quantile):
        """returns input signal clipped between min_value and max_value
        and then normalized between -0.5 and 0.5.
        """
        max_val = np.quantile(x, max_quantile)
        min_val = np.quantile(x, min_quantile)
        # x = np.clip(x, min_val, max_val)
        x = (x - min_val)/(max_val - min_val) - 0.5
        return x

    return clipper

def mask_clip_and_normalize(min_value, max_value, mask_value):
    """returns a function to clip and normalize data"""

    def clipper(x, min_value=min_value, max_value=max_value,
                mask_value=mask_value):
        """returns input signal clipped between min_value and max_value
        and then normalized between -0.5 and 0.5.
        """
        mask = np.ma.masked_equal(x, mask_value)
        x = np.clip(x, min_value, max_value)
        x = ((x - min_value) /
             (max_value - min_value)) - 0.5
        x[mask.mask] = mask_value
        return x

    return clipper
