
def calc_fan(weight_shape):
    """
    Compute the fan-in and fan-out for a weight matrix/volume.

    Parameters
    ----------
    weight_shape : tuple
        The dimensions of the weight matrix/volume. The final 2 entries must be
        `in_ch`, `out_ch`.

    Returns
    -------
    fan_in : int
        The number of input units in the weight tensor
    fan_out : int
        The number of output units in the weight tensor
    """
    if len(weight_shape) == 2:
        fan_in, fan_out = weight_shape
    elif len(weight_shape) in [3, 4]:
        in_ch, out_ch = weight_shape[-2:]
        kernel_size = np.prod(weight_shape[:-2])
        fan_in, fan_out = in_ch * kernel_size, out_ch * kernel_size
    else:
        raise ValueError(f"Unrecognized weight dimension: {weight_shape}")
    return fan_in, fan_out
