# %%
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


x = np.arange(10)

# %%
signal.hilbert(x)


# %%
def hilbert(x, axis=-1):
    """
    Compute the analytic signal, using the Hilbert transform.

    The transformation is done along the last axis by default.

    Parameters
    ----------
    x : array_like
        Signal data.  Must be real.
    N : int, optional
        Number of Fourier components.  Default: ``x.shape[axis]``
    axis : int, optional
        Axis along which to do the transformation.  Default: -1.

    Returns
    -------
    xa : ndarray
        Analytic signal of `x`, of each 1-D array along `axis`
    """

    x = np.asarray(x)
    if np.iscomplexobj(x):
        raise ValueError("x must be real.")
    N = x.shape[axis]
    if N <= 0:
        raise ValueError("N must be positive.")

    Xf = np.fft.fft(x, N, axis=axis)
    h = np.zeros(N, dtype=Xf.dtype)
    if N % 2 == 0:
        h[0] = h[N // 2] = 1
        h[1 : N // 2] = 2
    else:
        h[0] = 1
        h[1 : (N + 1) // 2] = 2

    if x.ndim > 1:
        ind = [np.newaxis] * x.ndim
        ind[axis] = slice(None)
        h = h[tuple(ind)]
    x = np.fft.ifft(Xf * h, axis=axis)
    return x


def hilbert_r2c(x, axis=-1):
    """
    Compute the analytic signal, using the Hilbert transform.

    The transformation is done along the last axis by default.

    Parameters
    ----------
    x : array_like
        Signal data.  Must be real.
    N : int, optional
        Number of Fourier components.  Default: ``x.shape[axis]``
    axis : int, optional
        Axis along which to do the transformation.  Default: -1.

    Returns
    -------
    xa : ndarray
        Analytic signal of `x`, of each 1-D array along `axis`
    """

    x = np.asarray(x)
    if np.iscomplexobj(x):
        raise ValueError("x must be real.")
    N = x.shape[axis]
    if N <= 0:
        raise ValueError("N must be positive.")

    Xf = np.fft.rfft(x, N, axis=axis)
    x = np.fft.irfft(Xf * -1j, axis=axis)
    return x


print(hilbert_r2c(x))
print(hilbert(x))
print(np.allclose(hilbert_r2c(x), hilbert(x).imag))

# %%

# %%
np.fft.fft(x)

# %%
np.fft.rfft(x)

# %%
