# %%
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np

x = np.random.random(11) * 2 - 1

x = np.array(
    [
        -0.999984,
        -0.736924,
        0.511211,
        -0.0826997,
        0.0655345,
        -0.562082,
        -0.905911,
        0.357729,
        0.358593,
        0.869386,
    ]
)

np.abs(signal.hilbert(x))
# %%
# print(x)
np_hilb = signal.hilbert(x)
print(np_hilb)


# %%
def hilbert_(x: np.ndarray):
    n = len(x)
    X = np.fft.fft(x)

    n_half = n // 2

    if n % 2 == 0:  # Even-length signal
        X[1:n_half] *= 2  # Scale positive frequencies
        X[n_half + 1 :] = 0  # Zero out negative frequencies
    else:  # Odd-length signal
        X[1 : n_half + 1] *= 2  # Scale positive frequencies
        X[n_half + 1 :] = 0  # Zero out negative frequencies

    x = np.fft.ifft(X)
    return x


def hilbert_rfft(x: np.ndarray) -> np.ndarray:
    n = len(x)
    X = np.fft.rfft(x)  # Compute the real-to-complex FFT

    # Create scaling for the Hilbert transform in the RFFT domain
    # Frequency scaling is different for DC and Nyquist components
    X[1:] *= 2  # Scale all positive frequencies (excluding DC)

    # Ensure correct reconstruction for even-length signals
    if n % 2 == 0:  # Handle Nyquist for even-length signals
        X[-1] *= 1  # Nyquist remains unchanged

    # Inverse RFFT to get the analytic signal
    x_analytic = np.fft.irfft(X, n=n)  # Reconstruct the time-domain signal

    return x_analytic


res = hilbert_rfft(x)

print(np.allclose(res, np.abs(np_hilb)))

plt.plot(x, label="x")
plt.plot(np_hilb.imag, label="np")
plt.plot(res, label="custom")
plt.legend()


# %%
res

# %%
np_hilb
