from math import sqrt
import numpy as np

# Daubechies 4 Constant
c0 = (1+sqrt(3))/(4*sqrt(2))
c1 = (3+sqrt(3))/(4*sqrt(2))
c2 = (3-sqrt(3))/(4*sqrt(2))
c3 = (1-sqrt(3))/(4*sqrt(2))

def conv(x, h):
    """ Perform the convolution operation between two input signals. The output signal length
    is the sum of the lenght of both input signal minus 1."""
    return np.convolve(x, h)


def db4_dec(x, level=1):
    """ Perform the wavelet decomposition to signal x with Daubechies order 4 basis
    function as many as specified level"""

    # Decomposition coefficient for low pass and high pass
    # Daubechies 4 Constant
    lpk = [c0, c1, c2, c3]
    hpk = [c3, -c2, c1, -c0]

    result = [[]] * (level + 1)
    x_temp = x[:]
    for i in range(level):
        lp = conv(x_temp, lpk)
        hp = conv(x_temp, hpk)

        # Downsample both output by half
        index = np.arange(1, len(lp), 2)
        lp_ds = lp[index]
        hp_ds = hp[index]

        result[level - i] = hp_ds
        x_temp = lp_ds[:]

    result[0] = lp_ds
    return result


def db4_rec(signals, level):
    """ Perform reconstruction from a set of decomposed low pass and high pass signals as deep as specified level"""

    # Reconstruction coefficient
    # Daubechies 4 Constant
    lpk = [c3, c2, c1, c0]
    hpk = [-c0, c1, -c2, c3]

    cp_sig = signals[:]
    for i in range(level):
        lp = cp_sig[0]
        hp = cp_sig[1]

        # Verify new length
        if len(lp) > len(hp):
            length = 2 * len(hp)
        else:
            length = 2 * len(lp)

        # Upsampling by 2
        lpu = np.zeros(length + 1)
        hpu = np.zeros(length + 1)
        index = np.arange(0, length, 2)
        lpu[index] = lp
        hpu[index] = hp
        # Convolve with reconstruction coefficient
        lpc = conv(lpu, lpk)
        hpc = conv(hpu, hpk)

        # Truncate the convolved output by the length of filter kernel minus 1 at both end of the signal
        lpt = lpc[3:-3]
        hpt = hpc[3:-3]

        # Add both signals
        org = lpt + hpt

        if len(cp_sig) > 2:
            cp_sig = [org] + cp_sig[2:]
        else:
            cp_sig = [org]

    return cp_sig[0]


def calcEnergy(x):
    """ Calculate the energy of a signal which is the sum of square of each points in the signal."""
    return np.sum(x * x)


def bwr_dwt(raw, level=10):
    """ Perform the baseline wander removal process against signal raw. The output of this method is signal with correct baseline
        and its baseline """
    en1 = 0
    en2 = 0

    curlp = raw[:]
    num_dec = 0
    while num_dec < level:

        # Decompose 1 level
        [lp, hp] = db4_dec(curlp, 1)

        # Shift and calculate the energy of detail/high pass coefficient
        en0 = en1
        en1 = en2
        en2 = calcEnergy(hp)

        # Check if we are in the local minimum of energy function of high-pass signal
        if en0 > en1 and en1 < en2:
            break

        curlp = lp[:]
        num_dec = num_dec + 1

    # Reconstruct the baseline from this level low pass signal up to the original length
    base = curlp[:]
    for i in range(num_dec):
        base = db4_rec([base, np.zeros(len(base))], 1)

    # Correct the original signal by subtract it with its baseline
    ecg_out = raw - base[:len(raw)]
    # if num_dec == level:
    #     from matplotlib import pyplot as pl
    #     pl.title("level = {}".format(num_dec))
    #     pl.plot(raw + 2.0, label="raw")
    #     pl.plot(ecg_out - 2.0, label="out")
    #     pl.plot(base[:len(raw)], label="base")
    #     pl.legend(loc=4)
    #     pl.show()

    return ecg_out
