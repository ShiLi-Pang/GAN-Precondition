import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import warnings

warnings.filterwarnings('ignore')


def matrix3fd4(nx, ny, nz, dlx, dly, dlz, f, delta, c_vec):
    # imaginary unit
    ii = 1j

    # problem scale
    N = (nx - 2) * (ny - 2) * (nz - 2)

    # nonzero elements of impedance matrix
    if nx < 6 or ny < 6 or nz < 6:
        spn = N * N
    else:
        spn = 14 * (nx - 6) * (ny - 6) * (nz - 6) + \
              12 * 2 * ((nx - 6) * (ny - 6) + (ny - 6) * (nz - 6) + (nx - 6) * (nz - 6)) + \
              11 * 2 * ((nx - 6) * (ny - 6) + (ny - 6) * (nz - 6) + (nx - 6) + (nz - 6)) + \
              11 * 2 * 4 * ((nx - 6) + (ny - 6) + (nz - 6)) + \
              9 * 2 * 4 * ((nx - 6) + (ny - 6) + (nz - 6)) + \
              8 * ((13 - 6) + 3 * (13 - 5) + 3 * (13 - 4) + (13 - 3))
    #print(spn)
    # sparse store: vector space
    ai = np.zeros(spn, dtype=int)
    aj = np.zeros(spn, dtype=int)
    as_ = np.zeros(spn, dtype=complex)
    pa = 0

    # set angular frequency
    omega = 2 * np.pi * f

    # PML parameter: the ratio of reflection
    R = 1e-3

    # stencil coefficients of 4-order FD
    coeff2 = np.array([-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12])
    coeff1 = np.array([-1 / 12, 2 / 3, 0, -2 / 3, 1 / 12])

    # set values
    a2, a1, a0, a_1, a_2 = coeff2
    b2, b1, b0, b_1, b_2 = coeff1

    for n in range(N):
        # grid coordinate conversion: y-slice rule
        j = n // ((nx - 2) * (nz - 2))
        k = (n - (nx - 2) * (nz - 2) * j) // (nx - 2)
        i = n - (nx - 2) * k - (nx - 2) * (nz - 2) * j

        # set velocity
        c = c_vec[n]

        # set PML attenuation function
        if i < delta - 1:
            dx = -3 * c / (2 * delta * dlx) * np.log(R) * ((delta - i - 1) / delta) ** 2
            dxp = -3 * c / (delta * dlx) ** 2 * np.log(R) * (-(delta - i - 1) / delta)
        elif i > nx - delta - 2:
            dx = -3 * c / (2 * delta * dlx) * np.log(R) * ((i - nx + delta + 2) / delta) ** 2
            dxp = -3 * c / (delta * dlx) ** 2 * np.log(R) * ((i - nx + delta + 2) / delta)
        else:
            dx = 0
            dxp = 0

        if j < delta - 1:
            dy = -3 * c / (2 * delta * dly) * np.log(R) * ((delta - j - 1) / delta) ** 2
            dyp = -3 * c / (delta * dly) ** 2 * np.log(R) * (-(delta - j - 1) / delta)
        elif j > ny - delta - 2:
            dy = -3 * c / (2 * delta * dly) * np.log(R) * ((j - ny + delta + 2) / delta) ** 2
            dyp = -3 * c / (delta * dly) ** 2 * np.log(R) * ((j - ny + delta + 2) / delta)
        else:
            dy = 0
            dyp = 0

        if k < delta - 1:
            dz = -3 * c / (2 * delta * dlz) * np.log(R) * ((delta - k - 1) / delta) ** 2
            dzp = -3 * c / (delta * dlz) ** 2 * np.log(R) * (-(delta - k - 1) / delta)
        elif k > nz - delta - 2:
            dz = -3 * c / (2 * delta * dlz) * np.log(R) * ((k - nz + delta + 2) / delta) ** 2
            dzp = -3 * c / (delta * dlz) ** 2 * np.log(R) * ((k - nz + delta + 2) / delta)
        else:
            dz = 0
            dzp = 0

        tx = 1 - ii * dx / omega
        ty = 1 - ii * dy / omega
        tz = 1 - ii * dz / omega

        nt = n + 1

        # set the nonzeros of matrix
        # left1
        if i != 0:
            ai[pa] = nt
            aj[pa] = nt - 1
            as_[pa] = a_1 / (tx ** 2) + b_1 * ii * dxp * dlx / (omega * tx ** 3)
            pa += 1

        # left2
        if i > 1:
            ai[pa] = nt
            aj[pa] = nt - 2
            as_[pa] = a_2 / (tx ** 2) + b_2 * ii * dxp * dlx / (omega * tx ** 3)
            pa += 1

        # right1
        if i != nx - 3:
            ai[pa] = nt
            aj[pa] = nt + 1
            as_[pa] = a1 / (tx ** 2) + b1 * ii * dxp * dlx / (omega * tx ** 3)
            pa += 1

        # right2
        if i < nx - 4:
            ai[pa] = nt
            aj[pa] = nt + 2
            as_[pa] = a2 / (tx ** 2) + b2 * ii * dxp * dlx / (omega * tx ** 3)
            pa += 1

        # up1
        if k != 0:
            ai[pa] = nt
            aj[pa] = nt - (nx - 2)
            as_[pa] = a_1 / (tz ** 2) * (dlx ** 2 / dlz ** 2) + b_1 * ii * dzp / (omega * tz ** 3) * (dlx ** 2 / dlz)
            pa += 1

        # up2
        if k > 1:
            ai[pa] = nt
            aj[pa] = nt - 2 * (nx - 2)
            as_[pa] = a_2 / (tz ** 2) * (dlx ** 2 / dlz ** 2) + b_2 * ii * dzp / (omega * tz ** 3) * (dlx ** 2 / dlz)
            pa += 1

        # down1
        if k != nz - 3:
            ai[pa] = nt
            aj[pa] = nt + (nx - 2)
            as_[pa] = a1 / (tz ** 2) * (dlx ** 2 / dlz ** 2) + b1 * ii * dzp / (omega * tz ** 3) * (dlx ** 2 / dlz)
            pa += 1

        # down2
        if k < nz - 4:
            ai[pa] = nt
            aj[pa] = nt + 2 * (nx - 2)
            as_[pa] = a2 / (tz ** 2) * (dlx ** 2 / dlz ** 2) + b2 * ii * dzp / (omega * tz ** 3) * (dlx ** 2 / dlz)
            pa += 1

        # back1
        if j != 0:
            ai[pa] = nt
            aj[pa] = nt - (nx - 2) * (nz - 2)
            as_[pa] = a_1 / (ty ** 2) * (dlx ** 2 / dly ** 2) + b_1 * ii * dyp / (omega * ty ** 3) * (dlx ** 2 / dly)
            pa += 1

        # back2
        if j > 1:
            ai[pa] = nt
            aj[pa] = nt - 2 * (nx - 2) * (nz - 2)
            as_[pa] = a_2 / (ty ** 2) * (dlx ** 2 / dly ** 2) + b_2 * ii * dyp / (omega * ty ** 3) * (dlx ** 2 / dly)
            pa += 1

        # front1
        if j != ny - 3:
            ai[pa] = nt
            aj[pa] = nt + (nx - 2) * (nz - 2)
            as_[pa] = a1 / (ty ** 2) * (dlx ** 2 / dly ** 2) + b1 * ii * dyp / (omega * ty ** 3) * (dlx ** 2 / dly)
            pa += 1

        # front2
        if j < ny - 4:
            ai[pa] = nt
            aj[pa] = nt + 2 * (nx - 2) * (nz - 2)
            as_[pa] = a2 / (ty ** 2) * (dlx ** 2 / dly ** 2) + b2 * ii * dyp / (omega * ty ** 3) * (dlx ** 2 / dly)
            pa += 1

        # inner
        ai[pa] = nt
        aj[pa] = nt
        if ny == 3:  # since y-slice, this return to 2D case
            as_[pa] = ((omega * dlx / c) ** 2 +
                       a0 * (1 / tx ** 2 + 1 / tz ** 2 * (dlx ** 2 / dlz ** 2)) +
                       b0 * ii / omega * (dlx * dxp / tx ** 3 + (dlx ** 2 / dlz) * dzp / tz ** 3))
        else:
            as_[pa] = ((omega * dlx / c) ** 2 +
                       a0 * (1 / tx ** 2 + 1 / ty ** 2 * (dlx ** 2 / dly ** 2) + 1 / tz ** 2 * (dlx ** 2 / dlz ** 2)) +
                       b0 * ii / omega * (dlx * dxp / tx ** 3 + (dlx ** 2 / dly) * dyp / ty ** 3 + (
                            dlx ** 2 / dlz) * dzp / tz ** 3))
        pa += 1

    # extract potential zeros
    mask = as_ != 0
    ai = ai[:pa][mask[:pa]]
    aj = aj[:pa][mask[:pa]]
    as_ = as_[:pa][mask[:pa]]

    # create sparse matrix
    C = sp.csr_matrix((as_, (ai-1, aj-1)), shape=(N, N))

    return C


def source_3ofd(f, f0, N, h, c_vec, s_loc):
    # generate the right hand side term of source

    # initial unit
    ii = 1j

    # source location (count from 0)
    s0 = s_loc

    # source velocity
    sc = c_vec[s0][0]

    # Amplitude and phase
    t0 = 0.12
    Amp = 1e+5

    s = np.sqrt(2) * Amp / (np.pi * f0) * (f / f0) ** 2 * np.exp(-(f / f0) ** 2) * \
        sp.coo_matrix(([-h ** 2 / sc ** 2 * np.exp(-ii * 2 * np.pi * f * t0)], ([s0], [0])), shape=(N, 1))

    return s
