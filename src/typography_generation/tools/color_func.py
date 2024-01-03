from typing import Any, Tuple

import numpy as np

_illuminants = {
    "A": {
        "2": (1.098466069456375, 1, 0.3558228003436005),
        "10": (1.111420406956693, 1, 0.3519978321919493),
        "R": (1.098466069456375, 1, 0.3558228003436005),
    },
    "B": {
        "2": (0.9909274480248003, 1, 0.8531327322886154),
        "10": (0.9917777147717607, 1, 0.8434930535866175),
        "R": (0.9909274480248003, 1, 0.8531327322886154),
    },
    "C": {
        "2": (0.980705971659919, 1, 1.1822494939271255),
        "10": (0.9728569189782166, 1, 1.1614480488951577),
        "R": (0.980705971659919, 1, 1.1822494939271255),
    },
    "D50": {
        "2": (0.9642119944211994, 1, 0.8251882845188288),
        "10": (0.9672062750333777, 1, 0.8142801513128616),
        "R": (0.9639501491621826, 1, 0.8241280285499208),
    },
    "D55": {
        "2": (0.956797052643698, 1, 0.9214805860173273),
        "10": (0.9579665682254781, 1, 0.9092525159847462),
        "R": (0.9565317453467969, 1, 0.9202554587037198),
    },
    "D65": {
        "2": (0.95047, 1.0, 1.08883),  # This was: `lab_ref_white`
        "10": (0.94809667673716, 1, 1.0730513595166162),
        "R": (0.9532057125493769, 1, 1.0853843816469158),
    },
    "D75": {
        "2": (0.9497220898840717, 1, 1.226393520724154),
        "10": (0.9441713925645873, 1, 1.2064272211720228),
        "R": (0.9497220898840717, 1, 1.226393520724154),
    },
    "E": {"2": (1.0, 1.0, 1.0), "10": (1.0, 1.0, 1.0), "R": (1.0, 1.0, 1.0)},
}
xyz_from_rgb = np.array(
    [
        [0.412453, 0.357580, 0.180423],
        [0.212671, 0.715160, 0.072169],
        [0.019334, 0.119193, 0.950227],
    ]
)


def xyz_tristimulus_values(illuminant: str, observer: str, dtype: Any) -> np.array:
    """Get the CIE XYZ tristimulus values.

    Given an illuminant and observer, this function returns the CIE XYZ tristimulus
    values [2]_ scaled such that :math:`Y = 1`.

    Parameters
    ----------
    illuminant : {"A", "B", "C", "D50", "D55", "D65", "D75", "E"}
        The name of the illuminant (the function is NOT case sensitive).
    observer : {"2", "10", "R"}
        One of: 2-degree observer, 10-degree observer, or 'R' observer as in
        R function ``grDevices::convertColor`` [3]_.
    dtype: dtype, optional
        Output data type.

    Returns
    -------
    values : array
        Array with 3 elements :math:`X, Y, Z` containing the CIE XYZ tristimulus values
        of the given illuminant.

    Raises
    ------
    ValueError
        If either the illuminant or the observer angle are not supported or
        unknown.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Standard_illuminant#White_points_of_standard_illuminants
    .. [2] https://en.wikipedia.org/wiki/CIE_1931_color_space#Meaning_of_X,_Y_and_Z
    .. [3] https://www.rdocumentation.org/packages/grDevices/versions/3.6.2/topics/convertColor

    Notes
    -----
    The CIE XYZ tristimulus values are calculated from :math:`x, y` [1]_, using the
    formula

    .. math:: X = x / y

    .. math:: Y = 1

    .. math:: Z = (1 - x - y) / y

    The only exception is the illuminant "D65" with aperture angle 2° for
    backward-compatibility reasons.

    Examples
    --------
    Get the CIE XYZ tristimulus values for a "D65" illuminant for a 10 degree field of
    view

    >>> xyz_tristimulus_values(illuminant="D65", observer="10")
    array([0.94809668, 1.        , 1.07305136])
    """
    illuminant = illuminant.upper()
    observer = observer.upper()
    try:
        return np.asarray(_illuminants[illuminant][observer], dtype=dtype)
    except KeyError:
        raise ValueError(
            f"Unknown illuminant/observer combination "
            f"(`{illuminant}`, `{observer}`)"
        )


def xyz2lab(
    xyz: np.array, illuminant: str = "D65", observer: str = "2", channel_axis: int = -1
) -> np.array:
    """XYZ to CIE-LAB color space conversion.

    Parameters
    ----------
    xyz : (..., 3, ...) array_like
        The image in XYZ format. By default, the final dimension denotes
        channels.
    illuminant : {"A", "B", "C", "D50", "D55", "D65", "D75", "E"}, optional
        The name of the illuminant (the function is NOT case sensitive).
    observer : {"2", "10", "R"}, optional
        One of: 2-degree observer, 10-degree observer, or 'R' observer as in
        R function grDevices::convertColor.
    channel_axis : int, optional
        This parameter indicates which axis of the array corresponds to
        channels.

        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.

    Returns
    -------
    out : (..., 3, ...) ndarray
        The image in CIE-LAB format. Same dimensions as input.

    Raises
    ------
    ValueError
        If `xyz` is not at least 2-D with shape (..., 3, ...).
    ValueError
        If either the illuminant or the observer angle is unsupported or
        unknown.

    Notes
    -----
    By default Observer="2", Illuminant="D65". CIE XYZ tristimulus values
    x_ref=95.047, y_ref=100., z_ref=108.883. See function
    :func:`~.xyz_tristimulus_values` for a list of supported illuminants.

    References
    ----------
    .. [1] http://www.easyrgb.com/en/math.php
    .. [2] https://en.wikipedia.org/wiki/CIELAB_color_space

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.color import rgb2xyz, xyz2lab
    >>> img = data.astronaut()
    >>> img_xyz = rgb2xyz(img)
    >>> img_lab = xyz2lab(img_xyz)
    """
    arr = np.array(xyz)

    xyz_ref_white = xyz_tristimulus_values(
        illuminant=illuminant, observer=observer, dtype=arr.dtype
    )

    # scale by CIE XYZ tristimulus values of the reference white point
    arr = arr / xyz_ref_white

    # Nonlinear distortion and linear transformation
    mask = arr > 0.008856
    arr[mask] = np.cbrt(arr[mask])
    arr[~mask] = 7.787 * arr[~mask] + 16.0 / 116.0

    x, y, z = arr[..., 0], arr[..., 1], arr[..., 2]

    # Vector scaling
    L = (116.0 * y) - 16.0
    a = 500.0 * (x - y)
    b = 200.0 * (y - z)

    return np.concatenate([x[..., np.newaxis] for x in [L, a, b]], axis=-1)


def rgb2xyz(rgb: np.array, channel_axis: int = -1) -> np.array:
    """RGB to XYZ color space conversion.

    Parameters
    ----------
    rgb : (..., 3, ...) array_like
        The image in RGB format. By default, the final dimension denotes
        channels.
    channel_axis : int, optional
        This parameter indicates which axis of the array corresponds to
        channels.

        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.

    Returns
    -------
    out : (..., 3, ...) ndarray
        The image in XYZ format. Same dimensions as input.

    Raises
    ------
    ValueError
        If `rgb` is not at least 2-D with shape (..., 3, ...).

    Notes
    -----
    The CIE XYZ color space is derived from the CIE RGB color space. Note
    however that this function converts from sRGB.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/CIE_1931_color_space

    Examples
    --------
    >>> from skimage import data
    >>> img = data.astronaut()
    >>> img_xyz = rgb2xyz(img)
    """
    # Follow the algorithm from http://www.easyrgb.com/index.php
    # except we don't multiply/divide by 100 in the conversion
    arr = np.array(rgb)
    mask = arr > 0.04045
    arr[mask] = np.power((arr[mask] + 0.055) / 1.055, 2.4)
    arr[~mask] /= 12.92
    return arr @ xyz_from_rgb.T.astype(arr.dtype)


def rgb2lab(
    rgb: np.array,
    illuminant: str = "D65",
    observer: str = "2",
    channel_axis: int = -1,
) -> np.array:
    """Conversion from the sRGB color space (IEC 61966-2-1:1999)
    to the CIE Lab colorspace under the given illuminant and observer.

    Parameters
    ----------
    rgb : (..., 3, ...) array_like
        The image in RGB format. By default, the final dimension denotes
        channels.
    illuminant : {"A", "B", "C", "D50", "D55", "D65", "D75", "E"}, optional
        The name of the illuminant (the function is NOT case sensitive).
    observer : {"2", "10", "R"}, optional
        The aperture angle of the observer.
    channel_axis : int, optional
        This parameter indicates which axis of the array corresponds to
        channels.

        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.

    Returns
    -------
    out : (..., 3, ...) ndarray
        The image in Lab format. Same dimensions as input.

    Raises
    ------
    ValueError
        If `rgb` is not at least 2-D with shape (..., 3, ...).

    Notes
    -----
    RGB is a device-dependent color space so, if you use this function, be
    sure that the image you are analyzing has been mapped to the sRGB color
    space.

    This function uses rgb2xyz and xyz2lab.
    By default Observer="2", Illuminant="D65". CIE XYZ tristimulus values
    x_ref=95.047, y_ref=100., z_ref=108.883. See function
    :func:`~.xyz_tristimulus_values` for a list of supported illuminants.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Standard_illuminant
    """
    return xyz2lab(rgb2xyz(rgb), illuminant, observer)


def _cart2polar_2pi(x: np.array, y: np.array) -> Tuple[np.array, np.array]:
    """convert cartesian coordinates to polar (uses non-standard theta range!)

    NON-STANDARD RANGE! Maps to ``(0, 2*pi)`` rather than usual ``(-pi, +pi)``
    """
    r, t = np.hypot(x, y), np.arctan2(y, x)
    t += np.where(t < 0.0, 2 * np.pi, 0)
    return r, t


def _float_inputs(
    lab1: np.array, lab2: np.array, allow_float32: bool = True
) -> Tuple[np.array, np.array]:
    lab1 = np.asarray(lab1)
    lab2 = np.asarray(lab2)
    float_dtype = np.float64
    lab1 = lab1.astype(float_dtype, copy=False)
    lab2 = lab2.astype(float_dtype, copy=False)
    return lab1, lab2


def deltaE_ciede2000(
    lab1: np.array,
    lab2: np.array,
    kL: int = 1,
    kC: int = 1,
    kH: int = 1,
    channel_axis: int = -1,
) -> np.array:
    """Color difference as given by the CIEDE 2000 standard.

    CIEDE 2000 is a major revision of CIDE94.  The perceptual calibration is
    largely based on experience with automotive paint on smooth surfaces.

    Parameters
    ----------
    lab1 : array_like
        reference color (Lab colorspace)
    lab2 : array_like
        comparison color (Lab colorspace)
    kL : float (range), optional
        lightness scale factor, 1 for "acceptably close"; 2 for "imperceptible"
        see deltaE_cmc
    kC : float (range), optional
        chroma scale factor, usually 1
    kH : float (range), optional
        hue scale factor, usually 1
    channel_axis : int, optional
        This parameter indicates which axis of the arrays corresponds to
        channels.

        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.

    Returns
    -------
    deltaE : array_like
        The distance between `lab1` and `lab2`

    Notes
    -----
    CIEDE 2000 assumes parametric weighting factors for the lightness, chroma,
    and hue (`kL`, `kC`, `kH` respectively).  These default to 1.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Color_difference
    .. [2] http://www.ece.rochester.edu/~gsharma/ciede2000/ciede2000noteCRNA.pdf
           :DOI:`10.1364/AO.33.008069`
    .. [3] M. Melgosa, J. Quesada, and E. Hita, "Uniformity of some recent
           color metrics tested with an accurate color-difference tolerance
           dataset," Appl. Opt. 33, 8069-8077 (1994).
    """
    lab1, lab2 = _float_inputs(lab1, lab2, allow_float32=True)

    channel_axis = channel_axis % lab1.ndim
    unroll = False
    if lab1.ndim == 1 and lab2.ndim == 1:
        unroll = True
        if lab1.ndim == 1:
            lab1 = lab1[None, :]
        if lab2.ndim == 1:
            lab2 = lab2[None, :]
        channel_axis += 1
    L1, a1, b1 = np.moveaxis(lab1, source=channel_axis, destination=0)[:3]
    L2, a2, b2 = np.moveaxis(lab2, source=channel_axis, destination=0)[:3]

    # distort `a` based on average chroma
    # then convert to lch coordinates from distorted `a`
    # all subsequence calculations are in the new coordinates
    # (often denoted "prime" in the literature)
    Cbar = 0.5 * (np.hypot(a1, b1) + np.hypot(a2, b2))
    c7 = Cbar**7
    G = 0.5 * (1 - np.sqrt(c7 / (c7 + 25**7)))
    scale = 1 + G
    C1, h1 = _cart2polar_2pi(a1 * scale, b1)
    C2, h2 = _cart2polar_2pi(a2 * scale, b2)
    # recall that c, h are polar coordinates.  c==r, h==theta

    # cide2000 has four terms to delta_e:
    # 1) Luminance term
    # 2) Hue term
    # 3) Chroma term
    # 4) hue Rotation term

    # lightness term
    Lbar = 0.5 * (L1 + L2)
    tmp = (Lbar - 50) ** 2
    SL = 1 + 0.015 * tmp / np.sqrt(20 + tmp)
    L_term = (L2 - L1) / (kL * SL)

    # chroma term
    Cbar = 0.5 * (C1 + C2)  # new coordinates
    SC = 1 + 0.045 * Cbar
    C_term = (C2 - C1) / (kC * SC)

    # hue term
    h_diff = h2 - h1
    h_sum = h1 + h2
    CC = C1 * C2

    dH = h_diff.copy()
    dH[h_diff > np.pi] -= 2 * np.pi
    dH[h_diff < -np.pi] += 2 * np.pi
    dH[CC == 0.0] = 0.0  # if r == 0, dtheta == 0
    dH_term = 2 * np.sqrt(CC) * np.sin(dH / 2)

    Hbar = h_sum.copy()
    mask = np.logical_and(CC != 0.0, np.abs(h_diff) > np.pi)
    Hbar[mask * (h_sum < 2 * np.pi)] += 2 * np.pi
    Hbar[mask * (h_sum >= 2 * np.pi)] -= 2 * np.pi
    Hbar[CC == 0.0] *= 2
    Hbar *= 0.5

    T = (
        1
        - 0.17 * np.cos(Hbar - np.deg2rad(30))
        + 0.24 * np.cos(2 * Hbar)
        + 0.32 * np.cos(3 * Hbar + np.deg2rad(6))
        - 0.20 * np.cos(4 * Hbar - np.deg2rad(63))
    )
    SH = 1 + 0.015 * Cbar * T

    H_term = dH_term / (kH * SH)

    # hue rotation
    c7 = Cbar**7
    Rc = 2 * np.sqrt(c7 / (c7 + 25**7))
    dtheta = np.deg2rad(30) * np.exp(-(((np.rad2deg(Hbar) - 275) / 25) ** 2))
    R_term = -np.sin(2 * dtheta) * Rc * C_term * H_term

    # put it all together
    dE2 = L_term**2
    dE2 += C_term**2
    dE2 += H_term**2
    dE2 += R_term
    ans = np.sqrt(np.maximum(dE2, 0))
    if unroll:
        ans = ans[0]
    return ans
