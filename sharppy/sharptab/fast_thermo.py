'''
Numba-accelerated scalar thermodynamic kernels.

These mirror the scalar code paths in `sharppy.sharptab.thermo` exactly
(same constants, same algorithms) but are compiled with numba so that the
per-level Newton iterations in `params.parcelx`/`params.cape`/`params.dcape`
(which call these functions hundreds of times per parcel, and are run for
several parcels per sounding) no longer pay Python/NumPy call overhead on
every single level.

numba is a hard dependency (see setup.py/environment.yml) -- there is no
pure-Python fallback here. The scalar math is duplicated once, cleanly, in
compiled form; keeping a second hand-maintained pure-Python copy "just in
case" isn't worth the ongoing maintenance cost of two implementations that
must always agree bit-for-bit.
'''

import numpy as np
from numba import njit

ZEROCNK = 273.15
ROCP = 0.28571426
eps = 0.62197
c1 = 0.0498646455; c2 = 2.4082965; c3 = 7.07475
c4 = 38.9114; c5 = 0.0915; c6 = 1.2035


@njit(cache=True, fastmath=True, nogil=True)
def wobf(t):
    t = t - 20.0
    if t <= 0.0:
        npol = 1. + t * (-8.841660499999999e-3 + t * (1.4714143e-4 + t * (-9.671989000000001e-7 + t * (-3.2607217e-8 + t * (-3.8598073e-10)))))
        npol = 15.13 / (npol ** 4)
        return npol
    else:
        ppol = t * (4.9618922e-07 + t * (-6.1059365e-09 + t * (3.9401551e-11 + t * (-1.2588129e-13 + t * (1.6688280e-16)))))
        ppol = 1 + t * (3.6182989e-03 + t * (-1.3603273e-05 + ppol))
        ppol = (29.93 / (ppol ** 4)) + (0.96 * t) - 14.8
        return ppol


@njit(cache=True, fastmath=True, nogil=True)
def theta(p, t, p2=1000.):
    return ((t + ZEROCNK) * (p2 / p) ** ROCP) - ZEROCNK


@njit(cache=True, fastmath=True, nogil=True)
def ctok(t):
    return t + ZEROCNK


@njit(cache=True, fastmath=True, nogil=True)
def vappres(t):
    pol = t * (1.1112018e-17 + (t * -3.0994571e-20))
    pol = t * (2.1874425e-13 + (t * (-1.789232e-15 + pol)))
    pol = t * (4.3884180e-09 + (t * (-2.988388e-11 + pol)))
    pol = t * (7.8736169e-05 + (t * (-6.111796e-07 + pol)))
    pol = 0.99999683 + (t * (-9.082695e-03 + pol))
    return 6.1078 / pol ** 8


@njit(cache=True, fastmath=True, nogil=True)
def mixratio(p, t):
    x = 0.02 * (t - 12.5 + (7500. / p))
    wfw = 1. + (0.0000045 * p) + (0.0014 * x * x)
    fwesw = wfw * vappres(t)
    return 621.97 * (fwesw / (p - fwesw))


@njit(cache=True, fastmath=True, nogil=True)
def virtemp(p, t, td):
    tk = t + ZEROCNK
    w = 0.001 * mixratio(p, td)
    vt = (tk * (1. + w / eps) / (1. + w)) - ZEROCNK
    if np.isnan(vt):
        return t
    return vt


@njit(cache=True, fastmath=True, nogil=True)
def lcltemp(t, td):
    s = t - td
    dlt = s * (1.2185 + 0.001278 * t + s * (-0.00219 + 1.173e-5 * s - 0.0000052 * t))
    return t - dlt


@njit(cache=True, fastmath=True, nogil=True)
def thalvl(theta_, t):
    t = t + ZEROCNK
    theta_ = theta_ + ZEROCNK
    return 1000. / ((theta_ / t) ** (1. / ROCP))


@njit(cache=True, fastmath=True, nogil=True)
def drylift(p, t, td):
    t2 = lcltemp(t, td)
    p2 = thalvl(theta(p, t, 1000.), t2)
    return p2, t2


@njit(cache=True, fastmath=True, nogil=True)
def satlift(p, thetam, conv=0.1):
    if abs(p - 1000.) - 0.001 <= 0:
        return thetam
    eor = 999.0
    t1 = 0.0
    e1 = 0.0
    t2 = 0.0
    e2 = 0.0
    rate = 1.0
    pwrp = (p / 1000.) ** ROCP
    first = True
    while abs(eor) - conv > 0:
        if first:
            t1 = (thetam + ZEROCNK) * pwrp - ZEROCNK
            e1 = wobf(t1) - wobf(thetam)
            rate = 1.0
            first = False
        else:
            rate = (t2 - t1) / (e2 - e1)
            t1 = t2
            e1 = e2
        t2 = t1 - (e1 * rate)
        e2 = (t2 + ZEROCNK) / pwrp - ZEROCNK
        e2 += wobf(t2) - wobf(e2) - thetam
        eor = e2 * rate
    return t2 - eor


@njit(cache=True, fastmath=True, nogil=True)
def wetlift(p, t, p2):
    thta = theta(p, t, 1000.)
    thetam = thta - wobf(thta) + wobf(t)
    return satlift(p2, thetam)


@njit(cache=True, fastmath=True, nogil=True)
def lifted(p, t, td, lev):
    p2, t2 = drylift(p, t, td)
    return wetlift(p2, t2, lev)


@njit(cache=True, fastmath=True, nogil=True)
def temp_at_mixrat(w, p):
    x = np.log10(w * p / (622. + w))
    x = (10. ** ((c1 * x) + c2) - c3 + (c4 * (10. ** (c5 * x) - c6) ** 2)) - ZEROCNK
    return x


# Warm the JIT cache once at import time (cheap: scalar calls) so the first
# real sounding computed by the app isn't the one that eats the ~1-2s
# compilation cost.
def _warm():
    try:
        wobf(5.0)
        theta(1000., 20.)
        vappres(20.)
        mixratio(1000., 10.)
        virtemp(1000., 20., 10.)
        drylift(1000., 20., 10.)
        satlift(900., 15.)
        wetlift(900., 15., 800.)
        lifted(1000., 20., 10., 800.)
        temp_at_mixrat(10., 900.)
    except Exception:
        pass
_warm()
