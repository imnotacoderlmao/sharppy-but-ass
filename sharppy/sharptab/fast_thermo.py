'''
Numba-accelerated scalar thermodynamic kernels.

These mirror the scalar code paths in `sharppy.sharptab.thermo` exactly
(same constants, same algorithms) but are compiled with numba so that the
per-level Newton iterations in `params.parcelx`/`params.cape`/`params.dcape`
(which call these functions hundreds of times per parcel, and are run for
several parcels per sounding) no longer pay Python/NumPy call overhead on
every single level.

numba is a hard dependency (see setup.py/environment.yml). there is no
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


@njit(cache=True, fastmath=False, nogil=True)
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


@njit(cache=True, fastmath=False, nogil=True)
def theta(p, t, p2=1000.):
    return ((t + ZEROCNK) * (p2 / p) ** ROCP) - ZEROCNK


@njit(cache=True, fastmath=False, nogil=True)
def ctok(t):
    return t + ZEROCNK


@njit(cache=True, fastmath=False, nogil=True)
def vappres(t):
    pol = t * (1.1112018e-17 + (t * -3.0994571e-20))
    pol = t * (2.1874425e-13 + (t * (-1.789232e-15 + pol)))
    pol = t * (4.3884180e-09 + (t * (-2.988388e-11 + pol)))
    pol = t * (7.8736169e-05 + (t * (-6.111796e-07 + pol)))
    pol = 0.99999683 + (t * (-9.082695e-03 + pol))
    return 6.1078 / pol ** 8


@njit(cache=True, fastmath=False, nogil=True)
def mixratio(p, t):
    x = 0.02 * (t - 12.5 + (7500. / p))
    wfw = 1. + (0.0000045 * p) + (0.0014 * x * x)
    fwesw = wfw * vappres(t)
    return 621.97 * (fwesw / (p - fwesw))


@njit(cache=True, fastmath=False, nogil=True)
def virtemp(p, t, td):
    tk = t + ZEROCNK
    w = 0.001 * mixratio(p, td)
    vt = (tk * (1. + w / eps) / (1. + w)) - ZEROCNK
    if np.isnan(vt):
        return t
    return vt


@njit(cache=True, fastmath=False, nogil=True)
def lcltemp(t, td):
    s = t - td
    dlt = s * (1.2185 + 0.001278 * t + s * (-0.00219 + 1.173e-5 * s - 0.0000052 * t))
    return t - dlt


@njit(cache=True, fastmath=False, nogil=True)
def thalvl(theta_, t):
    t = t + ZEROCNK
    theta_ = theta_ + ZEROCNK
    return 1000. / ((theta_ / t) ** (1. / ROCP))


@njit(cache=True, fastmath=False, nogil=True)
def drylift(p, t, td):
    t2 = lcltemp(t, td)
    p2 = thalvl(theta(p, t, 1000.), t2)
    return p2, t2


@njit(cache=True, fastmath=False, nogil=True)
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


@njit(cache=True, fastmath=False, nogil=True)
def wetlift(p, t, p2):
    thta = theta(p, t, 1000.)
    thetam = thta - wobf(thta) + wobf(t)
    return satlift(p2, thetam)


@njit(cache=True, fastmath=False, nogil=True)
def lifted(p, t, td, lev):
    p2, t2 = drylift(p, t, td)
    return wetlift(p2, t2, lev)


@njit(cache=True, fastmath=False, nogil=True)
def temp_at_mixrat(w, p):
    x = np.log10(w * p / (622. + w))
    x = (10. ** ((c1 * x) + c2) - c3 + (c4 * (10. ** (c5 * x) - c6) ** 2)) - ZEROCNK
    return x


G = 9.80665  # sharptab.constants.G, duplicated here so this stays a
             # self-contained compiled unit (avoids importing params.py,
             # which imports thermo.py, which imports this module).


@njit(cache=True, fastmath=False, nogil=True)
def cape_lift_loop(pres_p, tmpc_p, hght_p, vtmp_p, tp_precomputed, lptr, uptr, pe1, h1, te1, tp1, trunc):
    '''
    Fully-compiled version of the per-level accumulation loop in
    params.cape()'s per level accumulation loop (the only lifting path / common path).

    `tp_precomputed[i]` is the parcel's temperature at pres_p[i], computed
    ONCE for the whole profile via a vectorized satlift (see params.cape())
    rather than recomputed sequentially level-by-level here. This matters
    because moist-adiabatic lift is path-independent: the temperature at
    any pressure along a moist adiabat depends only on the adiabat's
    defining theta-w, not on which point along it you last computed.
    so deriving that theta-w once at the LCL and solving for every level
    at once (which vectorizes cleanly, since satlift's Newton iteration
    can run on the whole pressure array simultaneously) gives the same
    curve as the old level-by-level recomputation, just without paying
    for the recomputation N times. (It also avoids that old approach's
    accumulating drift, documented in thermo.satlift's own docstring where
    each step used to re-derive theta-w from the *previous* step's
    Newton converged to 0.1C rather than the exact original,
    so small per-step errors compounded over many levels.)

    Everything else here, the truncation condition, the totp/totn
    accumulation, the "i >= uptr and not utils.QC(pcl.bplus)" reduction to
    "i >= uptr" is unchanged from the original; see prior versions of
    this docstring / params.cape() for that reasoning.
    '''
    totp = 0.0
    totn = 0.0
    lyre = 0.0
    n = pres_p.shape[0]
    truncated = False

    for i in range(lptr, n):
        tmpc_i = tmpc_p[i]
        if tmpc_i != tmpc_i:  # NaN check == "not utils.QC(prof.tmpc[i])"
            continue

        pe2 = pres_p[i]
        h2 = hght_p[i]
        te2 = vtmp_p[i]
        tp2 = tp_precomputed[i]

        tdef1 = (virtemp(pe1, tp1, tp1) - te1) / (te1 + ZEROCNK)
        tdef2 = (virtemp(pe2, tp2, tp2) - te2) / (te2 + ZEROCNK)
        lyre = G * (tdef1 + tdef2) / 2. * (h2 - h1)

        if lyre > 0:
            totp += lyre
        else:
            if pe2 > 500.:
                totn += lyre

        pe1 = pe2
        h1 = h2
        te1 = te2
        tp1 = tp2

        if (trunc and pe2 <= 500.) or (i >= uptr):
            truncated = True
            break

    return totp, totn, lyre, pe1, h1, te1, tp1, truncated


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
        _warm_pres = np.array([1000., 950., 900.], dtype=np.float64)
        _warm_tmpc = np.array([20., 18., 16.], dtype=np.float64)
        _warm_hght = np.array([0., 500., 1000.], dtype=np.float64)
        _warm_vtmp = np.array([21., 19., 17.], dtype=np.float64)
        _warm_tp = np.array([20., 18.5, 17.], dtype=np.float64)
        cape_lift_loop(_warm_pres, _warm_tmpc, _warm_hght, _warm_vtmp, _warm_tp,
                       0, 2, 1000., 0., 21., 20., False)
    except Exception:
        pass
_warm()
