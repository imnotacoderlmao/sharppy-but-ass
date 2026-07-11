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

@njit(cache=True, fastmath=False, nogil=True)
def thetae(p, t, td):
    p2, t2 = drylift(p, t, td)
    return theta(100., wetlift(p2, t2, 100.), 1000.)


@njit(cache=True, fastmath=False, nogil=True)
def wetbulb(p, t, td):
    p2, t2 = drylift(p, t, td)
    return wetlift(p2, t2, p)


@njit(cache=True, fastmath=False, nogil=True)
def theta_profile_loop(pres, tmpc):
    '''Whole-profile version of thermo.theta, run inside one compiled
    loop instead of a Python for-loop calling the scalar dispatch path
    once per level (see profile.py get_theta_profile).'''
    n = pres.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        p = pres[i]
        t = tmpc[i]
        if p != p or t != t:
            out[i] = np.nan
        else:
            out[i] = theta(p, t, 1000.)
    return out


@njit(cache=True, fastmath=False, nogil=True)
def wetbulb_profile_loop(pres, tmpc, dwpc):
    '''Whole-profile version of thermo.wetbulb (see theta_profile_loop).'''
    n = pres.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        p = pres[i]
        t = tmpc[i]
        td = dwpc[i]
        if p != p or t != t or td != td:
            out[i] = np.nan
        else:
            out[i] = wetbulb(p, t, td)
    return out


@njit(cache=True, fastmath=False, nogil=True)
def thetae_profile_loop(pres, tmpc, dwpc):
    '''Whole-profile version of thermo.thetae (see theta_profile_loop).'''
    n = pres.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        p = pres[i]
        t = tmpc[i]
        td = dwpc[i]
        if p != p or t != t or td != td:
            out[i] = np.nan
        else:
            out[i] = thetae(p, t, td)
    return out



@njit(cache=True, fastmath=False, nogil=True)
def _interp1_clamp_nan(q, x, y):
    '''
    Scalar linear interpolation against an ascending x array, matching
    np.interp(q, x, y, left=np.nan, right=np.nan) exactly (numba's
    np.interp doesn't support the left/right kwargs).
    '''
    n = x.shape[0]
    if q < x[0] or q > x[n - 1]:
        return np.nan
    lo = 0
    hi = n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if x[mid] <= q:
            lo = mid
        else:
            hi = mid
    x0 = x[lo]
    x1 = x[hi]
    if x1 == x0:
        return y[lo]
    y0 = y[lo]
    y1 = y[hi]
    return y0 + (y1 - y0) * (q - x0) / (x1 - x0)


@njit(cache=True, fastmath=False, nogil=True)
def dcape_min_mean_thetae_loop(pres_cands, logp_valid, thetae_valid):
    '''
    Fully-compiled replacement for dcape()'s downdraft-source search:

        for i in idx:
            thta_e_mean = mean_thetae(prof, pbot=pres[i], ptop=pres[i]-100.)
            if utils.QC(thta_e_mean) and thta_e_mean < mine: ...

    which (profiled on a 16k-level sounding) called mean_thetae() up to
    ~14000 times, each call building a fresh 1mb-step np.arange, a fresh
    interp.thetae() array query (masked-array construction, ma.log10,
    isnan-masking), and a ma.average. This loop does the same 1mb-step
    pressure-weighted average of interpolated theta-e, using the exact
    same (cached, mask-dropped, ascending) logp/theta-e arrays
    interp.thetae() itself interpolates against (see interp._valid_xy),
    without rebuilding a MaskedArray or a Python-level np.arange per
    candidate.

    `pres_cands` is prof.pres[idx] (the candidate bottom pressures,
    already restricted to the lowest 400mb by the caller). Returns
    (mine, minp), matching the original loop's mine/minp. Candidates
    whose ptop falls outside the profile's valid theta-e range
    interpolate to NaN and are skipped, matching the original
    mean_thetae()'s `return ma.masked` (rejected by utils.QC()) for an
    out-of-range ptop.
    '''
    mine = 1000.0
    minp = -999.0
    n = pres_cands.shape[0]
    for k in range(n):
        pbot = pres_cands[k]
        ptop = pbot - 100.

        num_wsum = 0.0
        den_wsum = 0.0
        valid = True
        p = pbot
        # matches np.arange(pbot, ptop - 1., -1.) exactly: pbot, pbot-1,
        # ..., down to (and including) ptop.
        while p > ptop - 1.0:
            q = np.log10(p)
            te = _interp1_clamp_nan(q, logp_valid, thetae_valid)
            if te != te:
                valid = False
                break
            num_wsum += te * p
            den_wsum += p
            p -= 1.0

        if valid and den_wsum > 0.0:
            thta_e_mean = num_wsum / den_wsum
            if thta_e_mean < mine:
                minp = pbot - 50.
                mine = thta_e_mean

    return mine, minp


@njit(cache=True, fastmath=False, nogil=True)
def satlift_array_loop(p, thetam):
    '''
    Compiled replacement for thermo.satlift(p_array, thetam_array) (the
    generic-vectorized/numpy path taken whenever satlift is called with
    array arguments, e.g. cape()/parcelx()/dcape()'s "solve the whole
    moist adiabat in one call instead of level-by-level" precomputation).

    The numpy array path runs ONE Newton iteration loop across the whole
    array at once: every element is re-evaluated (including a fresh
    wobf() array allocation) on every iteration until the *slowest*
    element converges, even though most elements converge in far fewer
    steps. Profiled: satlift()'s internal wobf() calls were the single
    largest cost in effective_inflow_layer() (which calls cape(), and
    therefore this precomputation, once per level scanned).

    This loop instead calls the existing scalar-converging njit
    `satlift` kernel per element, so each element stops iterating the
    moment *it* converges -- same math, same convergence criterion
    (conv=0.1, matching the array path's default and every call site's
    usage), no repeated whole-array reallocation.

    `p` and `thetam` must be same-length float64 arrays (thetam is
    typically a single value broadcast via np.full_like at the call
    site; kept as an array here to match np.full_like(target_pres,
    thetam)'s literal shape rather than assuming it's constant).
    '''
    n = p.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        pi = p[i]
        if pi != pi:
            out[i] = np.nan
        else:
            out[i] = satlift(pi, thetam[i])
    return out


@njit(cache=True, fastmath=False, nogil=True)
def _safe_div(a, b):
    '''
    a/b, matching numpy's array-division semantics (+-inf or nan on an
    exact-zero denominator, no exception) instead of numba's default
    python error model, which raises ZeroDivisionError on exact-zero
    float division the way CPython's plain `float / float` does. The
    original cape()/effective_inflow_layer code this mirrors operates
    on numpy arrays throughout, so an extreme candidate parcel hitting
    a zero denominator there silently produces inf/nan (propagating
    harmlessly into a failed ecape/ecinh comparison) rather than
    crashing -- this keeps that same behavior in the compiled version.
    fastmath=False deliberately here (unlike the rest of this kernel)
    since the whole point is exact zero-comparison, which fastmath's
    reassociation could undermine.
    '''
    if b == 0.0:
        if a > 0.0:
            return np.inf
        elif a < 0.0:
            return -np.inf
        else:
            return np.nan
    return a / b


@njit(cache=True, fastmath=True, nogil=True)
def cape_only(pres, tmpc, dwpc, pbot, ptop, trunc,
              pres_p, tmpc_p, hght_p, vtmp_p,
              logp_hght_v, hght_v, logp_tmpc_v, tmpc_v,
              logp_dwpc_v, dwpc_v, logp_vtmp_v, vtmp_v):
    '''
    Fully-compiled equivalent of params.cape(prof, pbot=pbot, ptop=ptop,
    trunc=trunc, pres=pres, tmpc=tmpc, dwpc=dwpc) covering cape()'s core
    numerics (mixing-layer CINH + moist-adiabat lift, matching its exact
    algorithm including the "close the integral out to ptop" step when
    truncated) without the Parcel/DefineParcel object construction or
    kwargs dispatch overhead.

    Caller contract, matching effective_inflow_layer()/convective_temp()
    usage (both callers always pass an explicit numeric pbot/ptop, never
    cape()'s own "None means default to surface/last-level" sentinel --
    resolve that default in the CALLING Python code before calling this,
    same as passing prof.pres[prof.sfc]/prof.pres[-1] explicitly):
      - pres/tmpc/dwpc, pbot, ptop must be plain non-NaN floats.
      - the profile arrays (pres_p/tmpc_p/hght_p/vtmp_p) and the cached
        (logp, field) interpolation arrays must contain no NaN in the
        sense that fastmath=True requires -- masked/NaN candidate
        entries must be routed around this function in Python, exactly
        as effective_inflow_layer's per-candidate masked-check already
        does. This is the single hottest per-level call in the app
        (effective_inflow_layer scans O(N) candidates; convective_temp
        iterates until CINH clears), so avoiding fastmath's conservative
        default semantics is worth this extra call-site care.

    Returns (bplus, bminus, pbot_adjusted). bplus/bminus match cape()'s
    pcl.bplus/pcl.bminus. pbot_adjusted is pbot after both of cape()'s
    internal adjustments (the pre-mixing-layer "if pbot > pres: pbot =
    pres" and the post-LCL "if pbot > pe2: pbot = pe2") -- callers that
    need to mirror cape()'s pcl.blayer/pcl.pbot bookkeeping should use
    this rather than the pbot they originally passed in. If the LCL
    ends up above the top of the profile's data (a dropsonde-type edge
    case -- interp.vtmp(prof, pbot)/interp.vtmp(prof, ptop) would be
    masked in the original), returns (nan, nan, pbot_adjusted) instead;
    the caller checks for NaN the same way it would check utils.QC() on
    the masked original.
    '''
    n = pres_p.shape[0]

    # First pbot adjustment (matches cape()'s "if pbot > pres: pbot =
    # pres", which happens BEFORE the mixing layer and so affects the
    # mixing-layer range itself -- this was missing from an earlier
    # version of this kernel that only had the second (post-LCL)
    # adjustment below, which happened to still be correct for
    # effective_inflow_layer's specific call pattern (pbot always
    # already equals pres there) but wasn't a faithful general port).
    if pbot > pres:
        pbot = pres

    pe2, tp2 = drylift(pres, tmpc, dwpc)
    if pe2 != pe2:
        return np.nan, np.nan, pbot
    blupper = pe2
    theta_parcel = theta(pe2, tp2, 1000.)
    blmr = mixratio(pres, dwpc)

    # Mixing-layer CINH: pbot down to blupper, 1mb steps. Matches
    # np.arange(pbot, blupper - 1., -1.) exactly (dp=-1). No-op (loop
    # body never runs) if pbot <= blupper, matching an empty/negative
    # np.arange there.
    totn = 0.0
    p = pbot
    have_prev = False
    prev_h = 0.0
    prev_tdef = 0.0
    while p > blupper - 1.0:
        q = np.log10(p)
        hh_i = _interp1_clamp_nan(q, logp_hght_v, hght_v)
        t_i = _interp1_clamp_nan(q, logp_tmpc_v, tmpc_v)
        d_i = _interp1_clamp_nan(q, logp_dwpc_v, dwpc_v)
        if hh_i == hh_i and t_i == t_i and d_i == d_i:
            th_env = theta(p, t_i, 1000.)
            tv_env = virtemp(p, th_env, d_i)
            tmp1 = virtemp(p, theta_parcel, temp_at_mixrat(blmr, p))
            tdef_i = _safe_div(tmp1 - tv_env, ctok(tv_env))
            if have_prev:
                lyre = G * (prev_tdef + tdef_i) / 2. * (hh_i - prev_h)
                if lyre < 0:
                    totn += lyre
            prev_h = hh_i
            prev_tdef = tdef_i
            have_prev = True
        else:
            have_prev = False
        p -= 1.0

    # Move bottom layer to top of boundary layer (matches cape()'s
    # "if pbot > pe2: pbot = pe2").
    if pbot > pe2:
        pbot = pe2
    if pbot < ptop:
        return np.nan, np.nan, pbot

    # lptr: smallest index where pres_p[i] < pbot (matches
    # ma.where(pbot > prof.pres)[0].min()).
    lptr = 0
    while lptr < n and not (pres_p[lptr] < pbot):
        lptr += 1
    # uptr: LARGEST index where pres_p[i] > ptop (matches
    # ma.where(ptop < prof.pres)[0].max() exactly -- note this is
    # generally n-2, not n-1, when ptop equals the profile's last
    # element: that element itself equals ptop, not strictly greater,
    # so it's excluded from the main loop and picked up by the closing
    # step below instead, same as the original).
    uptr = 0
    for ii in range(n):
        if pres_p[ii] > ptop:
            uptr = ii

    pe1 = pbot
    q_pe1 = np.log10(pe1)
    h1 = _interp1_clamp_nan(q_pe1, logp_hght_v, hght_v)
    te1 = _interp1_clamp_nan(q_pe1, logp_vtmp_v, vtmp_v)
    tp1 = tp2

    thta_lcl = theta(pe1, tp1, 1000.)
    thetam = thta_lcl - wobf(thta_lcl) + wobf(tp1)

    tp_precomputed = np.full(n, np.nan, dtype=np.float64)
    target_pres = pres_p[lptr:]
    thetam_arr = np.full(target_pres.shape[0], thetam, dtype=np.float64)
    tp_precomputed[lptr:] = satlift_array_loop(target_pres, thetam_arr)

    totp_add, totn_add, lyre, pe1_f, h1_f, te1_f, tp1_f, truncated = cape_lift_loop(
        pres_p, tmpc_p, hght_p, vtmp_p, tp_precomputed, lptr, uptr,
        pe1, h1, te1, tp1, trunc)

    totp = totp_add
    totn = totn + totn_add

    bplus = np.nan
    bminus = np.nan
    if truncated:
        lyrf = lyre
        pe2f = pe1_f
        if lyrf > 0:
            bplus = totp - lyrf
            bminus = totn
        else:
            bplus = totp
            if pe2f > 500.:
                bminus = totn + lyrf
            else:
                bminus = totn

        # Close the integral out to ptop. pe1_f/h1_f/te1_f only equal
        # ptop's own height/virtual-temp exactly when the main loop
        # happened to stop exactly there (e.g. effective_inflow_layer's
        # ptop == pres_p[-1] pattern, where uptr's largest-strictly-
        # greater definition above still leaves a one-level gap -- see
        # that case's own zero-width closing layer below); in general
        # (particularly trunc=True stopping early at <=500mb, as
        # convective_temp uses) it's a genuine nonzero layer and needs
        # the real computation, not a skip.
        pe3, h3, te3, tp3 = pe1_f, h1_f, te1_f, tp1_f
        pe2c = ptop
        if pe3 != pe2c:
            q_pe2c = np.log10(pe2c)
            h2c = _interp1_clamp_nan(q_pe2c, logp_hght_v, hght_v)
            te2c = _interp1_clamp_nan(q_pe2c, logp_vtmp_v, vtmp_v)
            tp2c = wetlift(pe3, tp3, pe2c)
            tdef3 = _safe_div(virtemp(pe3, tp3, tp3) - te3, ctok(te3))
            tdef2 = _safe_div(virtemp(pe2c, tp2c, tp2c) - te2c, ctok(te2c))
            lyrf2 = G * (tdef3 + tdef2) / 2. * (h2c - h3)
            if lyrf2 > 0:
                bplus += lyrf2
            else:
                if pe2c > 500.:
                    bminus += lyrf2
        if bplus == 0:
            bminus = 0.

    return bplus, bminus, pbot


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


@njit(cache=True, fastmath=False, nogil=True)
def posneg_wetbulb_loop(pres_p, hght_p, temp_p, dwpc_p, lptr, uptr, pe1, h1, te1):
    '''
    Fully-compiled version of watch_type.posneg_wetbulb()'s main loop.

    Unlike cape()/parcelx()'s lifting loops, this one genuinely can't be
    vectorized with a single batch solve: warmlayer/coldlayer are "first
    time this crossing is seen" flags that depend on the *order* levels
    are visited in, not just their final values. (An earlier attempt at
    vectorizing this loop with a batched wetbulb() array call was
    reverted -- see git history -- because numpy vectorization's fixed
    per-call overhead (new array allocations, dispatch into the masked-
    array-aware array-code path) made it *slower* for the short ranges
    this function is typically called with, only paying off at
    whole-profile scale.)

    Compiling the loop itself sidesteps that tradeoff: there's no
    per-call array-allocation overhead to amortize, so this wins
    regardless of range length -- it just removes Python interpreter
    dispatch and MaskedArray.__getitem__ overhead per level, while still
    doing genuine scalar-at-a-time wetbulb() Newton solves (calling the
    compiled `wetbulb` kernel directly, with none of the Python-call
    overhead a `thermo.wetbulb()` call would have paid on top of it).

    `temp_p`/`dwpc_p` are precomputed once via a single vectorized
    interp.temp()/interp.dwpt() call over the same range (cheap linear
    interpolation, not a Newton solve, so vectorizing *that* part is
    still worthwhile) -- see watch_type.posneg_wetbulb().
    '''
    warmlayer = 0
    coldlayer = 0
    lyre = 0.0
    totp = 0.0
    totn = 0.0
    tote = 0.0
    ptop = 0.0
    pbot = 0.0

    for i in range(uptr, lptr - 1, -1):
        pe2 = pres_p[i]
        h2 = hght_p[i]
        te2 = wetbulb(pe2, temp_p[i], dwpc_p[i])

        tdef1 = (0. - te1) / (te1 + ZEROCNK)
        tdef2 = (0. - te2) / (te2 + ZEROCNK)
        lyre = 9.8 * (tdef1 + tdef2) / 2.0 * (h2 - h1)

        if te2 > 0:
            if warmlayer == 0:
                warmlayer = 1
                ptop = pe2

        if te2 < 0:
            if warmlayer == 1 and coldlayer == 0:
                coldlayer = 1
                pbot = pe2

        if warmlayer > 0:
            if lyre > 0:
                totp += lyre
            else:
                totn += lyre
            tote += lyre

        pe1 = pe2
        h1 = h2
        te1 = te2

    return warmlayer, coldlayer, totp, totn, ptop, pbot


@njit(cache=True, fastmath=False, nogil=True)
def posneg_temperature_loop(pres_p, hght_p, temp_p, lptr, uptr, pe1, h1, te1):
    '''
    Fully-compiled version of watch_type.posneg_temperature()'s main loop
    -- same "sequential, not vectorizable, but still worth compiling"
    reasoning as posneg_wetbulb_loop above, minus the Newton solve (this
    one only ever looks at temperature, no wetbulb calculation needed).
    '''
    warmlayer = 0
    coldlayer = 0
    lyre = 0.0
    totp = 0.0
    totn = 0.0
    tote = 0.0
    ptop = 0.0
    pbot = 0.0

    for i in range(uptr, lptr - 1, -1):
        pe2 = pres_p[i]
        h2 = hght_p[i]
        te2 = temp_p[i]

        tdef1 = (0. - te1) / (te1 + ZEROCNK)
        tdef2 = (0. - te2) / (te2 + ZEROCNK)
        lyre = 9.8 * (tdef1 + tdef2) / 2.0 * (h2 - h1)

        if te2 > 0:
            if warmlayer == 0:
                warmlayer = 1
                ptop = pe2

        if te2 < 0:
            if warmlayer == 1 and coldlayer == 0:
                coldlayer = 1
                pbot = pe2

        if warmlayer > 0:
            if lyre > 0:
                totp += lyre
            else:
                totn += lyre
            tote += lyre

        pe1 = pe2
        h1 = h2
        te1 = te2

    return warmlayer, coldlayer, totp, totn, ptop, pbot


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
        cape_lift_loop(_warm_pres, _warm_tmpc, _warm_hght, _warm_vtmp, _warm_tp, 0, 2, 1000., 0., 21., 20., False)
        thetae(1000., 20., 10.)
        wetbulb(1000., 20., 10.)
        _warm_dwpc = np.array([10., 8., 6.], dtype=np.float64)
        theta_profile_loop(_warm_pres, _warm_tmpc)
        wetbulb_profile_loop(_warm_pres, _warm_tmpc, _warm_dwpc)
        thetae_profile_loop(_warm_pres, _warm_tmpc, _warm_dwpc)
        posneg_wetbulb_loop(_warm_pres, _warm_hght, _warm_tmpc, _warm_dwpc, 0, 2, 1000., 0., 21.)
        posneg_temperature_loop(_warm_pres, _warm_hght, _warm_tmpc, 0, 2, 1000., 0., 21.)
        _warm_logp = np.log10(_warm_pres[::-1].copy())
        _warm_thetae = np.array([340., 345., 350.], dtype=np.float64)
        dcape_min_mean_thetae_loop(_warm_pres, _warm_logp, _warm_thetae)
        satlift_array_loop(_warm_pres, np.full_like(_warm_pres, 18.))
        _warm_hght_v = _warm_hght
        _warm_vtmp_v = _warm_vtmp
        cape_only(1000., 20., 10., 1000., 900., False,
                  _warm_pres, _warm_tmpc, _warm_hght, _warm_vtmp,
                  _warm_logp, _warm_hght_v, _warm_logp, _warm_tmpc,
                  _warm_logp, _warm_dwpc, _warm_logp, _warm_vtmp_v)
        cape_only(1000., 20., 10., 1000., 900., True,
                  _warm_pres, _warm_tmpc, _warm_hght, _warm_vtmp,
                  _warm_logp, _warm_hght_v, _warm_logp, _warm_tmpc,
                  _warm_logp, _warm_dwpc, _warm_logp, _warm_vtmp_v)
    except Exception:
        pass
_warm()