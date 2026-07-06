''' Interpolation Routines '''
from __future__ import division
import numpy as np
import numpy.ma as ma
import numpy.testing as npt
from sharppy.sharptab import utils, thermo
from sharppy.sharptab.constants import *


__all__ = ['pres', 'hght', 'temp', 'dwpt', 'vtmp', 'components', 'vec']
__all__ += ['thetae', 'wetbulb', 'theta', 'mixratio']
__all__ += ['to_agl', 'to_msl']


# ---------------------------------------------------------------------------
# Profile-scoped interpolation cache
#
# params.parcelx()/cape()/etc. call interp.temp/vtmp/dwpt/hght/... hundreds of
# times per parcel (LFC/EL bisection, layer-top/bottom lookups, temp_lvl,
# bulk_rich, ...), always against the *same* prof.pres/hght/tmpc/dwpc/vtmp/...
# arrays. Those arrays never mutate once a Profile is built (a "changed"
# sounding is a new Profile object, see prof_collection.py), so the
# mask-filtering work generic_interp_pres/hght did on every single call is
# pure repeated work: same inputs, same output, every time.
#
# We cache the filtered (mask-dropped) x/y arrays on the profile object
# itself the first time a given field pair is interpolated, keyed by a
# fixed string per field (not by id() of the arrays), so there's no
# use-after-free/identity-reuse hazard: the cache lives and dies with the
# Profile instance that owns the data it describes.
# ---------------------------------------------------------------------------

def _valid_xy(prof, key, x, y):
    '''
    Return (x_valid, y_valid) plain ndarrays with masked/invalid entries
    dropped from both, computed once per Profile instance and cached under
    `key` for reuse by every subsequent interpolation against that pair.
    '''
    cache = prof.__dict__.get('_interp_valid_cache')
    if cache is None:
        cache = {}
        try:
            prof._interp_valid_cache = cache
        except Exception:
            # If for some reason attribute assignment isn't allowed (custom
            # prof-like object), just skip caching rather than fail the call.
            cache = None
    if cache is not None:
        entry = cache.get(key)
        if entry is not None:
            return entry

    not_masked = ~ma.getmaskarray(x) & ~ma.getmaskarray(y)
    x_valid = np.asarray(x)[not_masked]
    y_valid = np.asarray(y)[not_masked]
    entry = (x_valid, y_valid)
    if cache is not None:
        cache[key] = entry
    return entry


def _interp_cached(prof, key, query, x, y, log_query=False, log_result=False):
    '''
    Cached counterpart to generic_interp_pres for the fixed, profile-bound
    (x, y) pairs used by the wrapper functions below (pres/hght/temp/dwpt/
    vtmp/theta/thetae/wetbulb/mixratio/components/omeg). Numerically
    identical to generic_interp_pres(query, x, y); see that function's
    docstring for the (already reversed/log-transformed as needed) inputs.

    The overwhelmingly common case (profiled: params.parcelx's LFC/EL
    bisection, temp_lvl, layer lookups, ...) is a single scalar pressure/
    height query. That path is kept entirely free of numpy.ma: no
    MaskedArray construction, no ma.where, no ma.log10 -- those turned out
    (by profiling) to be a bigger cost than the interpolation itself.
    The array-query path (rare from these wrappers, but still supported)
    falls back to the original masked-array-returning behavior.
    '''
    x_valid, y_valid = _valid_xy(prof, key, x, y)

    if x_valid.shape[0] == 0 or y_valid.shape[0] == 0:
        return ma.masked_where(ma.ones(np.shape(query)), query)

    is_array_query = np.ndim(query) != 0

    if not is_array_query:
        # Scalar fast path.
        if query is ma.masked:
            return ma.masked
        q = np.log10(query) if log_query else query
        val = np.interp(q, x_valid, y_valid, left=np.nan, right=np.nan)
        val = float(val)
        if log_result:
            val = 10. ** val
        if val != val:  # NaN check (cheaper than np.isnan for a plain float)
            return ma.masked
        return val

    # Array-query path: keep the original masked-array-returning contract.
    # np.arange(...)-built query arrays (the common case, e.g.
    # mean_thetae's `p = np.arange(pbot, ptop+dp, dp)`, called ~200 or more
    # times per dcape() call) are plain ndarrays, not MaskedArrays, using
    # ma.log10 on those pays masked-array construction overhead for
    # nothing. Only fall back to ma.log10 if query is genuinely masked.
    if log_query:
        q = ma.log10(query) if isinstance(query, ma.MaskedArray) else np.log10(query)
    else:
        q = query
    field_intrp = np.interp(q, x_valid, y_valid, left=np.nan, right=np.nan)
    field_intrp = ma.masked_where(np.isnan(field_intrp), field_intrp)
    if log_result:
        field_intrp = 10 ** field_intrp
    return field_intrp



def pres(prof, h):
    '''
    Interpolates the given data to calculate a pressure at a given height

    Parameters
    ----------
    prof : profile object
        Profile object
    h : number, numpy array
        Height (m) of the level for which pressure is desired

    Returns
    -------
    Pressure (hPa) at the given height : number, numpy array

    '''
    return _interp_cached(prof, 'hght_logp', h, prof.hght, prof.logp, log_result=True)


def hght(prof, p):
    '''
    Interpolates the given data to calculate a height at a given pressure

    Parameters
    ----------
    prof : profile object
        Profile object
    p : number, numpy array
        Pressure (hPa) of the level for which height is desired

    Returns
    -------
    Height (m) at the given pressure : number, numpy array

    '''
    # Note: numpy's interpolation routine expects the interpolation
    # routine to be in ascending order. Because pressure decreases in the
    # vertical, we must reverse the order of the two arrays to satisfy
    # this requirement.
    return _interp_cached(prof, 'logp_hght', p, prof.logp[::-1], prof.hght[::-1], log_query=True)

def omeg(prof, p):
    '''
    Interpolates the given data to calculate a omega at a given pressure

    Parameters
    ----------
    prof : profile object
        Profile object
    p : number, numpy array
        Pressure (hPa) of the level for which temperature is desired

    Returns
    -------
    Omega (microbars/second) at the given pressure : number, numpy array

    '''
    # Note: numpy's interpolation routine expects the interpolation
    # routine to be in ascending order. Because pressure decreases in the
    # vertical, we must reverse the order of the two arrays to satisfy
    # this requirement.
    return _interp_cached(prof, 'logp_omeg', p, prof.logp[::-1], prof.omeg[::-1], log_query=True)

def temp(prof, p):
    '''
    Interpolates the given data to calculate a temperature at a given pressure

    Parameters
    ----------
    prof : profile object
        Profile object
    p : number, numpy array
        Pressure (hPa) of the level for which temperature is desired

    Returns
    -------
    Temperature (C) at the given pressure : number, numpy array

    '''
    # Note: numpy's interpolation routine expects the interpolation
    # routine to be in ascending order. Because pressure decreases in the
    # vertical, we must reverse the order of the two arrays to satisfy
    # this requirement.
    return _interp_cached(prof, 'logp_tmpc', p, prof.logp[::-1], prof.tmpc[::-1], log_query=True)

def thetae(prof, p):
    '''
        Interpolates the given data to calculate theta-e at a given pressure

        Parameters
        ----------
        prof : profile object
        Profile object
        p : number, numpy array
        Pressure (hPa) of the level for which temperature is desired

        Returns
        -------
        Theta-E (C) at the given pressure : number, numpy array

        '''
    # Note: numpy's interpolation routine expects the interpolation
    # routine to be in ascending order. Because pressure decreases in the
    # vertical, we must reverse the order of the two arrays to satisfy
    # this requirement.
    return _interp_cached(prof, 'logp_thetae', p, prof.logp[::-1], prof.thetae[::-1], log_query=True)

def mixratio(prof, p):
    '''
        Interpolates the given data to calculate water vapor mixing ratio at a given pressure

        Parameters
        ----------
        prof : profile object
        Profile object
        p : number, numpy array
        Pressure (hPa) of the level for which mixing ratio is desired

        Returns
        -------
        Water vapor mixing ratio (g/kg) at the given pressure : number, numpy array

        '''
    # Note: numpy's interpolation routine expects the interpolation
    # routine to be in ascending order. Because pressure decreases in the
    # vertical, we must reverse the order of the two arrays to satisfy
    # this requirement.
    return _interp_cached(prof, 'logp_wvmr', p, prof.logp[::-1], prof.wvmr[::-1], log_query=True)


def theta(prof, p):
    '''
        Interpolates the given data to calculate theta at a given pressure

        Parameters
        ----------
        prof : profile object
        Profile object
        p : number, numpy array
        Pressure (hPa) of the level for which potential temperature is desired

        Returns
        -------
        Theta (C) at the given pressure : number, numpy array

        '''
    # Note: numpy's interpolation routine expects the interpolation
    # routine to be in ascending order. Because pressure decreases in the
    # vertical, we must reverse the order of the two arrays to satisfy
    # this requirement.
    return _interp_cached(prof, 'logp_theta', p, prof.logp[::-1], prof.theta[::-1], log_query=True)

def wetbulb(prof, p):
    '''
        Interpolates the given data to calculate a wetbulb temperature at a given pressure

        Parameters
        ----------
        prof : profile object
        Profile object
        p : number, numpy array
        Pressure (hPa) of the level for which wetbulb temperature is desired

        Returns
        -------
        Wetbulb temperature (C) at the given pressure : number, numpy array

        '''
    # Note: numpy's interpolation routine expects the interpolation
    # routine to be in ascending order. Because pressure decreases in the
    # vertical, we must reverse the order of the two arrays to satisfy
    # this requirement.
    return _interp_cached(prof, 'logp_wetbulb', p, prof.logp[::-1], prof.wetbulb[::-1], log_query=True)

def dwpt(prof, p):
    '''
    Interpolates the given data to calculate a dew point temperature
    at a given pressure

    Parameters
    ----------
    prof : profile object
        Profile object
    p : number, numpy array
        Pressure (hPa) of the level for which dew point temperature is desired

    Returns
    -------
    Dew point tmperature (C) at the given pressure : number, numpy array

    '''
    # Note: numpy's interpolation routine expects the interpolation
    # routine to be in ascending order. Because pressure decreases in the
    # vertical, we must reverse the order of the two arrays to satisfy
    # this requirement.
    return _interp_cached(prof, 'logp_dwpc', p, prof.logp[::-1], prof.dwpc[::-1], log_query=True)


def vtmp(prof, p):
    '''
    Interpolates the given data to calculate a virtual temperature
    at a given pressure

    Parameters
    ----------
    prof : profile object
        Profile object
    p : number, numpy array
        Pressure (hPa) of the level for which virtual temperature is desired

    Returns
    -------
    Virtual tmperature (C) at the given pressure : number, numpy array

    '''
    return _interp_cached(prof, 'logp_vtmp', p, prof.logp[::-1], prof.vtmp[::-1], log_query=True)


def components(prof, p):
    '''
    Interpolates the given data to calculate the U and V components at a
    given pressure

    Parameters
    ----------
    prof : profile object
        Profile object
    p : number, numpy array
        Pressure (hPa) of a level

    Returns
    -------
    U and V components at the given pressure (kts) : number, numpy array
    '''
    # Note: numpy's interpolation routine expects the interpolation
    # routine to be in ascending order. Because pressure decreases in the
    # vertical, we must reverse the order of the two arrays to satisfy
    # this requirement.
    if prof.wdir.count() == 0:
        # JTS - Fixed a bug where clicking "Interpolate Focused Profile" throws an error for NUCAPS.
        return ma.masked_where(ma.ones(np.shape(p)), p), ma.masked_where(ma.ones(np.shape(p)), p)
    U = _interp_cached(prof, 'logp_u', p, prof.logp[::-1], prof.u[::-1], log_query=True)
    V = _interp_cached(prof, 'logp_v', p, prof.logp[::-1], prof.v[::-1], log_query=True)
    return U, V


def vec(prof, p):
    '''
    Interpolates the given data to calculate the wind direction and speed
    at a given pressure

    Parameters
    ----------
    p : number, numpy array
        Pressure (hPa) of a level
    prof : profile object
        Profile object

    Returns
    -------
    Wind direction (degrees) and magnitude (kts) at the given pressure : number, numpy array
    '''
    U, V = components(prof, p)
    return utils.comp2vec(U, V)


def to_agl(prof, h):
    '''
    Convert a height from mean sea-level (MSL) to above ground-level (AGL)

    Parameters
    ----------
    h : number, numpy array
        Height of a level
    prof : profile object
        Profile object

    Returns
    -------
    Converted height (m AGL) : number, numpy array

    '''
    return h - prof.hght[prof.sfc]


def to_msl(prof, h):
    '''
    Convert a height from above ground-level (AGL) to mean sea-level (MSL)

    Parameters
    ----------
    h : number, numpy array
        Height of a level
    prof : profile object
        Profile object

    Returns
    -------
    Converted height (m MSL) : number, numpy array

    '''
    return h + prof.hght[prof.sfc]


def generic_interp_hght(h, hght, field, log=False):
    '''
    Generic interpolation routine

    Parameters
    ----------
    h : number, numpy array
        Height (m) of the level for which pressure is desired
    hght : numpy array
        The array of heights
    field : numpy array
        The variable which is being interpolated
    log : bool
        Flag to determine whether the 'field' variable is in log10 space

    Returns
    -------
    Value of the 'field' variable at the given height : number, numpy array

    '''
    if field.count() == 0 or hght.count() == 0:
        return ma.masked_where(ma.ones(np.shape(h)), h) # JTS

    # Avoid repeated MaskedArray.__getitem__/__array_finalize__ overhead (which
    # dominates cost when this is called scalar-at-a-time hundreds of times
    # per sounding, e.g. from parcelx's LFC/EL bisection and layer lookups).
    # ma.getmaskarray() is a single fast call instead of two allocations, and
    # indexing the raw .data ndarray skips MaskedArray's per-getitem machinery.
    not_masked = ~ma.getmaskarray(hght) & ~ma.getmaskarray(field)

    hght_data = np.asarray(hght)
    field_data = np.asarray(field)

    field_intrp = np.interp(h, hght_data[not_masked], field_data[not_masked], left=np.nan, right=np.nan)

    if hasattr(h, 'shape') and h.shape == tuple():
        h = h[()]

    # Another bug fix: np.interp() returns masked values as nan. We want ma.masked, dangit!
    field_intrp = ma.where(np.isnan(field_intrp), ma.masked, field_intrp)

    # ma.where() returns a 0-d array when the arguments are floats, which confuses subsequent code.
    if hasattr(field_intrp, 'shape') and field_intrp.shape == tuple():
        field_intrp = field_intrp[()]

    if log:
        return 10 ** field_intrp
    else:
        return field_intrp

def generic_interp_pres(p, pres, field):
    '''
    Generic interpolation routine

    Parameters
    ----------
    p : number, numpy array
        Pressure (hPa) of the level for which the field variable is desired
    pres : numpy array
        The array of pressure
    field : numpy array
        The variable which is being interpolated
    log : bool
        Flag to determine whether the 'field' variable is in log10 space

    Returns
    -------
    Value of the 'field' variable at the given pressure : number, numpy array

    '''
    if field.count() == 0 or pres.count() == 0:
        return ma.masked_where(ma.ones(np.shape(p)), p) # JTS

    not_masked = ~ma.getmaskarray(pres) & ~ma.getmaskarray(field)

    pres_data = np.asarray(pres)
    field_data = np.asarray(field)

    field_intrp = np.interp(p, pres_data[not_masked], field_data[not_masked], left=np.nan, right=np.nan)

    if hasattr(p, 'shape') and p.shape == tuple():
        p = p[()]

    # Another bug fix: np.interp() returns masked values as nan. We want ma.masked, dangit!
    field_intrp = ma.where(np.isnan(field_intrp), ma.masked, field_intrp)

    # ma.where() returns a 0-d array when the arguments are floats, which confuses subsequent code.
    if hasattr(field_intrp, 'shape') and field_intrp.shape == tuple():
        field_intrp = field_intrp[()]

    return field_intrp
