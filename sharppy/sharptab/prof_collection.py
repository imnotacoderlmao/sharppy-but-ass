
from __future__ import absolute_import

import sharppy.sharptab.profile as profile
import sharppy.sharptab.interp as interp
from sutils.frozenutils import Process, Queue
import platform
import numpy as np

def _ask_interp_options():
    """
    Pop up a small Tkinter window that asks the user how they want to
    interpolate the profile.

    Returns a dict:
        {
            'mode':  'fill_missing' | 'pressure_grid',
            'dp':    float  (negative step, e.g. -25),   # pressure_grid only
            'pbot':  float | None,                        # pressure_grid only
            'ptop':  float | None,                        # pressure_grid only
        }
    this returns None if the user cancelled.
    """
    import tkinter as tk
    from tkinter import ttk, messagebox

    result = {}

    root = tk.Tk()
    root.title("Interpolation Options")
    root.resizable(False, False)

    body = tk.Frame(root, padx=18, pady=12, bg="#f4f7fb")
    body.pack(fill=tk.BOTH, expand=True)

    mode_var = tk.StringVar(value="fill_missing")

    def _toggle(*_):
        state = "normal" if mode_var.get() == "pressure_grid" else "disabled"
        for w in grid_widgets:
            w.configure(state=state)

    tk.Label(body, text="Interpolation mode:", bg="#f4f7fb", anchor="w").grid(
        row=0, column=0, columnspan=2, sticky="w", pady=(0, 4)
    )
    rb1 = tk.Radiobutton(
        body,
        text="Fill missing values only  (linear interp between valid points)",
        variable=mode_var,
        value="fill_missing",
        bg="#f4f7fb",
        command=_toggle,
    )
    rb1.grid(row=1, column=0, columnspan=2, sticky="w")

    rb2 = tk.Radiobutton(
        body,
        text="Interpolate to regular pressure grid",
        variable=mode_var,
        value="pressure_grid",
        bg="#f4f7fb",
        command=_toggle,
    )
    rb2.grid(row=2, column=0, columnspan=2, sticky="w", pady=(2, 8))

    # Pressure-grid options 
    sep = ttk.Separator(body, orient="horizontal")
    sep.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(0, 8))

    tk.Label(body, text="Pressure step (hPa, negative):", bg="#f4f7fb").grid(
        row=4, column=0, sticky="w"
    )
    dp_var = tk.StringVar(value="-25")
    dp_entry = tk.Entry(body, textvariable=dp_var, width=10)
    dp_entry.grid(row=4, column=1, sticky="w", padx=(6, 0))

    tk.Label(body, text="Bottom pressure (hPa, blank = sfc):", bg="#f4f7fb").grid(
        row=5, column=0, sticky="w", pady=(4, 0)
    )
    pbot_var = tk.StringVar(value="")
    pbot_entry = tk.Entry(body, textvariable=pbot_var, width=10)
    pbot_entry.grid(row=5, column=1, sticky="w", padx=(6, 0))

    tk.Label(body, text="Top pressure    (hPa, blank = top):", bg="#f4f7fb").grid(
        row=6, column=0, sticky="w", pady=(4, 0)
    )
    ptop_var = tk.StringVar(value="")
    ptop_entry = tk.Entry(body, textvariable=ptop_var, width=10)
    ptop_entry.grid(row=6, column=1, sticky="w", padx=(6, 0))

    grid_widgets = [dp_entry, pbot_entry, ptop_entry]
    _toggle()  # set initial disabled state

    # Buttons
    btn_frame = tk.Frame(root, bg="#f4f7fb", pady=8)
    btn_frame.pack(fill=tk.X)

    def _ok():
        mode = mode_var.get()
        result["mode"] = mode
        if mode == "pressure_grid":
            try:
                dp = float(dp_var.get())
                if dp >= 0:
                    raise ValueError
            except ValueError:
                messagebox.showerror(
                    "Invalid input", "Pressure step must be a negative number."
                )
                return
            result["dp"] = dp
            result["pbot"] = float(pbot_var.get()) if pbot_var.get().strip() else None
            result["ptop"] = float(ptop_var.get()) if ptop_var.get().strip() else None
        root.destroy()

    def _cancel():
        result.clear()
        root.destroy()

    ok_btn = tk.Button(
        btn_frame,
        text="  OK  ",
        command=_ok,
        bg="#1c6ea4",
        fg="white",
        relief="flat",
        padx=8,
        pady=4,
    )
    ok_btn.pack(side=tk.RIGHT, padx=(0, 12))

    cancel_btn = tk.Button(
        btn_frame,
        text="Cancel",
        command=_cancel,
        bg="#d0d0d0",
        relief="flat",
        padx=8,
        pady=4,
    )
    cancel_btn.pack(side=tk.RIGHT, padx=(0, 4))

    root.mainloop()
    return result if result else None

def doCopy(target_type, prof, idx, pipe):
    pipe.put((target_type.copy(prof), idx))
    
class ProfCollection(object):
    """
    ProfCollection: A class to keep track of profiles from a single data source. Handles time switching, ensemble member switching,
        and modifications to profiles.
    """
    def __init__(self, profiles, dates, target_type=profile.ConvectiveProfile, **kwargs):
        """
        Initialize the collection.
        profiles:   A dictionary of lists of profiles.  The keys of the dictionary are the ensemble member names, the
            values are lists of profiles for those members over time.
        dates:      A list of datetime objects corresponding to the times for each element of the lists in profiles.
        target_type:    The type to copy the profiles to when requested. Default is a ConvectiveProfile.
        **kwargs:   Metadata for the profile.
        """
        self._profs = profiles
        self._dates = dates
        self._meta = kwargs
        self._target_type = target_type
        self._highlight = kwargs.get('highlight', list(profiles.keys())[0])
        self._prof_idx = 0
        self._analog_date = None

        self._mod_therm = [ False for d in self._dates ]
        self._mod_wind = [ False for d in self._dates ]
        self._interp = [ False for d in self._dates ]

        self._orig_profs = {}
        self._interp_profs = {}
        self._async = None
        self._cancel_copy = False
        self._procs = []

    def subset(self, idxs):
        """
        Subset the profile collection over time.
        idxs:   The time indices to include in the subsetted collection.
        """
        def extract_profile_indexes(prof):
            prof_indexed = []
            for idx in idxs:
                try:
                    prof_indexed.append(prof[idx])
                except IndexError:
                    pass

            return prof_indexed

        profiles = dict( (mem, extract_profile_indexes(prof)) for mem, prof in self._profs.items() )
        dates = [ self._dates[idx] for idx in idxs ]
        return ProfCollection(profiles, dates, highlight=self._highlight, **self._meta)

    def _backgroundCopy(self, member, max_procs=2):
        """
        Copies the profile objects in the background while the user can continue to do things.
        This upgrades the project object types from Profile to ConvectiveProfile via the
        _target_type variable.
        
        member:     the key indicating a specific member
        max_procs:  max number of processors to perform this action
        """
        pipe = Queue(max_procs)

        for idx, prof in enumerate(self._profs[member]):
            proc = Process(target=doCopy, args=(self._target_type, prof, idx, pipe))
            proc.start()

            self._procs.append(proc)
            
            if (idx % max_procs) == 0 or idx == len(self._profs[member]) - 1:
                for proc in self._procs:

                    if platform.system() != "Windows":
                        # Windows hangs here for some reason, but runs fine without it.
                        proc.join()
                        
                    prof, copy_idx  = pipe.get()
                    self._profs[member][copy_idx] = prof
                    
                self._procs = []
        return

    def setAsync(self, async_obj):
        """
        Start an asynchronous process to load objects of type 'target_type' in the background.
        Used to upgrade the Profile objects to ConvectiveProfile objects in the background

        async:  An AsyncThreads instance.
        """
        self._async = async_obj
        self._async.post(self._backgroundCopy, None, self._highlight)

    def cancelCopy(self):
        """
        Terminates any threads that are running in the background.
        """
        for proc in self._procs:
            proc.terminate()
        if self._async is not None:
            self._async.clearQueue()

    def getMeta(self, key, index=False):
        """
        Returns metadata about the profile.
        key:    What metadata to return.
        index [optional]: If true, treat the metadata as an array with the same length as dates passed in the constructor.
            Returns value of that array at this time index..
        """
        meta = self._meta[key]
        if index:
            meta = meta[self._prof_idx]
        return meta

    def getCurrentDate(self):
        """
        Returns the current date in the profile object
        """
        if not self.hasCurrentProf():
            return

        return self._dates[self._prof_idx]

    def getHighlightedProf(self):
        """
        Returns which profile is highlighted.
        """
        if not self.hasCurrentProf():
            return

        cur_prof = self._profs[self._highlight][self._prof_idx]
        # If the currently selected profile is not of the target_type (e.g., ConvectiveProfile), then
        # then upgrade it via the copy function.
        if type(cur_prof) != self._target_type:
            self._profs[self._highlight][self._prof_idx] = self._target_type.copy(cur_prof)
        return self._profs[self._highlight][self._prof_idx]

    def getCurrentProfs(self):
        """
        Returns the profiles at the current time.
        """
        if not self.hasCurrentProf():
            return {}

        for mem, profs in self._profs.items():
            # Copy the profiles on the fly
            try:
                cur_prof = profs[self._prof_idx]
            except IndexError:
                continue
            else:
                if mem == self._highlight and type(cur_prof) != self._target_type:
                    self._profs[mem][self._prof_idx] = self._target_type.copy(cur_prof)
                elif type(cur_prof) not in [ profile.BasicProfile, self._target_type ]:
                    self._profs[mem][self._prof_idx] = profile.BasicProfile.copy(cur_prof)

        profs = dict( (mem, profs[self._prof_idx]) for mem, profs in self._profs.items() if len(profs) > self._prof_idx ) 
        return profs

    def getAnalogDate(self):
        """
        If this is an analog, return the date of the analog. Otherwise, returns None.
        """
        return self._analog_date

    def isModified(self):
        """
        Returns True if the profiles at the current time have been modified.  Returns False otherwise.
        """
        if not self.hasCurrentProf():
            return False
        return self._mod_therm[self._prof_idx] or self._mod_wind[self._prof_idx]

    def isInterpolated(self):
        """
        Returns True if the profiles at the current time have been modified.  Returns False otherwise.
        """
        if not self.hasCurrentProf():
            return False
        return self._interp[self._prof_idx]

    def isEnsemble(self):
        """
        Returns True if this collection has multiple ensemble members. Otherwise, returns False.
        """
        return len(list(self._profs.keys())) > 1

    def hasCurrentProf(self):
        """
        Returns True if the collection has a profile at the current time. Otherwise, returns False.
        """
        return self._prof_idx >= 0

    def hasMeta(self, key):
        """
        Returns True if the collection has metadata corresponding to 'key'. Otherwise returns False.
        """
        return key in self._meta

    def setMeta(self, key, value):
        """
        Sets the metadata 'key' to 'value'.
        """
        self._meta[key] = value

    def setHighlightedMember(self, member_name):
        """
        Sets the highlighted ensemble member to be 'member_name'.
        """
        self._highlight = member_name

    def getHighlightedMemberName(self):
        """
        Gets the name of the member that is currently highlighted.
        """
        return self._highlight

    def setCurrentDate(self, cur_dt):
        """
        Sets the current date to be 'cur_dt'.
        cur_dt:     A datetime object specifiying which date to set it to.
        """
        try:
            self._prof_idx = self._dates.index(cur_dt)
        except ValueError:
            pass

    def setAnalogToDate(self, analog_to_date):
        """
        Specify that this collection represents an analog; the date is set to 'analog_to_date', and the 
            analog date is set to the former date.
        analog_to_date: A datetime object that specifies the date to which this collection is an analog.
        """
        self._analog_date = self._dates[0]
        self._dates = [ analog_to_date ]

    def advanceTime(self, direction):
        """
        Advance time in a direction specified by 'direction'. Returns a datetime object containing the new time.
        direction:  An integer (ether 1 or -1) specifying which direction to move time in. 1 moves time forward,
            -1 moves time backward.
        """
        length = len(self._dates)
        if direction > 0 and self._prof_idx == length - 1:
            self._prof_idx = 0
        elif direction < 0 and self._prof_idx == 0:
            self._prof_idx = length - 1
        else:
            self._prof_idx += direction
        return self._dates[self._prof_idx]

    def advanceHighlight(self, direction):
        """
        Change which member is highlighted.
        direction:  An integer (either 1 or -1) specifying which direction to go in the list. The list is in
            alphabetical order, so the members will be gone through in that order. 
        """
        mem_names = sorted(self._profs.keys())
        high_idx = mem_names.index(self._highlight)
        length = len(mem_names)
        
        def doAdvance(adv_idx):
            if direction > 0 and adv_idx == length - 1:
                adv_idx = 0
            elif direction < 0 and adv_idx == 0:
                adv_idx = length - 1
            else:
                adv_idx = adv_idx + direction
            return adv_idx

        adv_idx = doAdvance(high_idx)
        highlight = mem_names[adv_idx]
        while len(self._profs[highlight]) <= self._prof_idx:
            adv_idx = doAdvance(adv_idx)
            highlight = mem_names[adv_idx]
        self._highlight = highlight

    def defineUserParcel(self, parcel):
        """
        Defines a custom parcel for the current profile.
        parcel:     A parcel object to use as the custom parcel.
        """
        if self.hasCurrentProf():
            self._profs[self._highlight][self._prof_idx].usrpcl = parcel

    def modify(self, idx, **kwargs):
        """
        Modify the profile at the current time.
        idx:    The vertical index to modify
        **kwargs:   The variables to modify ('tmpc', 'dwpc', 'u', or 'v')
        
        TODO: Allow modification of layers.  Could be that idx is -999 for layer
              and kwargs passes information about the layers to be modified.
        """
        if self.isEnsemble():
            raise ValueError("Can't modify ensemble profiles")

        prof = self._profs[self._highlight][self._prof_idx]

        # Save original, if one hasn't already been saved
        if self._prof_idx not in self._orig_profs:
            self._orig_profs[self._prof_idx] = prof

        cls = type(prof)
        # Copy the variables to be modified
        prof_vars = dict( (k, prof.__dict__[k].copy()) for k in kwargs.keys() if k != 'idx_range')
        
        if idx != -999:
            # Do the modification
            for var, val in kwargs.items():
                prof_vars[var][idx] = val
        else:
            idx = kwargs.get('idx_range')
            for key in prof_vars.keys():
                prof_vars[key] = kwargs.get(key)
 
        # Make a copy of the profile object with the newly modified variables inserted.
        self._profs[self._highlight][self._prof_idx] = cls.copy(prof, **prof_vars)

        # Update bookkeeping
        if 'tmpc' in kwargs or 'dwpc' in kwargs:
            self._mod_therm[self._prof_idx] = True

        if 'u' in kwargs or 'v' in kwargs or 'wdir' in kwargs or 'wspd' in kwargs:
            self._mod_wind[self._prof_idx] = True

    def modifyStormMotion(self, deviant, vec_u, vec_v):
        if deviant == 'left':
            self._profs[self._highlight][self._prof_idx].set_srleft(vec_u, vec_v)
        elif deviant == 'right':
            self._profs[self._highlight][self._prof_idx].set_srright(vec_u, vec_v)

    def resetStormMotion(self):
        self._profs[self._highlight][self._prof_idx].reset_srm()

    def interp(self, dp=-25, mode=None, pbot=None, ptop=None, ask=True):
        """
        Interpolate / gap-fill the current profile.

        Parameters
        ----------
        dp : float, optional
            Pressure step (negative) used when mode='pressure_grid'.
            Default -25 hPa.
        mode : str or None
            'fill_missing'  - fill -9999/-9990 gaps by linear interpolation
                              between the nearest valid neighbours.
            'pressure_grid' - re-sample the profile onto a regular pressure
                              grid with spacing *dp* between *pbot* and *ptop*.
            None (default)  - show a Tkinter dialog to ask the user (when
                              ask=True) or fall back to 'pressure_grid'
                              behaviour for backward compatibility.
        pbot : float or None
            Bottom of the new pressure grid (hPa). None → use sfc.
        ptop : float or None
            Top of the new pressure grid (hPa). None → use top of sounding.
        ask : bool
            If True and *mode* is None, open the Tkinter dialog.
        """

        if self.isEnsemble():
            raise ValueError("Cannot interpolate ensemble profiles.")

        if mode is None:
            if ask:
                opts = _ask_interp_options()
                if opts is None:
                    return  # user cancelled
                mode = opts['mode']
                if mode == 'pressure_grid':
                    dp   = opts.get('dp',   dp)
                    pbot = opts.get('pbot', pbot)
                    ptop = opts.get('ptop', ptop)
            else:
                mode = 'pressure_grid'  # legacy default

        prof = self._profs[self._highlight][self._prof_idx]

        if self._prof_idx not in self._orig_profs:
            self._orig_profs[self._prof_idx] = prof

        cls = type(prof)

        if mode == 'fill_missing':
            pres = prof.pres  # keep original pressure levels unchanged

            prof_vars = {'pres': pres}
            prof_vars['tmpc'] = interp.temp(prof, pres)
            prof_vars['dwpc'] = interp.dwpt(prof, pres)
            prof_vars['hght'] = interp.hght(prof, pres)

            if prof.omeg.all() is not np.ma.masked:
                prof_vars['omeg'] = interp.omeg(prof, pres)
            else:
                prof_vars['omeg'] = np.ma.masked_array(
                    pres, mask=np.ones(len(pres), dtype=int)
                )

            u, v = interp.components(prof, pres)
            prof_vars['u'] = u
            prof_vars['v'] = v

            interp_prof = cls.copy(prof, **prof_vars)

        elif mode == 'pressure_grid':
            p_sfc = prof.pres[prof.sfc]
            p_top = prof.pres[prof.top]

            if pbot is not None:
                p_sfc = float(pbot)
            if ptop is not None:
                p_top = float(ptop)

            dp = abs(float(dp))

            # Use linspace instead of arange to avoid float rounding errors that can silently drop the last pressure level.
            n_levels = int(round((p_sfc - p_top) / dp)) + 1
            new_pres = np.linspace(p_sfc, p_top, n_levels)

            prof_vars = {'pres': new_pres}
            prof_vars['tmpc'] = interp.temp(prof, new_pres)
            prof_vars['dwpc'] = interp.dwpt(prof, new_pres)
            prof_vars['hght'] = interp.hght(prof, new_pres)

            if prof.omeg.all() is not np.ma.masked:
                prof_vars['omeg'] = interp.omeg(prof, new_pres)
            else:
                prof_vars['omeg'] = np.ma.masked_array(
                    new_pres, mask=np.ones(len(new_pres), dtype=int)
                )

            u, v = interp.components(prof, new_pres)
            prof_vars['u'] = u
            prof_vars['v'] = v

            interp_prof = cls.copy(prof, **prof_vars)

        else:
            raise ValueError(f"Unknown interpolation mode: {mode!r}. "
                             "Use 'fill_missing' or 'pressure_grid'.")

        # ── 4. Commit ────────────────────────────────────────────────────────
        self._profs[self._highlight][self._prof_idx] = interp_prof

        if self._prof_idx not in self._interp_profs:
            self._interp_profs[self._prof_idx] = interp_prof

        self._interp[self._prof_idx] = True


    def resetModification(self, *args):
        """
        Reset the profile to its original state.
        *args:  The variables to reset ('tmpc', 'dwpc', 'u', or 'v').
        """
        if not self._prof_idx in self._orig_profs:
            return

        if self._interp[self._prof_idx]:
            orig_prof = self._interp_profs[self._prof_idx]
        else:
            orig_prof = self._orig_profs[self._prof_idx]

        prof = self._profs[self._highlight][self._prof_idx]
        cls = type(prof)

        # Get the original variables
        prof_vars = dict( (k, orig_prof.__dict__[k]) for k in args )

        # Make a copy of the profile object with the original variables inserted
        self._profs[self._highlight][self._prof_idx] = cls.copy(prof, **prof_vars)

        # Update bookkeeping
        if 'tmpc' in args or 'dwpc' in args:
            self._mod_therm[self._prof_idx] = False

        if 'u' in args or 'v' in args or 'wdir' in args or 'wspd' in args:
            self._mod_wind[self._prof_idx] = False

        if not self.isModified() and not self.isInterpolated():
            del self._orig_profs[self._prof_idx]

    def resetInterpolation(self):
        if not self._prof_idx in self._interp_profs:
            return

        self._profs[self._highlight][self._prof_idx] = self._orig_profs[self._prof_idx]

        prof = self._profs[self._highlight][self._prof_idx]
#       print dict( (k, prof.__dict__[k].shape[0]) for k in [ 'pres', 'hght', 'tmpc', 'dwpc', 'u', 'v' ])

        del self._orig_profs[self._prof_idx]
        del self._interp_profs[self._prof_idx]

        self._mod_wind[self._prof_idx] = False
        self._mod_therm[self._prof_idx] = False
        self._interp[self._prof_idx] = False
