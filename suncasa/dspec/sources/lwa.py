import os
import h5py
import numpy as np
import datetime
from astropy.time import Time
import itertools
from astropy import units as u
import pandas as pd
from astropy.coordinates import SkyCoord, EarthLocation, get_body, AltAz

def rebin1d(arr, new_len):
    shape = (new_len, len(arr) // new_len)
    return arr.reshape(shape).mean(1)


def rebin2d(arr, new_shape):
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)


def _read_hdf5_chunked(ds, ti0, ti1, fi0, fi1, timebin, chunk_size=512):
    """Read an HDF5 dataset in time chunks and rebin on-the-fly to avoid
    loading the full time axis into memory at once.

    Returns a float32 array of shape (nt_new, nf) where nt_new = n_time // timebin.
    Any trailing samples that don't fill a complete timebin are discarded.
    """
    nf = fi1 - fi0
    n_time = ti1 - ti0
    nt_new = n_time // timebin          # number of output time samples
    n_time_used = nt_new * timebin      # trim to exact multiple

    # chunk_size must be a multiple of timebin so each chunk rebins cleanly
    chunk_size = max(timebin, (chunk_size // timebin) * timebin)

    out = np.empty((nt_new, nf), dtype=np.float32)
    out_row = 0

    for start in range(0, n_time_used, chunk_size):
        end = min(start + chunk_size, n_time_used)
        block = ds[ti0 + start: ti0 + end, fi0:fi1].astype(np.float32)
        n_block = end - start
        n_bins = n_block // timebin
        block = block[:n_bins * timebin].reshape(n_bins, timebin, nf).mean(axis=1)
        out[out_row: out_row + n_bins] = block
        out_row += n_bins

    return out


def timestamp_to_mjd(times):
    # This is from Ivey Davis's BeamTools.py
    t_flat = np.array(list(itertools.chain(*times)))
    ts_inds = np.linspace(0, len(times)-1, len(times), dtype  = int)*2
    other_inds = ts_inds + 1
    ts = Time(t_flat[ts_inds],format = 'unix') + t_flat[other_inds]* u.s
    ts = ts.mjd
    return ts

def apply_xhand_delay_UV(specU,specV,frequencies, xhand_delay): # delay in ns, freq in Hz

    xhand_phase = 2*np.pi*frequencies*xhand_delay/1e9
    outU =  specU* np.cos(xhand_phase) + specV*np.sin(xhand_phase)
    outV = -specU* np.sin(xhand_phase) + specV*np.cos(xhand_phase)

    return outU, outV


def read_data(filename, stokes='I', timerange=[], freqrange=[], timebin=1, freqbin=1, verbose=True, 
            flux_factor_file=None, bkg_file=None,  do_pb_correction=False, 
            flux_factor_calfac_x = None, flux_factor_calfac_y = None, bkg_flux_arr = None,
            xhand_delay=None):
    '''
    :param filename: name of the OVRO-LWA hdf5 beamforming file; 
              This can be a string (single file) or a list of strings (multiple files)
    :param stokes: currently supporting 'XX', 'YY', 'I', 'Q', 'U', 'V', 'IV' 
    :param timerange: list of [start_time, end_time], start_time and end_time should be recognized by astropy.time.Time
            e.g., ['2023-09-22T18:00', '2023-09-22T18:10']
    :param freqrange: list of [start_frequency, end_frequency] in MHz. Example: [23, 82]
    :param timebin: number to bin in time
    :param freqbin: number to bin in frequency
    :param verbose: if True, print extra information
    :param flux_factor_file: Path to the csv file that contains the flux correction factors
    :param bkg_file: Path to the csv file that contains the raw background (off-Sun scan) measurements (before scaling); 
            currently it only contains Stokes I.
    :param do_pb_correction: if True, apply primary beam correction. Currently only use the analytical form in Stokes I only.
    :param flux_factor_calfac_x: user input correction factor for the X polarization
    :param flux_factor_calfac_y: user input correction factor for the Y polarization
    :param bkg_flux_arr: user input background flux in Jy
    :param xhand_delay: user input crosshand delay in nanoseconds, applied if not None and stokes.upper() == 'IV'
    '''
    # Check the input filename
    if type(filename) == str:
        filename = [filename]

    obs = EarthLocation.of_site('ovro')
    filelist = []
    for ll in filename:
        if not os.path.exists(ll):
            print('{} does not exist. Skip this one.'.format(ll))
        else:
            filelist.append(ll)

    if not filelist:
        print('I cannot find any file. Abort.')
        return False
    else:
        filelist.sort()
    
    n0 = 0 # this is the index for the first file being read successfully
    firstset_read = False 
    for n, file in enumerate(filelist):
        if verbose:
            print('Processing {0:d} of {1:d} files'.format(n+1, len(filelist)))
        try:
            data = h5py.File(file, 'r',swmr=True)
            freqs = data['Observation1']['Tuning1']['freq'][:]
            ts = data['Observation1']['time'][:]

            # ts dtype may be structured with named fields (int, frac) or plain 2-D.
            # Extract the integer unix-seconds column for safe scalar comparison.
            if ts.dtype.names and 'int' in ts.dtype.names:
                ts_int = ts['int']
            else:
                ts_int = np.asarray(ts)[:, 0]
            # Strip trailing sentinel rows (int==0) written by the recorder.
            valid_mask = ts_int > 0
            ts = ts[valid_mask]
            ts_int = ts_int[valid_mask]
            if len(ts) <= 1:
                raise ValueError('Too few valid time stamps ({}).'.format(len(ts)))
            if not ts_int[-1] > ts_int[0]:
                raise ValueError('Time stamps are not monotonically increasing.')

            # The following line works the same way as timestamp_to_mjd(), but a bit too slow
            # times_mjd = np.array([(Time(t[0], format='unix') + TimeDelta(t[1], format='sec')).mjd for t in ts])
            times_mjd = timestamp_to_mjd(ts)
            idx0, = np.where(times_mjd > 50000.) # filter out those prior to 1995 (obviously wrong for OVRO-LWA)

        except Exception as e:
            print('Cannot read {0:s}: {1:s}. Skip this file.'.format(file, str(e)))
            continue
        
        # read the flux factors file if provided
        if not (flux_factor_file is None): 
            try:
                out = pd.read_csv(flux_factor_file)
                calfac_x = np.array(out['calfac_x'])
                calfac_y = np.array(out['calfac_y'])
            except:
                print('Failed in reading the flux factor csv file. Setting correction factors to unity.')
        else:
            print('Flux factor csv file does not exist. Setting correction factors to unity.')
            calfac_x = np.ones_like(freqs)
            calfac_y = np.ones_like(freqs)

        if not (flux_factor_calfac_x is None) and not (flux_factor_calfac_y is None):
            # user input correction factor
            calfac_x = calfac_x*flux_factor_calfac_x
            calfac_y = calfac_y*flux_factor_calfac_y
        
        if not (bkg_flux_arr is None):
            # add the user input background flux
            bkg_flux += bkg_flux_arr

        # read background flux file if provided
        if not (bkg_file is None): 
            try:
                out = pd.read_csv(bkg_file)
                bkg_flux = out['bkg_flux']
                print('Using the provided raw aackground flux csv file.')
            except:
                print('Failed in reading the background flux csv file. Setting background flux to zero.')
        else:
            print('No background csv file provided. Setting background flux to zero.')
            bkg_flux = np.zeros_like(freqs)

        if verbose:
            print('Data time range is from {0:s} to {1:s}'.format(Time(times_mjd[idx0][0], format='mjd').isot, 
                Time(times_mjd[idx0][-1], format='mjd').isot))
            print('Data has {0:d} time stamps and {1:d} frequency channels'.format(len(times_mjd[idx0]), len(freqs)))

        # Select time range
        if len(timerange) > 0:
            try:
                timerange_obj = Time(timerange)
                # Take the larger value of the supplied start time and the first time stamp of the data
                t0 = max(timerange_obj[0].mjd, min(times_mjd[idx0]))
                # Take the smaller value of the supplied end time and the last time stamp of the data
                t1 = min(timerange_obj[1].mjd, max(times_mjd[idx0]))
                ti0 = np.argmin(np.abs(times_mjd - t0))
                ti1 = np.argmin(np.abs(times_mjd - t1)) 
                if ti1 - ti0 < timebin:
                    print('Selected number of time samples {0:d} is less than the timebin {1:d}. Skip this file.'.format(ti1-ti0, timebin))
                    if not firstset_read:
                        n0 += 1
                    continue
                if verbose:
                    print('Selected time range is from {0:s} to {1:s}'.format(Time(times_mjd[ti0], format='mjd').isot, 
                                                                              Time(times_mjd[ti1], format='mjd').isot))
            except:
                print('timerange not parsed correctly. Use the full range in the data.')
                ti0 = 0
                ti1 = len(times_mjd) 
        else:
            ti0=0
            ti1=len(times_mjd)

        firstset_read = True
        times_mjd = times_mjd[ti0:ti1] 

        # Select frequency range
        if len(freqrange) > 0:
            if type(freqrange) == list:
                try:
                    f0 = freqrange[0]
                    f1 = freqrange[1]
                    if f0 > 100. or f1 > 100.:
                        # I am assuming input frequency range is in Hz
                        print('Input frequency range is greater than 100. Assuming unit in Hz.')
                    else:
                        # I am assuming input frequency range is in MHz
                        print('Input frequency range is less than 100. Assuming unit in MHz.')
                        f0 *= 1e6
                        f1 *= 1e6
                    fi0 = np.argmin(np.abs(freqs - f0))
                    fi1 = np.argmin(np.abs(freqs - f1)) 
                except:
                    print('freqrange not parsed correctly. Use the full range.')
                    fi0 = 0
                    fi1 = len(freqs) 
        else:
            fi0=0
            fi1=len(freqs)

        freqs = freqs[fi0:fi1] 

        # select stokes
        stokes_valid = ['XX', 'YY', 'I', 'Q', 'U', 'V', 'IV']
        if verbose:
            print('Reading dynamic spectrum for stokes {0:s}'.format(stokes))

        if stokes not in stokes_valid:
            raise Exception("Provided Stokes {0:s} is not in 'XX, YY, RR, LL, I, Q, U, V'".format(stokes))

        # --- calibration factor slices (small 1-D arrays, fine to keep in RAM) ---
        cx = calfac_x[fi0:fi1].astype(np.float32)   # shape (nf,)
        cy = calfac_y[fi0:fi1].astype(np.float32)
        bkg = bkg_flux[fi0:fi1].astype(np.float32)
        cavg = (cx + cy) / 2.

        try:
            ds = data['Observation1']['Tuning1']
            if stokes.upper() == 'XX':
                spec_new = _read_hdf5_chunked(ds['XX'], ti0, ti1, fi0, fi1, timebin) / cx[None, :]
                stokes_out = ['XX']
                npol = 1
            elif stokes.upper() == 'YY':
                spec_new = _read_hdf5_chunked(ds['YY'], ti0, ti1, fi0, fi1, timebin) / cy[None, :]
                stokes_out = ['YY']
                npol = 1
            elif stokes.upper() == 'I' or stokes.upper() == 'IV':
                xx_rb = _read_hdf5_chunked(ds['XX'], ti0, ti1, fi0, fi1, timebin)
                yy_rb = _read_hdf5_chunked(ds['YY'], ti0, ti1, fi0, fi1, timebin)
                spec_I = xx_rb / cx[None, :] / 2. + yy_rb / cy[None, :] / 2. - bkg[None, :] / cavg[None, :]
                del xx_rb, yy_rb
                if verbose:
                    print('Median of the subtracted background flux (Jy)', np.median(bkg / cavg))
                    print('RMS of the subtracted background flux (Jy)', np.std(bkg / cavg))
                if do_pb_correction:
                    nt_pb = spec_I.shape[0]
                    times_mjd_rb = rebin1d(times_mjd[:(nt_pb * timebin)], nt_pb) if nt_pb * timebin <= len(times_mjd) else times_mjd[:nt_pb]
                    pbfacs = np.ones(nt_pb, dtype=np.float32)
                    t0_pb = times_mjd_rb[0]
                    t1_pb = times_mjd_rb[-1]
                    if t1_pb - t0_pb > 5. / 1440.:
                        nstep = int((t1_pb - t0_pb) / (5. / 1440.))
                        ts_ref = np.linspace(t0_pb, t1_pb, nstep)
                        pbfacs_ref = np.ones_like(ts_ref)
                        print('Duration of the file is {0:.1f} hours, interpolating into {1:d} steps.'.format((t1_pb - t0_pb) * 24., nstep))
                        for i, t_ref in enumerate(ts_ref):
                            sun_loc = get_body('sun', Time(t_ref, format='mjd'), location=obs)
                            alt = sun_loc.transform_to(AltAz(obstime=Time(t_ref, format='mjd'), location=obs)).alt.radian
                            if np.degrees(alt) > 5.:
                                pbfacs_ref[i] = np.sin(alt) ** 1.6
                            else:
                                print('Warning! Calculated solar altitude is lower than 5 degrees. Something is wrong with the data (non-solar)?')
                                pbfacs_ref[i] = np.sin(np.radians(5.)) ** 1.6
                        pbfacs = np.interp(times_mjd_rb, ts_ref, pbfacs_ref).astype(np.float32)
                    else:
                        print('Duration of the file is {0:.1f} minutes, no interpolation will be done.'.format((t1_pb - t0_pb) * 24. * 60.))
                        t_ref = (t0_pb + t1_pb) / 2.
                        sun_loc = get_body('sun', Time(t_ref, format='mjd'), location=obs)
                        alt = sun_loc.transform_to(AltAz(obstime=Time(t_ref, format='mjd'), location=obs)).alt.radian
                        if np.degrees(alt) > 5.:
                            pbfacs *= np.sin(alt) ** 1.6
                        else:
                            print('Warning! Calculated solar altitude is lower than 5 degrees. Something is wrong with the data (non-solar)?')
                            pbfacs *= np.sin(np.radians(5.)) ** 1.6
                    spec_I /= pbfacs[:, None]

                if stokes.upper() == 'IV':
                    spec_V = _read_hdf5_chunked(ds['XY_imag'], ti0, ti1, fi0, fi1, timebin) / cavg[None, :]
                    if xhand_delay is not None:
                        spec_U = _read_hdf5_chunked(ds['XY_real'], ti0, ti1, fi0, fi1, timebin) / cavg[None, :]
                        spec_U, spec_V = apply_xhand_delay_UV(spec_U, spec_V, freqs, xhand_delay)
                    spec_new = np.stack((spec_I, spec_V), axis=2)
                    del spec_I, spec_V
                    stokes_out = ['I', 'V']
                    npol = 2
                else:
                    spec_new = spec_I
                    del spec_I
                    stokes_out = ['I']
                    npol = 1
            elif stokes.upper() == 'V':
                spec_new = _read_hdf5_chunked(ds['XY_imag'], ti0, ti1, fi0, fi1, timebin) / cavg[None, :]
                stokes_out = ['V']
                npol = 1
            elif stokes.upper() == 'Q':
                xx_rb = _read_hdf5_chunked(ds['XX'], ti0, ti1, fi0, fi1, timebin)
                yy_rb = _read_hdf5_chunked(ds['YY'], ti0, ti1, fi0, fi1, timebin)
                spec_new = xx_rb / cx[None, :] / 2. - yy_rb / cy[None, :] / 2.
                del xx_rb, yy_rb
                stokes_out = ['Q']
                npol = 1
            elif stokes.upper() == 'U':
                spec_new = _read_hdf5_chunked(ds['XY_real'], ti0, ti1, fi0, fi1, timebin) / cavg[None, :]
                stokes_out = ['U']
                npol = 1
        except Exception as e:
            print('Failed to read spectral data from {0:s}: {1:s}. Skip this file.'.format(file, str(e)))
            if not firstset_read:
                n0 += 1
            continue

        # spec_new is already time-rebinned by _read_hdf5_chunked; shape is (nt_new, nf[, npol])
        # Apply the valid-time filter (times_mjd > 50000) on the rebinned time axis
        nt_new = spec_new.shape[0]
        n_time_used = nt_new * timebin
        times_mjd_rb = rebin1d(times_mjd[:n_time_used], nt_new)
        idx, = np.where(times_mjd_rb > 50000.)
        times_mjd_rb = times_mjd_rb[idx]
        spec_new = spec_new[idx]

        try:
            # Frequency rebin
            nf = spec_new.shape[1]
            nf_new = nf // freqbin
            if spec_new.ndim == 2:
                spec_new = rebin2d(spec_new[:, :nf_new * freqbin], (nt_new, nf_new))
            else:  # ndim == 3
                spec_new = np.stack(
                    [rebin2d(spec_new[:, :nf_new * freqbin, i], (nt_new, nf_new)) for i in range(npol)],
                    axis=2)
            spec_new = np.transpose(spec_new).reshape((npol, 1, nf_new, nt_new)) / 1e4
            times_mjd_new = times_mjd_rb
            freqs_new = rebin1d(freqs[:nf_new * freqbin], nf_new)
        except Exception as e:
            print('Failed to rebin data from {0:s}: {1:s}. Skip this file.'.format(file, str(e)))
            if not firstset_read:
                n0 += 1
            continue

        if firstset_read:
            if n == n0:
                nfreq0 = len(freqs_new)
                freqs_out = freqs_new
                spec_out = spec_new
                times_mjd_out = times_mjd_new
            elif n > n0:
                if len(freqs_new) != nfreq0:
                    print('Something is wrong in concatenating {}'.format(file)) 
                    print('Dimension of the output frequency {0:d} does not match that of the first file {1:d}'.format(len(freqs_new), nfreq0)) 
                    continue
                else:
                    spec_out = np.concatenate((spec_out, spec_new), axis=3)
                    times_mjd_out = np.concatenate((times_mjd_out, times_mjd_new))
        else:
            continue

    if firstset_read:
        if verbose:
            print('Output time range is from {0:s} to {1:s}'.format(Time(times_mjd_out[0], format='mjd').isot, Time(times_mjd_out[-1], format='mjd').isot))
            print('Output data has {0:d} time stamps and {1:d} frequency channels'.format(len(times_mjd_out), len(freqs_out)))
        return spec_out, times_mjd_out, freqs_out, stokes_out, calfac_x, calfac_y, bkg_flux
    else:
        return False
