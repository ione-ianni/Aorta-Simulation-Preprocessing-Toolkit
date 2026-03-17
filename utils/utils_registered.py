import sys
import os
from os.path import join
import numpy as np
from itertools import groupby
from tqdm import tqdm
import pydicom
from collections import Counter
import re
from scipy import interpolate
from scipy.interpolate import RBFInterpolator, NearestNDInterpolator
from scipy.spatial import distance
import pyvista as pv
import vtk
#from vmtk import vmtkscripts
from vtk import *
import subprocess
from pymeshfix._meshfix import PyTMesh
import meshio
import pymeshfix as mf
import os
from os.path import join
import pydicom
import numpy as np
from collections import Counter
from itertools import groupby
from tqdm import tqdm
from scipy.ndimage import gaussian_filter


def get_dz(ds):
    try:
        dz = float(ds.SpacingBetweenSlices)
    except:
        dz = float(ds.SliceThickness)
    return dz

def get_venc(data):
    venc = [0] * 3
    # Check venc from the sequence name (e.g. fl3d1_v150fh)
    j = 0

    if hasattr(data['series0'][0]['info'], 'SequenceName'):
        pattern = re.compile(".*?_v(\\d+)(\\w+)")
        for i in range(4):
            ser = data['series' + str(i)]
            found = pattern.search(ser[0]['info'].SequenceName)
            if found:
                venc[j] = int(found.group(1))
                j += 1

    elif hasattr(data['series0'][0]['info'], 'SeriesDescription'):
        pattern = re.compile(".*?VENC (\\d+).*?")
        for i in range(3):
            ser = data['series' + str(i)]
            found = pattern.search(ser[0]['info'].SeriesDescription)
            if found:
                venc[j] = int(found.group(1))
                j += 1
    print('Detected venc:', venc)
    return venc



def read_acquisition(dataDir):
    series0 = []
    series1 = []
    series2 = []
    series3 = []
    series = []
    sNum = []
    subdirs = [d for d in os.listdir(dataDir) if os.path.isdir(join(dataDir, d)) and d.lower().startswith('s')]

    for subdir in subdirs:
        subdir_path = join(dataDir, subdir)
        files = os.listdir(subdir_path)
        for file in tqdm(files, desc=f'Reading {subdir}', disable=len(files) == 0):
            file_path = join(subdir_path, file)
            ds = pydicom.dcmread(file_path, force=True)
            sNum.append(ds.SeriesNumber)
            dataTemp = dict()
            dataTemp['FileName'] = file
            dataTemp['pixel_array'] = ds.pixel_array.astype('float')
            dataTemp['info'] = ds
            series.append(dataTemp)

    counter = Counter(sNum)
    sNum = np.unique(sNum)
    print(sNum)

    if len(counter) == 4:
        for i in range(len(series)):
            if int(series[i]['info'].SeriesNumber) == sNum[0]:
                series0.append(series[i])
            elif int(series[i]['info'].SeriesNumber) == sNum[1]:
                series1.append(series[i])
            elif int(series[i]['info'].SeriesNumber) == sNum[2]:
                series2.append(series[i])
            elif int(series[i]['info'].SeriesNumber) == sNum[3]:
                series3.append(series[i])
            else:
                print('Series number not found.')
                print(series[i]['info'].SeriesNumber)
                sys.exit(0)

    elif len(counter) == 2:
        num_imgs = list(counter.values())

        if num_imgs[0] > num_imgs[1]:
            assert num_imgs[0] == 3 * num_imgs[1]
            series_count = 0
            for i in range(len(series)):
                if int(series[i]['info'].SeriesNumber) == sNum[0]:
                    if series_count < num_imgs[1]:
                        series0.append(series[i])
                        series_count += 1
                    elif series_count < 2 * num_imgs[1]:
                        series1.append(series[i])
                        series_count += 1
                    elif series_count < 3 * num_imgs[1]:
                        series2.append(series[i])
                        series_count += 1
                else:
                    series3.append(series[i])

        if num_imgs[0] < num_imgs[1]:
            assert num_imgs[1] == 3 * num_imgs[0]
            series_count = 0
            for i in range(len(series)):
                if int(series[i]['info'].SeriesNumber) == sNum[1]:
                    if series_count < num_imgs[1]:
                        series0.append(series[i])
                        series_count += 1
                    elif series_count < 2 * num_imgs[1]:
                        series1.append(series[i])
                        series_count += 1
                    elif series_count < 3 * num_imgs[1]:
                        series2.append(series[i])
                        series_count += 1
                else:
                    series3.append(series[i])

    K = []
    for k, v in groupby(series0, key=lambda x: x['info'].SliceLocation):
        K.append(k)

    if hasattr(ds, 'NominalInterval'):  # Use NominalInterval if available
        T = float(ds.NominalInterval) / 1000  # Convert from milliseconds to seconds
    elif hasattr(ds, 'TemporalResolution'):
        T = float(ds.TemporalResolution) / 1000

    scale_tag = (0x0019, 0x10e2)
    if scale_tag in ds:
        scale = ds[scale_tag].value
        print("VelocityEncodeScale =", scale)
    else:
        print("VelocityEncodeScale not found in dataset.")
        scale = 1

    vendor = ds.Manufacturer
    slices = len(set(K))
    frames = len(series0) // slices
    dt = float(T/frames)
    rows = ds.Rows
    columns = ds.Columns
    origin1      = ds.ImagePositionPatient
    origin = [0.0, 0.0, 0.0]
    orientation = ds.ImageOrientationPatient
    position = ds.PatientPosition
    # period      = float(ds.NominalInterval) / 1000
    spacing = [get_dz(ds), float(ds.PixelSpacing[0]),  float(ds.PixelSpacing[1])] #[float(ds.PixelSpacing[1]), float(ds.PixelSpacing[0]), get_dz(ds)]
    spacing = [s for s in spacing]

    series0 = sorted(series0, key=lambda k: k['FileName'])
    series1 = sorted(series1, key=lambda k: k['FileName'])
    series2 = sorted(series2, key=lambda k: k['FileName'])
    series3 = sorted(series3, key=lambda k: k['FileName'])

    meta = {'vendor': vendor,
            'num_slices': slices,
            'num_frames': frames,
            'num_rows': rows,
            'num_cols': columns,
            'origin': origin,
            'origin1': origin1,
            'orientation': orientation,
            'position': position,
            'spacing': spacing,
            'HighBit': ds.HighBit,
            'VelocityEncodeScale': scale,
            'dt': dt,
            'T': T
    }

    series_data = {'series0': series0,
            'series1': series1,
            'series2': series2,
            'series3': series3
    }

    # venc detection e velcoity scaling
    venc = get_venc(series_data)
    if np.mean(venc) > 80:
        venc = [vv * 0.01 for vv in venc]
    meta['venc'] = venc

    return series_data, meta


def seriesData_to_arrayData(seriesData, meta):
    arrayData = []
    for s in seriesData.keys():
        series = seriesData[s]
        newArr = np.zeros((meta['num_rows'], meta['num_cols'], meta['num_slices'], meta['num_frames']))
        try:
            #IPP = []
            for j in range(1, meta['num_frames'] + 1):
                frameBlock = [elem for elem in series if int(elem['info'].TemporalPositionIdentifier) == j]
                frameBlock = sorted(frameBlock, key=lambda k: k['info'].SliceLocation)
                for i in range(meta['num_slices']):
                    newArr[:, :, i, j - 1] = frameBlock[i]['pixel_array']
                    #IPP.append(frameBlock[i]['IPP'])
            arrayData.append(newArr)
        except:
            series = sorted(series, key=lambda k: k['info'].SliceLocation)
            #series = sorted(series, key=lambda k: k['FileName'])
            ids = np.arange(0, meta['num_slices'] * meta['num_frames'] - meta['num_frames'], meta['num_frames'])
            for i in range(len(ids)):
                for j in range(meta['num_frames']):
                    newArr[:, :, i, j] = series[ids[i] + j]['pixel_array']
            arrayData.append(newArr)

    return arrayData



def _poly_exponent_list_3d(order):
    """Return list of exponent triplets (i,j,k) for 3D polynomial up to given total order."""
    exps = []
    for total in range(order+1):
        for i in range(total+1):
            for j in range(total - i + 1):
                k = total - i - j
                exps.append((i, j, k))
    return exps

def _build_design_matrix(coords, exps):
    """
    coords: (N,3) array of normalized coordinates (x,y,z)
    exps: list of (i,j,k)
    returns: B of shape (N, num_basis)
    """
    N = coords.shape[0]
    B = np.empty((N, len(exps)), dtype=np.float64)
    x = coords[:,0]; y = coords[:,1]; z = coords[:,2]
    for col, (i,j,k) in enumerate(exps):
        # compute x**i * y**j * z**k
        B[:, col] = (x**i) * (y**j) * (z**k)
    return B


def eddy_current_correction(vel, #(nx,ny,nz,nt,3)
                            mag, #(nx,ny,nz,nt)
                            order=4, #polynomial order
                            p=2, #exponent for SD weighting based on paper p=2
                            reg=1e-6, #Tikhonov regularization
                            mag_threshold=None,
                            verbose=True):
    """
    Apply weighted higher-order polynomial correction to time-resolved 3D velocity field.
    Based on Ebbers et al.
    Returns:
      vel_corr: corrected velocity array (same shape as vel)
    """
    # Validate shapes
    assert vel.ndim == 5 and vel.shape[4] == 3, "vel must be (nx,ny,nz,nt,3)"
    nx, ny, nz, nt, _ = vel.shape
    if mag.ndim == 4:
        pass
    elif mag.ndim == 3:
        mag = np.repeat(mag[..., None], nt, axis=3)
    else:
        raise ValueError("mag must be 3D or 4D (nx,ny,nz,nt)")

    # 1) compute temporal SD of velocity per voxel (use magnitude of velocity vector)
    #    SDv(x,y,z) = std_t( |v|(x,y,z,t) )
    vel_mag = np.linalg.norm(vel, axis=4)  # (nx,ny,nz,nt)
    SDv = np.std(vel_mag, axis=3, ddof=0)  # (nx,ny,nz)

    # 2) build weighting map w(x,y,z,t) = m(x,y,z,t) * (1 / (SDv(x,y,z) ** p))
    eps = 1e-8
    inv_SDv_p = 1.0 / (np.maximum(SDv, eps) ** p)  # (nx,ny,nz)
    w = mag * inv_SDv_p[..., None]  # (nx,ny,nz,nt)

    # Magnitude threshold: zero-out weights for very low mean mag voxels, only if there are very low values
    if mag_threshold is not None:
        mean_mag = np.mean(mag, axis=3)
        low_mask = mean_mag < mag_threshold
        w[low_mask, :] = 0.0

    # Flatten voxel grid (we perform fit in spatial domain, using all voxels)
    X, Y, Z = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij')
    coords = np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=1).astype(np.float64)  # (N,3)
    N = coords.shape[0]

    # Normalize coords to [-1,1] to improve numerical conditioning
    coords_min = coords.min(axis=0)
    coords_max = coords.max(axis=0)
    denom = coords_max - coords_min
    denom[denom == 0] = 1.0
    coords_norm = (coords - coords_min) / denom * 2.0 - 1.0  # -> [-1,1]

    exps = _poly_exponent_list_3d(order)
    B = _build_design_matrix(coords_norm, exps)  # (N, nb)

    # Precompute weights summed over time, because fit is spatial:
    # the weighted fit is computed for each time frame using w(x,y,z,t) and v(x,y,z,t).
    nb = B.shape[1]
    vel_corr = np.copy(vel)

    # Precompute B^T once (it's same for all frames)
    Bt = B.T  # (nb, N)

    # Flatten w and v spatially for each t
    for t in range(nt):
        if verbose and (t % max(1, nt//5) == 0):
            print(f"  Fitting frame {t+1}/{nt} ...")
        wt = w[..., t].ravel()  # (N,)
        vt_x = vel[..., t, 0].ravel()
        vt_y = vel[..., t, 1].ravel()
        vt_z = vel[..., t, 2].ravel()

        # Only use voxels with non-zero weight
        nz_idx = wt > 0
        if nz_idx.sum() < nb:
            # not enough equations; skip correction for this frame
            if verbose:
                print(f"    Warning: insufficient weighted voxels ({nz_idx.sum()}) for fitting (need >= {nb}). Skipping frame.")
            continue

        W_sqrt = np.sqrt(wt[nz_idx])  # shape (M,)
        # Form weighted design and targets
        B_w = (B[nz_idx, :].T * W_sqrt).T  # (M, nb)
        yx = vt_x[nz_idx] * W_sqrt
        yy = vt_y[nz_idx] * W_sqrt
        yz = vt_z[nz_idx] * W_sqrt

        # Solve (B_w^T B_w + reg I) a = B_w^T y
        Nmat = B_w.T.dot(B_w)

        # regularize
        Nmat.flat[::nb+1] += reg
        rhs_x = B_w.T.dot(yx)
        rhs_y = B_w.T.dot(yy)
        rhs_z = B_w.T.dot(yz)
        try:
            a_x = np.linalg.solve(Nmat, rhs_x)
            a_y = np.linalg.solve(Nmat, rhs_y)
            a_z = np.linalg.solve(Nmat, rhs_z)
        except np.linalg.LinAlgError:
            # fallback to least-squares
            a_x, _, _, _ = np.linalg.lstsq(B_w, yx, rcond=None)
            a_y, _, _, _ = np.linalg.lstsq(B_w, yy, rcond=None)
            a_z, _, _, _ = np.linalg.lstsq(B_w, yz, rcond=None)

        # Evaluate fitted polynomial at all voxels and subtract from vel[...,t,:]
        fitted_all_x = B.dot(a_x).reshape((nx, ny, nz))
        fitted_all_y = B.dot(a_y).reshape((nx, ny, nz))
        fitted_all_z = B.dot(a_z).reshape((nx, ny, nz))

        vel_corr[..., t, 0] = vel[..., t, 0] - fitted_all_x
        vel_corr[..., t, 1] = vel[..., t, 1] - fitted_all_y
        vel_corr[..., t, 2] = vel[..., t, 2] - fitted_all_z

    if verbose:
        print("Eddy current correction completed.")
    return vel_corr


def gaussian_smooth_velocity(vel, sigma=1.0):
    vel_sm = np.zeros_like(vel)
    nx, ny, nz, nt, _ = vel.shape
    for t in range(nt):
        for d in range(3):
            vel_sm[..., t, d] = gaussian_filter(
                vel[..., t, d],
                sigma=sigma,
                mode='nearest'
            )
    return vel_sm



def unalias_velocity_laplacian_4d(
    vel_mps: np.ndarray,          # (X,Y,Z,T,3)
    venc_mps,                     # array-like (3,) in m/s or in mms/s there MUST be consisten with the vel_mps data
    spacing_xyz=(1.0, 1.0, 1.0),  # (dx,dy,dz)
    dt=1.0,                       # seconds (or frame spacing)
    temporal_scale=1.0,
):
    """
    De-alias 4D flow velocities by converting to wrapped phase, applying 4D Laplacian unwrapping,
    and converting back to velocity.

    Returns:
    vel_unaliased_mps : np.ndarray same shape as input
    n_wraps           : np.ndarray (X,Y,Z,T,3) integer wrap counts

    Refernce article: Loecher et al., https://doi.org/10.1002/jmri.25045

    """
    vel = np.asarray(vel_mps)
    if vel.ndim != 5 or vel.shape[-1] != 3:
        raise ValueError(f"Expected vel shape (X,Y,Z,T,3), got {vel.shape}")

    venc = np.asarray(venc_mps, dtype=np.float64).reshape(3,)

    X, Y, Z, T, C = vel.shape
    dx, dy, dz = map(float, spacing_xyz)
    dt = float(dt)
    temporal_scale = float(temporal_scale)

    # --- helper: wrap to (-pi, pi] --- this function is called only if the input data are unwrapped velocities
    def wrap_pi(phi):
        return (phi + np.pi) % (2.0 * np.pi) - np.pi

    # --- build discrete Laplacian eigenvalues
    wx = 2.0 * np.pi * np.fft.fftfreq(X)  # rad/sample
    wy = 2.0 * np.pi * np.fft.fftfreq(Y)
    wz = 2.0 * np.pi * np.fft.fftfreq(Z)
    wt = 2.0 * np.pi * np.fft.fftfreq(T)

    lam_x = (-4.0 * (np.sin(wx / 2.0) ** 2)) / (dx * dx)
    lam_y = (-4.0 * (np.sin(wy / 2.0) ** 2)) / (dy * dy)
    lam_z = (-4.0 * (np.sin(wz / 2.0) ** 2)) / (dz * dz)
    lam_t = temporal_scale * (-4.0 * (np.sin(wt / 2.0) ** 2)) / (dt * dt)

    lam = (lam_x[:, None, None, None] +
           lam_y[None, :, None, None] +
           lam_z[None, None, :, None] +
           lam_t[None, None, None, :])  # (X,Y,Z,T), <=0, DC=0

    def lap(f):
        F = np.fft.fftn(f, axes=(0, 1, 2, 3))
        return np.fft.ifftn(lam * F, axes=(0, 1, 2, 3)).real

    def inv_lap(g):
        G = np.fft.fftn(g, axes=(0, 1, 2, 3))
        outF = np.zeros_like(G)
        mask = lam != 0.0
        outF[mask] = G[mask] / lam[mask]   # solve Poisson
        return np.fft.ifftn(outF, axes=(0, 1, 2, 3)).real

    two_pi = 2.0 * np.pi

    vel_unaliased = np.empty_like(vel, dtype=np.float32)
    n_wraps = np.empty_like(vel, dtype=np.int32)

    for d in range(3):
        # 1) velocity -> wrapped phase (Eq.1: u = v*pi/Venc) only if the input data are unwrapped velocities
        phi_w = wrap_pi((np.pi * vel[..., d]) / venc[d]).astype(np.float64)

        # 2) compute Laplacian(true phase) from wrapped phase via sin/cos
        sinw = np.sin(phi_w)
        cosw = np.cos(phi_w)
        lap_true = cosw * lap(sinw) - sinw * lap(cosw)

        # 3) laplacian of wrapped phase
        lap_w = lap(phi_w)

        # 4) n(r) via inverse Laplacian
        n_float = inv_lap(lap_true - lap_w) / two_pi
        n_int = np.rint(n_float).astype(np.int32)

        # 5) unwrap phase and back to velocity
        phi_u = phi_w + two_pi * n_int
        vel_unaliased[..., d] = (phi_u * venc[d] / np.pi).astype(np.float32)
        n_wraps[..., d] = n_int

    return vel_unaliased, n_wraps


def needs_unalias(vel, mag, venc_mps, spacing_xyz, tol=0.05, frac_thr=1e-4, mag_thr_rel=0.05):
    """
    ## Aliasing check + optional unaliasing (run only if needed)
    vel: [X,Y,Z,T,3] in m/s
    mag: [X,Y,Z,T]
    venc_mps: [3] in m/s
    tol: 5% margin over VENC to avoid triggering on small noise
    frac_thr: minimum fraction of voxels exceeding (VENC*(1+tol)) to trigger unaliasing
    mag_thr_rel: mask threshold relative to max(mag) to ignore background
    """
    mag_thr = mag_thr_rel * float(mag.max())
    mask = mag > mag_thr
    if not np.any(mask):
        mask = np.ones(mag.shape, dtype=bool)

    trigger = False
    for d, comp in enumerate(["u", "v", "w"]):
        abs_v = np.abs(vel[..., d])
        vmax = float(abs_v[mask].max()) if np.any(mask) else float(abs_v.max())
        frac = float(np.mean(abs_v[mask] > venc_mps[d] * (1.0 + tol)))

        print(f"[alias check] {comp}: max|v|={vmax:.3f} m/s  |  VENC={venc_mps[d]:.3f} m/s  |  frac_exceed={frac:.6f}")

        if frac > frac_thr:
            trigger = True

    return trigger


