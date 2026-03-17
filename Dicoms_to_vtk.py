import sys
sys.path.append(fr"path to utils folder")
import os
import os.path as osp
import argparse
import numpy as np
import pyvista as pv
from tqdm import tqdm
import re
from vtk import *
import SimpleITK as sitk
import utils_registered as ut

##### HOW TO USE
# Folder should be organized in four differnt subfolder containing the dicom files of each of the four series of the 4d flow.
# The folder should be named Series_0, Series_1, Series_2, Series_3 and should contain the Anatomy Series, the SI, the AP and the AP respectively
# For the first series, the anatomy, a nii.gz file should be extracted
# Magnitude/Anatomy series always in the Series_0 folder


#-----------------------------------------------------------------------------------------------------------------------
## OPTIONS
#-----------------------------------------------------------------------------------------------------------------------
seg = False
show_plot = True
path_to_dicom = ''
vtk_grid_save = ''

args = argparse.Namespace(
    data_dir= path_to_dicom, #path_to_dicom
    save_dir= vtk_grid_save, #vtk_grid_save
    subject_id= '',
    venc=(0, 0, 0),
    flip_x=False,
    flip_y=True,
    flip_z=False,
    minus_u=False,
    minus_v=False,
    minus_w=True,
    write_pcmra=True,
    write_h5 = True,
    write_vel_mag = True,
    phase_coeff=1.0,
    mag_coeff=1.0
)

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

path_to_nifti = ''
pcrma_save_path = ''

#-----------------------------------------------------------------------------------------------------------------------
## Read dicom files and create numpy arrays
#-----------------------------------------------------------------------------------------------------------------------
data, meta = ut.read_acquisition(args.data_dir)
if args.venc == (0,0,0):
    args.venc = meta['venc']
arrayData = ut.seriesData_to_arrayData(data, meta)

#-----------------------------------------------------------------------------------------------------------------------
## VENC control
#-----------------------------------------------------------------------------------------------------------------------
magId = 0
magTemp = arrayData[magId]
arrayData.pop(magId)
velTemp = np.zeros((meta['num_rows'], meta['num_cols'], meta['num_slices'], meta['num_frames'], 3))
for i in range(3):
    velTemp[:, :, :, :, i] = arrayData[i][:]

v = meta['venc'] ### venc extracted in meters form the read acquisition function
venc = list(np.array(meta['venc'])*1000) # m/s -> mm/s
for d in range(3):
    lower_limit = venc[d]
    upper_limit = venc[d]

    min_vel = velTemp[:, :, :, :, d].min()
    max_vel = velTemp[:, :, :, :, d].max()

    # See if there are values outside the range
    if min_vel < lower_limit or max_vel > upper_limit:
        print(f"Expected Range: {lower_limit} a {upper_limit}")
        print(f"Measured range: {min_vel} a {max_vel}")
    else:
        print(f"Values inside the range")

#-----------------------------------------------------------------------------------------------------------------------
## Velocity adjustment
#-----------------------------------------------------------------------------------------------------------------------
print('Adjusting units.')
if re.search('GE', meta['vendor'], re.IGNORECASE):
    velTemp *= 1  # m/s

elif re.search('siemens', meta['vendor'], re.IGNORECASE) or re.search('philips', meta['vendor'], re.IGNORECASE):
    levels = 2**meta['HighBit']-1
    for d in range(3):
        velTemp[:, :, :, :, d] = (velTemp[:, :, :, :, d] - levels) * args.venc[d] / levels
    velTemp *= 0.01  # m/s
else:
    print('Manufacturer not found. Exiting.')
    sys.exit()

if meta['position'] == 'FFS':
    velTemp[:, :, :, :, 2] *= -1

if meta['position'] == 'HFS':
    velTemp[:, :, :, :, 0] *= -1
    velTemp[:, :, :, :, 1] *= -1


# swap X<->Y (axes 0 and 1)
velTemp = np.swapaxes(velTemp, 0, 1)
magTemp = np.swapaxes(magTemp, 0, 1)

# optional spatial flips
spatial_axes_to_flip = []
if args.flip_x: spatial_axes_to_flip.append(0)
if args.flip_y: spatial_axes_to_flip.append(1)
if args.flip_z: spatial_axes_to_flip.append(2)

if spatial_axes_to_flip:
    velTemp = np.flip(velTemp, axis=tuple(spatial_axes_to_flip))
    magTemp = np.flip(magTemp, axis=tuple(spatial_axes_to_flip))

# optional component sign flips (vectorized, no repeated slicing)
comp_sign = np.ones(3, dtype=velTemp.dtype)
if args.minus_u: comp_sign[0] = -1
if args.minus_v: comp_sign[1] = -1
if args.minus_w: comp_sign[2] = -1
velTemp *= comp_sign

# new_u = old_v ; new_v = old_w ; new_w = old_u
# then invert new_v and new_w
velCopy = velTemp.copy()
velTemp[..., 0] = velCopy[..., 1]
velTemp[..., 1] = velCopy[..., 2]
velTemp[..., 2] = velCopy[..., 0]

#-----------------------------------------------------------------------------------------------------------------------
## Meta data
#-----------------------------------------------------------------------------------------------------------------------
origin = meta["origin"]
pos = meta["position"]
venc_mps = np.array(venc)
s0 = meta["spacing"][1]  # row
s1 = meta["spacing"][2]  # col
s2 = meta["spacing"][0]  # dz
s0, s1 = s1, s0
dt = float(meta["dt"])

#-----------------------------------------------------------------------------------------------------------------------
## Processing of teh images: Eddy currents, Aliasing, Noise
#-----------------------------------------------------------------------------------------------------------------------
if ut.needs_unalias(velTemp, magTemp, venc_mps, spacing_xyz=(s0, s1, s2),tol=0.05, frac_thr=1e-4, mag_thr_rel=0.05):
    print("Aliasing detected -> running unaliasing.")
    velTemp, nwrap = ut.unalias_velocity_laplacian_4d(velTemp, venc_mps=venc_mps, spacing_xyz=(s0, s1, s2),dt=dt,temporal_scale=1.0)
else:
    print("No significant aliasing detected -> skipping unaliasing.")
velTemp = ut.eddy_current_correction(velTemp, magTemp, order=4, p=2, reg=1e-6, mag_threshold=None)
velTemp = ut.gaussian_smooth_velocity(velTemp, sigma=1.0)


#-----------------------------------------------------------------------------------------------------------------------
## Create vtk grids
#-----------------------------------------------------------------------------------------------------------------------
os.makedirs(osp.join(args.save_dir, 'flow'), exist_ok=True)
print('Creating grids and writing to file.')
for f in tqdm(range(meta['num_frames']), desc='Processing and saving frames'):
    mag = magTemp[:, :, :, f]
    u   = velTemp[:, :, :, f, 0]
    v   = velTemp[:, :, :, f, 1]
    w   = velTemp[:, :, :, f, 2]

    grid = pv.ImageData()
    grid.dimensions = np.array(mag.shape)
    grid.origin = meta['origin']
    #grid.origin = [0,0,0]
    grid.spacing = meta['spacing'][::-1]
    grid['MagnitudeSequence'] = mag.flatten(order='F')
    grid['Velocity'] = np.transpose(np.vstack((u.flatten(order='F'),
                                               v.flatten(order='F'),
                                               w.flatten(order='F'))))

    grid.save(osp.join(args.save_dir, 'flow', args.subject_id + '_{:02d}'.format(f) + '.vtk'), binary=True)


#-----------------------------------------------------------------------------------------------------------------------
## Create pcmra for VTK
#-----------------------------------------------------------------------------------------------------------------------
spacing = meta['spacing']
new_spacing = [spacing[1], spacing[2], spacing[0]]
ppp = meta['spacing'][::-1]

if args.write_pcmra:
    print('Writing PCMRA image.')
    pcmra = np.zeros((magTemp.shape[0], magTemp.shape[1], magTemp.shape[2]))
    for i in range(meta['num_frames']):
        mag = magTemp[:, :, :, i]
        phase1 = velTemp[:, :, :, i, 0]
        phase2 = velTemp[:, :, :, i, 1]
        phase3 = velTemp[:, :, :, i, 2]

        pcmra += (args.mag_coeff * mag) ** 2 * (
                    (args.phase_coeff * phase1) ** 2 + (args.phase_coeff * phase2) ** 2 + (args.phase_coeff * phase3) ** 2)

    pcmra /= meta['num_frames']
    pcmra = (pcmra ** 0.5)/100
    # save pcmra image
    pcmravtk = pv.wrap(pcmra)
    pcmravtk.SetOrigin(meta['origin'])
    pcmravtk.SetSpacing(meta['spacing'][::-1])
    pcmravtk.save(os.path.join(args.save_dir, args.subject_id + '_pcmra.vtk'))

#-----------------------------------------------------------------------------------------------------------------------
## Create pcmra for SLICER, VTK and SLicer use differnt orientation systems
#-----------------------------------------------------------------------------------------------------------------------
anatomy_series = sitk.ReadImage(path_to_nifti)
anatomy_spacing = anatomy_series.GetSpacing()
anatomy_origin = anatomy_series.GetOrigin()
anatomy_direction = anatomy_series.GetDirection()
anatomy_size = anatomy_series.GetSize()

if args.write_pcmra:
    print('Writing PCMRA image for Slicer.')
    pcmra2 = np.zeros((magTemp.shape[0], magTemp.shape[1], magTemp.shape[2])).T
    for i in range(meta['num_frames']):
        mag_T = magTemp[:, :, :, i].T
        phase1_T = velTemp[:, :, :, i, 0].T
        phase2_T = velTemp[:, :, :, i, 1].T
        phase3_T = velTemp[:, :, :, i, 2].T

        pcmra2 += (args.mag_coeff * mag_T) ** 2 * (
                    (args.phase_coeff * phase1_T) ** 2 + (args.phase_coeff * phase2_T) ** 2 + (args.phase_coeff * phase3_T) ** 2)

    pcmra2 /= meta['num_frames']
    pcmra2 = (pcmra2 ** 0.5)/100
    pcmravtk2 = pv.wrap(pcmra2)
    pcmravtk2.SetSpacing(meta['spacing'])#[::-1])


pcmr_img = sitk.GetImageFromArray(pcmra2.astype(np.float32))
pcmr_img.SetOrigin(anatomy_origin)
pcmr_img.SetSpacing(anatomy_spacing)
pcmr_img.SetDirection(anatomy_direction)
sitk.WriteImage(pcmr_img, pcrma_save_path)