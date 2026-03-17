import subprocess
import SimpleITK as sitk
import numpy as np
import pyvista as pv
import vtk

########## orient the image based on the chosen reference system
def reorient_image(in_img, target_orientation='RAS'):
    orient_filter = sitk.DICOMOrientImageFilter()
    orient_filter.SetDesiredCoordinateOrientation(target_orientation)
    out_img = orient_filter.Execute(in_img)
    return out_img

########## extract the rotation matrix used to reorient the image
def extract_rotation_matrix(in_img, target_orientation='RAS'):
    original_direction = np.array(in_img.GetDirection()).reshape(3, 3)
    orient_filter = sitk.DICOMOrientImageFilter()
    orient_filter.SetDesiredCoordinateOrientation(target_orientation)
    reoriented_img = orient_filter.Execute(in_img)
    reoriented_direction = np.array(reoriented_img.GetDirection()).reshape(3, 3)
    rotation_matrix = np.dot(reoriented_direction, np.linalg.inv(original_direction))
    return reoriented_img, rotation_matrix


def MarchingFromSeg(vtk_image_data, nval2gen, startvalue, endvalue, ft_angle = 45.0, smoothing = 0.1, bd_smoothing = True):
    discrete = vtk.vtkDiscreteMarchingCubes()
    discrete.SetInputData(vtk_image_data)
    discrete.GenerateValues(nval2gen, startvalue, endvalue)
    discrete.Update()
    mesh = pv.PolyData(discrete.GetOutput())
    smoothed_mesh = mesh.smooth_taubin(pass_band=smoothing,feature_angle=ft_angle, boundary_smoothing = bd_smoothing)
    return smoothed_mesh


######## total segmentator
def apply_totseg_aorta(img_fn, out_fn):
    subprocess.run([r'totalsegm path', '-i',
                    img_fn, '-o', out_fn, '-q', '-d', 'gpu', '-ta', 'total'])


def apply_totseg_ventricle(img_fn, out_fn):
    # Set license
    subprocess.run([r"totalseg_set_license.exe path",  '-l', 'put license number'])
    subprocess.run([r"path to totalsegm", '-i',
                    img_fn, '-o', out_fn, '-q', '-d', 'gpu', '-ta', 'heartchambers_highres'])