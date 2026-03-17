import sys
sys.path.append(fr"path to utils folder")
import os
import os.path as osp
import SimpleITK as sitk
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
import networkx as nx
import image_utils as imut
import mesh_utils as mut
import vmtk_utils as vmtkut


#-----------------------------------------------------------------------------------------------------------------------
## OPTIONS
#-----------------------------------------------------------------------------------------------------------------------
seg = False
show_plot = True
reorinet = False
seed_detection = 'ventricle-based' ## lb-based or ventricle-based

#-----------------------------------------------------------------------------------------------------------------------
## PATHS
#-----------------------------------------------------------------------------------------------------------------------
input_img = fr"" #path to your .nii.gz image
output_img = fr"" #path where to store the segmentations
out_dir = fr"" #path where saving the processed data
os.makedirs(out_dir, exist_ok=True)

#-----------------------------------------------------------------------------------------------------------------------
## SEGMENTATION
#-----------------------------------------------------------------------------------------------------------------------
if seg: ## if seg is set to true the code will call total segmentator to
    if seed_detection == 'ventricle-based':
        seg1 = imut.apply_totseg_ventricle(input_img, output_img)
        seg2 = imut.apply_totseg_aorta(input_img, output_img)
    else:
        seg2 = imut.apply_totseg_aorta(input_img, output_img)

pid = 'aorta'
pid1 = 'brachiocephalic_trunk'
pid2 = 'common_carotid_artery_left'
pid3 = 'subclavian_artery_left'
pid4 = 'heart_ventricle_left'
pid5 = 'pulmonary_artery'
pid6 = 'heart_ventricle_right'


#-----------------------------------------------------------------------------------------------------------------------
## Read files and create polydatas
#-----------------------------------------------------------------------------------------------------------------------
print('\nExtracting surface from label aorta and sovra-aortic vessels...')
ao_img = sitk.ReadImage(osp.join(output_img, f'{pid}.nii.gz'))
bct_img = sitk.ReadImage(osp.join(output_img, f'{pid1}.nii.gz'))
car_img = sitk.ReadImage(osp.join(output_img, f'{pid2}.nii.gz'))
suc_img = sitk.ReadImage(osp.join(output_img, f'{pid3}.nii.gz'))

### adjust the error in the segmetation if present
ao_img = sitk.BinaryMorphologicalOpening(ao_img, [1]*3)
ao_img = sitk.BinaryMorphologicalClosing(ao_img, [2]*3)
ao_img = sitk.BinaryDilate(ao_img, [1]*3)

##### merging the mask to obtain the aortic model
ao_img_tot = sitk.Or(ao_img, bct_img)
ao_img_tot = sitk.Or(ao_img_tot, car_img)
ao_img_tot = sitk.Or(ao_img_tot, suc_img)
ao_img_tot = sitk.BinaryErode(ao_img_tot, [1]*3)
ao_img_tot = sitk.VotingBinaryHoleFilling(ao_img_tot, [3]*3)

if reorinet:
    sitk.WriteImage(imut.reorient_image(ao_img_tot), f'outAo{pid}.nii.gz')
    sitk.WriteImage(imut.reorient_image(bct_img), f'outAo_{pid1}.nii.gz')
    sitk.WriteImage(imut.reorient_image(car_img), f'outAo_{pid2}.nii.gz')
    sitk.WriteImage(imut.reorient_image(suc_img), f'outAo_{pid3}.nii.gz')
else:
    sitk.WriteImage(ao_img_tot, f'outAo{pid}.nii.gz')
    sitk.WriteImage(bct_img, f'outAo_{pid1}.nii.gz')
    sitk.WriteImage(car_img, f'outAo_{pid2}.nii.gz')
    sitk.WriteImage(suc_img, f'outAo_{pid3}.nii.gz')

ao_img_tot = pv.read(f'outAo{pid}.nii.gz')
os.remove(f'outAo{pid}.nii.gz')
bct_img = pv.read(f'outAo_{pid1}.nii.gz')
os.remove(f'outAo_{pid1}.nii.gz')
car_img = pv.read(f'outAo_{pid2}.nii.gz')
os.remove(f'outAo_{pid2}.nii.gz')
suc_img = pv.read(f'outAo_{pid3}.nii.gz')
os.remove(f'outAo_{pid3}.nii.gz')

ao_surf = ao_img_tot.threshold([0.5, 1.5]).extract_surface().connectivity('largest').triangulate()
bct_surf = bct_img.threshold([0.5, 1.5]).extract_surface().connectivity('largest').triangulate()
car_surf = car_img.threshold([0.5, 1.5]).extract_surface().connectivity('largest').triangulate()
suc_surf = suc_img.threshold([0.5, 1.5]).extract_surface().connectivity('largest').triangulate()
sav_surf = bct_surf.merge(car_surf).merge(suc_surf)

if show_plot:
    pl = pv.Plotter()
    pl.add_mesh(ao_surf, color='red', opacity=0.5)
    pl.show()

if seed_detection=='ventricle-based':
    print('\nExtracting surface from label left ventricle, pulmonary artery ...')
    polm_img = sitk.ReadImage(osp.join(output_img, f'{pid5}.nii.gz'))
    polm_img = sitk.BinaryMorphologicalOpening(polm_img, [1]*3)
    lv_img = sitk.ReadImage(osp.join(output_img, f'{pid4}.nii.gz'))

    rv_img = sitk.ReadImage(osp.join(output_img, f'{pid6}.nii.gz'))
    rv_img = sitk.BinaryMorphologicalOpening(rv_img, [3] * 3)

    if reorinet:
        sitk.WriteImage(imut.reorient_image(polm_img), f'outAo_{pid5}.nii.gz')
        sitk.WriteImage(imut.reorient_image(lv_img), f'outAo_{pid4}.nii.gz')
        sitk.WriteImage(imut.reorient_image(rv_img), f'outAo_{pid6}.nii.gz')
    else:
        sitk.WriteImage(polm_img, f'outAo_{pid5}.nii.gz')
        sitk.WriteImage(lv_img, f'outAo_{pid4}.nii.gz')
        sitk.WriteImage(rv_img, f'outAo_{pid6}.nii.gz')


    polm_img = pv.read(f'outAo_{pid5}.nii.gz')
    os.remove(f'outAo_{pid5}.nii.gz')
    lv_img = pv.read(f'outAo_{pid4}.nii.gz')
    os.remove(f'outAo_{pid4}.nii.gz')
    rv_img = pv.read(f'outAo_{pid6}.nii.gz')
    os.remove(f'outAo_{pid6}.nii.gz')

    polm_surf = imut.MarchingFromSeg(polm_img, 1, 1, 1).connectivity('largest').triangulate()
    lv_surf = lv_img.threshold([0.5, 1.5]).extract_surface()
    rv_surf = rv_img.threshold([0.5, 1.5]).extract_surface()

    polm_surf = mut.meshfix(polm_surf)
    polm_surf = mut.mmg_remesh(polm_surf, hausd=0.3, hmax=1.5, hmin=0.9, max_aspect_ratio=5, max_iter=3, verbose=True)
    polm_surf = mut.smoothVtk(polm_surf, nIter=120, passband=0.015)
    polm_surf = mut.meshfix(polm_surf)

    if show_plot:
        pl = pv.Plotter()
        pl.add_mesh(polm_surf, color='b', opacity=0.5, show_edges=False)
        pl.show()


#-----------------------------------------------------------------------------------------------------------------------
## Surface smoothing and remeshing
#-----------------------------------------------------------------------------------------------------------------------
print('Remeshing surfaces...')
ao_surf = mut.meshfix(ao_surf)
ao_surf = mut.mmg_remesh(ao_surf, hausd=0.3, hmax=1.1, hmin=0.9, max_aspect_ratio=5, max_iter=3, verbose=True)
ao_surf = mut.mmg_remesh_adaptive_aorta(ao_surf, sav_surf, hausd=0.3, main_size=1, sec_size=0.5,  max_aspect_ratio=5, max_iter=1, verbose=True)

print('Smoothing surface...')
### set the number of iteration and the thershold to change the amount of smoothing
ao_surf = mut.smoothVtk(ao_surf, nIter=120, passband=0.03)

print('Mesh fixing...')
ao_surf = mut.meshfix(ao_surf)

# compute LB eigenvectors
print('Computing LB eigenvectors...')
ao_surf = mut.compute_LB_eigenvectors(ao_surf, num_eigs=6, normalize=True)

# compute HKS
print('Computing HKS...')
ao_surf = mut.compute_hks_v2(ao_surf, t=100, fn=f'ao_{pid}.vtp')

if show_plot:
    pl = pv.Plotter()
    pl.add_mesh(ao_surf, color='w', opacity=0.5, show_edges=False)
    pl.show()


#-----------------------------------------------------------------------------------------------------------------------
## Detect aortic valve plane and prepare inlet
#-----------------------------------------------------------------------------------------------------------------------
if seed_detection =='ventricle-based':
    print('Aortic valve plane detection...')
    lv_surf.compute_implicit_distance(ao_surf, inplace=True)
    av_region = lv_surf.threshold_percent(percent=0.1, scalars='implicit_distance', invert=True) #idetification of the region of the left ventricle closer to the aorta
    av_center = np.array(av_region.center)
    av_radius = np.mean([np.linalg.norm(pt - av_center) for pt in av_region.points])
    av_normal = pv.fit_plane_to_points(av_region.points).point_normals.mean(0)
    if av_normal[2] < 0: av_normal = -1 * av_normal
    if show_plot:
        pl = pv.Plotter()
        pl.add_mesh(ao_surf, color='w', opacity=1)
        pl.add_mesh(lv_surf, color='pink', opacity=0.5)
        pl.add_mesh(av_region, color='red', opacity=1)
        pl.show()

    print('Centerline seed detection...')
    source_pt, _ = ao_surf.ray_trace(av_center - 20 * av_normal, av_center + 20 * av_normal)
    st_id = ao_surf.find_closest_point(source_pt.T)
    source_pt = ao_surf.points[st_id]

else:
    #print('Centerline seed detection...')
    #source_pt = ao_surf.points[np.argmax(ao_surf['eigvec1'])]
    #st_id = ao_surf.find_closest_point(source_pt)
    print("Centerline seed detection...")

    eig = np.asarray(ao_surf["eigvec1"], dtype=float)
    maxv = float(eig.max())

    rel = 0.01  # 2% sotto il massimo (aumenta a 0.05 se pochi punti)
    cand_ids = np.where(eig >= maxv * (1.0 - rel))[0]

    #if the numnber of points is small change rel
    if cand_ids.size < 3:
        rel = 0.05
        cand_ids = np.where(eig >= maxv * (1.0 - rel))[0]
    if cand_ids.size < 3:
        N = min(500, ao_surf.n_points)
        cand_ids = np.argsort(eig)[-N:]

    cand_pts = ao_surf.points[cand_ids]
    i1 = cand_ids[np.argmax(eig[cand_ids])]
    p1 = ao_surf.points[i1]
    cand_pts2 = np.vstack([p1, cand_pts])

    p1, p2, p3 = mut.pick_3_points_max_triangle(cand_pts2, min_area=1e-6)
    center = (p1 + p2 + p3) / 3.0
    source_pt = ao_surf.points[ao_surf.find_closest_point(center)]
    st_id = ao_surf.find_closest_point(source_pt)

    if show_plot:
        pl = pv.Plotter()
        pl.add_mesh(ao_surf, color='white', opacity=0.4)
        pl.add_points(ao_surf.points[cand_ids], color='orange', point_size=3, render_points_as_spheres=True)
        pl.add_points(p1, color='red', point_size=15, render_points_as_spheres=True)
        pl.add_points(p2, color='cyan', point_size=12, render_points_as_spheres=True)
        pl.add_points(source_pt, color='green', point_size=18, render_points_as_spheres=True)
        pl.show()

#-----------------------------------------------------------------------------------------------------------------------
## Detect outlets points
#-----------------------------------------------------------------------------------------------------------------------
# Detect aortic centerline endpoints
G_ao = mut.convert_triangle_mesh_to_graph(ao_surf)
shortest_paths_lengths = nx.single_source_shortest_path_length(G_ao, st_id)
ao_surf['Geodesics'] = np.zeros((ao_surf.n_points,))
for j in range(ao_surf.n_points):
    geo_path = shortest_paths_lengths[j]
    ao_surf['Geodesics'][j] = geo_path
ao_surf['Geodesics_n'] = (ao_surf['Geodesics'] - np.min(ao_surf['Geodesics'])) / np.ptp(ao_surf['Geodesics'])
desc_ao_pt = ao_surf.points[np.argmax(ao_surf['Geodesics_n'])]
surf_with_geodesics = ao_surf.copy()

# Detect aortic centerline endpoints
G_ao = mut.convert_triangle_mesh_to_graph(ao_surf)
shortest_paths_lengths = nx.single_source_shortest_path_length(G_ao, st_id)
ao_surf['Geodesics'] = np.zeros((ao_surf.n_points,))
for j in range(ao_surf.n_points):
    geo_path = shortest_paths_lengths[j]
    ao_surf['Geodesics'][j] = geo_path
ao_surf['Geodesics_n'] = (ao_surf['Geodesics'] - np.min(ao_surf['Geodesics'])) / np.ptp(ao_surf['Geodesics'])
desc_ao_pt = ao_surf.points[np.argmax(ao_surf['Geodesics_n'])]
surf_with_geodesics = ao_surf.copy()

# Detect ends: recursive thresholding of geodesic distance to ostia, change delta based on your need to correctly selecting the candidates outlets
delta_0 = 0.05
cand_outlets = []
while np.size(cand_outlets,0) < 4:
    cand_outlets = []
    for delta in np.arange(delta_0, 1, delta_0):
        thr = ao_surf.threshold_percent(1-delta, scalars='Geodesics_n').connectivity()
        for j in np.unique(thr['RegionId']):
            thr_connected = thr.threshold([j, j+0.5], scalars='RegionId')
            cand_outlets.append(np.array(thr_connected.points[np.argmax(thr_connected['Geodesics_n'])]))
    cand_outlets = np.unique(cand_outlets, axis=0)
    delta_0 = delta_0/2

# control desc_ao_pt
desc_ao_v = (desc_ao_pt-source_pt)/np.linalg.norm(desc_ao_pt-source_pt)
vectors = [(pt - source_pt)/np.linalg.norm(pt-source_pt)for pt in cand_outlets]
scalar_products = np.asarray([np.dot(desc_ao_v, vec) for vec in vectors])
negative_count = sum(1 for s in scalar_products if s < 0)
if negative_count == 1:
    desc_ao_pt = cand_outlets[scalar_products<0][0]

# descending aorta point removed from the candidate outlets
toremove = np.argmin([np.linalg.norm(desc_ao_pt - c) for c in cand_outlets])
cand_outlets = np.array([cand_outlets[i] for i in range(len(cand_outlets)) if i != toremove])
branchTargetCoord = cand_outlets

# remove candidates with low HKS (outlet points are associeted with higher values of the HKS
if len(cand_outlets) > 3:
    cand_hks = ao_surf['hks'][[ao_surf.find_closest_point(i) for i in cand_outlets]]
    tokeep = np.argsort(cand_hks)[-3:]
    cand_outlets = np.array([cand_outlets[i] for i in tokeep])
    branchTargetCoord = cand_outlets
else:
    branchTargetCoord = cand_outlets

branchTargetCoord_index = np.argsort(branchTargetCoord[:, 2])[::-1]
branchTargetCoord = branchTargetCoord[branchTargetCoord_index[:3]]

if show_plot:
    pl = pv.Plotter()
    pl.add_mesh(ao_surf, color='w', opacity=0.6, show_edges=False)
    pl.add_mesh(source_pt, color='g', line_width=15)
    pl.add_mesh(branchTargetCoord, color='r', line_width=15)
    pl.add_mesh(desc_ao_pt, color='m', line_width=15)
    pl.show()


#-----------------------------------------------------------------------------------------------------------------------
## Centerline detection using VMTK library
#-----------------------------------------------------------------------------------------------------------------------
#####change the window size and the polyorder to filter the centerline
parentCenterline_temp= vmtkut.extract_centerline(ao_surf, list(source_pt), list(desc_ao_pt), resampling=0.1, appendEndPoints=1)
parentCenterline = mut.filter_centerline(parentCenterline_temp, window_size=int(parentCenterline_temp.n_points/5), polyorder=3, resampling=0.1)

centerlines_lz = vmtkut.extract_branch_centerlines(ao_surf, parentCenterline, branchTargetCoord)
centerlines = vmtkut.extract_centerline(ao_surf, list(source_pt), [c for pt in branchTargetCoord for c in pt],
                                        resampling=0.1, appendEndPoints=0).connectivity()

outlet_points = list(branchTargetCoord[0])[:] + list(branchTargetCoord[1])[:] + list(branchTargetCoord[2])[:] + list(desc_ao_pt)[:]

if show_plot:
    pl = pv.Plotter()
    pl.add_mesh(ao_surf, color='white', opacity=0.7)
    pl.add_mesh(parentCenterline, color='blue', line_width=10)
    pl.add_mesh(centerlines_lz, color='pink', line_width=10)
    pl.add_axes()
    pl.show()

parentCenterline = mut.orient_centerline_by_seeds(parentCenterline, source_pt)
centerlines = mut.orient_centerline_by_seeds(centerlines, source_pt)


print('Computing landing zones...')
#-----------------------------------------------------------------------------------------------------------------------
## Edge detection
#-----------------------------------------------------------------------------------------------------------------------
######## separate the centerlines between the aorta and the supra-aortic vessels
surface, centerlines_lz = vmtkut.divide_surface(ao_surf, centerlines_lz)
cl0, cl1, cl2, cl3 = vmtkut.separate_centerlines(centerlines_lz)

if show_plot:
    pl = pv.Plotter()
    pl.add_mesh(parentCenterline, color='k', line_width=1, opacity=0.5)
    pl.add_mesh(cl0, color='pink', line_width=1)
    pl.add_mesh(cl1, color='blue')
    pl.add_mesh(cl2, color='green')
    pl.add_mesh(cl3, color='magenta')
    pl.show()

cl0_n = pv.wrap(vmtkut.resample_centerline(parentCenterline, step=0.4))
cl1 = vmtkut.resample_centerline(cl1, step=0.4)
cl2 = vmtkut.resample_centerline(cl2, step=0.4)
cl3 = vmtkut.resample_centerline(cl3, step=0.4)

centerlines_list = [cl0_n, cl1, cl2, cl3]
for i in range(len(centerlines_list)): centerlines_list[i] = vmtkut.add_centerline_geometry(centerlines_list[i])

parentCenterline_temp = vmtkut.add_centerline_geometry(cl0_n)
tubeAo = mut.create_tube0(parentCenterline_temp, resamplingStep=0.8, radiusFactor=1.25, N=60, nodes=5000, remesh=True)

print('Tube generation')
structures = []
i = 0
for i in range(3):
    print(i)
    structures1 = dict()
    structures1['BranchTargetCoord'] = branchTargetCoord[i]
    structures1['Original_Centerline'] = centerlines_list[i+1]

    clnew1, pt_ref1 = vmtkut.modify_branch_centerline(tubeAo, centerlines_list[i+1], cl0)
    structures1['Modified_Centerline'] = clnew1
    structures1['Pt_Ref'] = pt_ref1

    clEnds_tract_list1 = vmtkut.get_top_cl_group(clnew1, pt_ref1)
    structures1['Cl_End_Tract'] = clEnds_tract_list1

    tube1 = mut.create_tubeall_finale(clnew1, resamplingStep=0.7, radiusFactor=0.9, N=80, nodes=1000, remesh=True)
    structures1['Tube'] = tube1

    edge1 = mut.extract_branch_edge(tubeAo, tube1)
    structures1['Edge'] = edge1
    structures.append(structures1)

structures = mut.sort_edges_modify2(structures, cl0)

if show_plot:
    p = pv.Plotter(shape=(1,2))
    p.subplot(0,0)
    p.add_mesh(surface, color='pink', opacity=0.6)
    p.add_mesh(tubeAo, color='white', opacity=0.7)
    p.add_mesh(structures[0]['Tube'], color='green',  opacity=1)
    p.add_mesh(structures[1]['Tube'], color='red',  opacity=1)
    p.add_mesh(structures[2]['Tube'], color='blue',  opacity=1)

    p.subplot(0,1)
    p.add_mesh(surface, color='white', opacity=0.6)
    p.add_mesh(structures[0]['Edge'], color='green',  line_width=4)
    p.add_mesh(structures[1]['Edge'], color='red',  line_width=4)
    p.add_mesh(structures[2]['Edge'], color='blue',  line_width=4)
    p.link_views()
    p.show()


#-----------------------------------------------------------------------------------------------------------------------
## Extract PA centerlines
#-----------------------------------------------------------------------------------------------------------------------
if seed_detection=='ventricle-based':
    rv_surf.compute_implicit_distance(polm_surf, inplace=True)
    pv_region = rv_surf.threshold_percent(percent=0.1, scalars='implicit_distance', invert=True)
    pv_center = np.array(pv_region.center)
    pv_radius = np.mean([np.linalg.norm(pt - pv_center) for pt in pv_region.points])
    pv_normal = pv.fit_plane_to_points(pv_region.points).point_normals.mean(0)
    # adjust av_nornal
    if pv_normal[2] < 0: pv_normal = -1 * pv_normal

    if show_plot:
        pl = pv.Plotter()
        pl.add_mesh(polm_surf, color='blue', opacity=0.5, show_edges=False)
        pl.add_mesh(rv_surf, color='lightblue', opacity=0.5, show_edges=False)
        pl.add_mesh(pv_region, color='red', opacity=1)
        pl.show()


    #-----------------------------------------------------------------------------------------------------------------------
    ## Detect centerlines keypoints: pulmonary artery
    #-----------------------------------------------------------------------------------------------------------------------
    print('Centerline seed detection...')
    source_pa_pt, _ = polm_surf.ray_trace(pv_center-20*pv_normal, pv_center+20*pv_normal)
    st_id = polm_surf.find_closest_point(source_pa_pt.T)
    source_pa_pt = polm_surf.points[st_id]

    # Detect aortic centerline endpoints at the pulmonary artery branches
    G_pa = mut.convert_triangle_mesh_to_graph(polm_surf)
    shortest_paths_lengths = nx.single_source_shortest_path_length(G_pa, st_id)
    polm_surf['Geodesics'] = np.zeros((polm_surf.n_points,))
    for j in range(polm_surf.n_points):
        geo_path = shortest_paths_lengths[j]
        polm_surf['Geodesics'][j] = geo_path
    polm_surf['Geodesics_n'] = (polm_surf['Geodesics'] - np.min(polm_surf['Geodesics'])) / np.ptp(polm_surf['Geodesics'])

    # Detect ends: recursive thresholding of geodesic distance
    delta = 0.05
    cand_outlets = []
    for delta in np.arange(delta, 1, delta):
        thr = polm_surf.threshold_percent(1-delta, scalars='Geodesics_n').connectivity()
        for j in np.unique(thr['RegionId']):
            thr_connected = thr.threshold([j, j+0.5], scalars='RegionId')
            cand_outlets.append(np.array(thr_connected.points[np.argmax(thr_connected['Geodesics_n'])]))
    branchTargetCoord_polm = np.unique(cand_outlets, axis=0)

    if len(branchTargetCoord_polm) > 2:
        branchTargetCoord_polm = mut.trova_distanza_massima(branchTargetCoord_polm)

    PT_cl, _ = vmtkut.extract_PT_centerline(polm_surf, source_pa_pt, branchTargetCoord_polm[0], branchTargetCoord_polm[1])
    if show_plot:
        p = pv.Plotter(notebook=0)
        p.add_mesh(ao_surf, color='white', opacity=0.4)
        p.add_mesh(polm_surf, color='white', opacity=0.4)
        p.add_points(branchTargetCoord_polm[0], color='r')
        p.add_points(branchTargetCoord_polm[1], color='black')
        p.show()

    #-----------------------------------------------------------------------------------------------------------------------
    ## Detect pulmonary artery bifurcation
    #-----------------------------------------------------------------------------------------------------------------------
    ab = np.asarray(parentCenterline["Abscissas"], dtype=float)
    max_ab = float(ab.max())
    if ab[0] != 0:
        parentCenterline["Abscissas"] = max_ab - ab

    cl0_n = pv.wrap(parentCenterline)
    _, centerlinesPT = vmtkut.divide_surface(polm_surf, PT_cl)
    cl0PT, cl1PT, cl2PT, cl3PT = vmtkut.separate_centerlines(centerlinesPT)
    bifPT, pTg, aotg, tgVec = mut.find_bifurcation_pt(cl0PT, cl1PT, cl0_n, thr=2)
    if reorinet:
        ao_z0_cands = cl0_n.slice(normal=tgVec, origin=bifPT)
    else:
        ao_z0_cands = cl0_n.slice(normal=[0,0,1], origin=bifPT)

    if show_plot:
        pl = pv.Plotter()
        pl.add_mesh(ao_z0_cands, color='r', point_size=20)
        pl.add_mesh(cl0_n, color='black')
        pl.show()

    dcand0 = np.linalg.norm(bifPT - ao_z0_cands.points[0, :])
    dcand1 = np.linalg.norm(bifPT - ao_z0_cands.points[1, :])
    z0Id = np.argmin([dcand0, dcand1])
    z0_coord = ao_z0_cands.points[z0Id]
    landmarkZ0_cl0_coord, landmarkZ0_cl0_idx = mut.find_closest_point_on_surface(z0_coord, cl0_n)

    landmarkZ0 = dict()
    landmarkZ0['coord'] = landmarkZ0_cl0_coord
    landmarkZ0['abscissa'] = cl0_n['Abscissas'][landmarkZ0_cl0_idx]

    if show_plot:
        p = pv.Plotter(notebook=0)
        p.add_mesh(ao_surf, color='white', opacity=0.4)
        p.add_mesh(polm_surf, color='white', opacity=0.4)
        p.add_points(np.array(bifPT), color='red', point_size=10)
        p.add_points(np.array(landmarkZ0_cl0_coord), color='yellow', point_size=10)
        p.add_mesh(PT_cl, color='black', line_width=5)
        p.add_mesh(cl0_n, color='black', line_width=5)
        p.show(full_screen=False)

#-----------------------------------------------------------------------------------------------------------------------
## Landing zone detetion
#-----------------------------------------------------------------------------------------------------------------------
ab = np.asarray(parentCenterline["Abscissas"], dtype=float)
max_ab = float(ab.max())
if ab[0] != 0:
    parentCenterline["Abscissas"] = max_ab - ab
cl0_n = pv.wrap(parentCenterline)
pStartB1, pEndB1 = mut.get_branch_landmarks(structures[0]['Edge'], cl0_n, surface)
pStartB2, pEndB2 = mut.get_branch_landmarks(structures[1]['Edge'], cl0_n, surface)
pStartB3, pEndB3 = mut.get_branch_landmarks(structures[2]['Edge'], cl0_n, surface)
landmarksB1 = [pStartB1, pEndB1]
landmarksB2 = [pStartB2, pEndB2]
landmarksB3 = [pStartB3, pEndB3]

if seed_detection=='ventricle-based':
    landmarks_on_cl0 = []
    if pEndB1['abscissa'] < pStartB2['abscissa']:
        landmarks_on_cl0 = [landmarkZ0, pEndB1, pEndB2, pEndB3]
    else:
        landmarks_on_cl0 = [landmarkZ0, pEndB2, pEndB3]
    z3Length = 20
    if len(landmarks_on_cl0) == 4:
        zoneIds = [0, 1, 2, 3]
    elif len(landmarks_on_cl0) == 3:
        zoneIds = [0, 2, 3]

else:
    landmarks_on_cl0 = []
    if pEndB1['abscissa'] < pStartB2['abscissa']:
        landmarks_on_cl0 = [pEndB1, pEndB2, pEndB3]
    else:
        landmarks_on_cl0 = [pEndB2, pEndB3]
    z3Length = 20
    if len(landmarks_on_cl0) == 3:
        zoneIds = [1, 2, 3]
    elif len(landmarks_on_cl0) == 2:
        zoneIds = [2, 3]


zoneId_array = -1 * np.ones(cl0_n.GetNumberOfPoints())

cl0_abscissas = cl0_n['Abscissas']
for i in range(len(landmarks_on_cl0) - 1):
    zone_i_idx = np.where(np.logical_and(cl0_abscissas >= landmarks_on_cl0[i]['abscissa'],
                                         cl0_abscissas < landmarks_on_cl0[i+1]['abscissa']))
    zoneId_array[zone_i_idx] = zoneIds[i]

zone_3_idx = np.where(np.logical_and(cl0_abscissas >= landmarks_on_cl0[-1]['abscissa'],
                                     cl0_abscissas < landmarks_on_cl0[-1]['abscissa'] + z3Length))

zoneId_array[zone_3_idx] = 3
cl0_n['ZoneIds'] = zoneId_array
cl0_n['Radius'] = cl0_n['MaximumInscribedSphereRadius']

if show_plot:
    pl = pv.Plotter()
    pl.add_mesh(ao_surf, color='white', opacity=0.5)
    pl.add_mesh(cl0_n, scalars='ZoneIds', cmap='viridis', show_scalar_bar=True, line_width=5)
    pl.show()

#-----------------------------------------------------------------------------------------------------------------------
## Clip aorta proximally
#-----------------------------------------------------------------------------------------------------------------------
abscissas = np.asarray(parentCenterline["Abscissas"], dtype=float)
if not abscissas[0]==0:
    maxab = np.max(abscissas)
    abscissas = maxab - abscissas
parentCenterline["Abscissas"] = abscissas

cut_mm = 5.0
cutoff_idx = np.max(np.where(parentCenterline['Abscissas'] < 5))
step = float(np.mean(np.linalg.norm(np.diff(parentCenterline.points, axis=0), axis=1)))
cyl_h = max(3.0, 3.0 * step)

t = np.asarray(parentCenterline["FrenetTangent"][cutoff_idx], dtype=float)
t /= (np.linalg.norm(t) + 1e-12)

cyl_center = parentCenterline.points[cutoff_idx] - t * (cyl_h / 2.0)

cut_cyl = pv.Cylinder(
    center=cyl_center,
    direction=t,
    radius=2.8 * float(parentCenterline["MaximumInscribedSphereRadius"][cutoff_idx]),
    height=cyl_h
)

cut_cyl_inlet = cut_cyl.copy(deep=True)

clipped_ao_clean = ao_surf.clip_surface(cut_cyl, invert=False).connectivity("largest")
clipped_parentCenterline = parentCenterline.clip_surface(cut_cyl, invert=False).clean().connectivity("largest")
clipped_centerlines = centerlines.clip_surface(cut_cyl, invert=False).connectivity("largest")

inlet_landmark = parentCenterline.points[cutoff_idx]

if show_plot:
    pl = pv.Plotter()
    pl.add_mesh(clipped_ao_clean, color="w", opacity=0.8)
    #pl.add_mesh(parentCenterline, color="k", line_width=3, opacity=0.4)
    pl.add_mesh(clipped_parentCenterline, color="k", line_width=6)
    pl.add_mesh(clipped_centerlines, color="r", line_width=4)
    pl.show_axes()
    pl.show()

#-----------------------------------------------------------------------------------------------------------------------
## Clip branches
#-----------------------------------------------------------------------------------------------------------------------
ao_surf_clipped_branches = clipped_ao_clean.copy(deep=True)
print('Clipping aorta at branches...')
cut_len_sovra = 60.0       # lunghezza da preservare dopo la biforcazione
tol = 1.0       # tolleranza per considerare "in comune" con parent centerline (mm)
cyl_h = 4.0
cyl_radius_factor = 1.2

# KDTree sulla parent centerline (già clippata inlet + tratto iniziale)
parent_pts = np.asarray(clipped_parentCenterline.points)
parent_tree = cKDTree(parent_pts)

ao_surf_clipped_sovra = ao_surf_clipped_branches.copy(deep=True)
cut_cyl_sovra = []
bif_landmarks = []
cut_landmarks = []


for cid in np.unique(centerlines["RegionId"]):
    dummy = pv.PolyData(ao_surf_clipped_sovra.points, ao_surf_clipped_sovra.faces)
    cl = centerlines.threshold([cid, cid + 0.5], scalars="RegionId").extract_surface()
    ab = np.asarray(cl["Abscissas"], dtype=float)
    pts = np.asarray(cl.points)
    dists, _ = parent_tree.query(pts, k=1)
    close = np.where(dists < tol)[0]

    if close.size == 0:
        bif_idx = int(np.argmin(dists))
    else:
        bif_idx = int(close[np.argmax(ab[close])])

    bif_pt = pts[bif_idx]
    bif_landmarks.append(bif_pt)

    # Find index afer 40 mm length
    target_ab = ab[bif_idx] + cut_len_sovra
    idx_after = np.where(ab >= target_ab)[0]

    if idx_after.size > 0:
        cutoff_idx = int(idx_after[0])
    else:
        cutoff_idx = cl.n_points - 5

    # Fallback to avoid taking the last point or teh bifurcation point
    cutoff_idx = min(cutoff_idx, cl.n_points - 2)
    cutoff_idx = max(cutoff_idx, bif_idx + 1)

    cut_pt = pts[cutoff_idx]
    cut_landmarks.append(cut_pt)
    tangent = np.asarray(cl["FrenetTangent"][cutoff_idx], dtype=float)
    tangent /= (np.linalg.norm(tangent) + 1e-12)

    #slice to determine the radius of each cylinder

    slice_ = dummy.slice(normal=tangent, origin=cut_pt).connectivity()
    region_ids = np.unique(slice_["RegionId"])
    dcenter = []
    for rid in region_ids:
        thr = slice_.threshold([rid, rid + 0.5], scalars="RegionId", preference="point").extract_surface()
        dcenter.append(np.linalg.norm(np.asarray(thr.center) - cut_pt))
    region2keep = region_ids[int(np.argmin(dcenter))]
    thr = slice_.threshold([region2keep, region2keep + 0.5], scalars="RegionId", preference="point").extract_surface()

    radius_for_cut = float(np.max(np.linalg.norm(np.asarray(thr.points) - cut_pt, axis=1)))
    radius_for_cut = max(radius_for_cut, 1e-3)

    cut_cyl = pv.Cylinder(
        center=cut_pt + tangent * (cyl_h / 2.0),
        direction=tangent,
        radius=radius_for_cut * cyl_radius_factor,
        height=cyl_h
    )
    cut_cyl_sovra.append(cut_cyl)

    # clip: come prima, connectivity('largest') tiene il pezzo principale e scarta il distale staccato
    ao_surf_clipped_sovra = ao_surf_clipped_sovra.clip_surface(cut_cyl, invert=False).connectivity("largest")
    clipped_centerlines   = clipped_centerlines.clip_surface(cut_cyl, invert=False).clean().connectivity("largest")

    if show_plot:
        pl = pv.Plotter()
        pl.add_mesh(ao_surf_clipped_sovra, color="w", opacity=0.5)
        pl.add_mesh(clipped_parentCenterline, color="k", line_width=5)
        pl.add_points(np.array([bif_pt]), color="orange", point_size=15, render_points_as_spheres=True)
        pl.add_points(np.array([cut_pt]), color="red", point_size=15, render_points_as_spheres=True)
        pl.add_mesh(cut_cyl, color="blue", opacity=0.4)
        pl.add_mesh(clipped_centerlines, color="yellow", line_width=5)
        pl.show_axes()
        pl.show()

ao_surf_clipped_branches = ao_surf_clipped_sovra
outlet_sovra_landmark = cut_landmarks

if show_plot:
    pl = pv.Plotter()
    pl.add_mesh(ao_surf_clipped_branches, color='w', opacity=0.5)
    pl.add_mesh(clipped_parentCenterline, color='k', line_width=5)
    pl.add_mesh(np.array(outlet_sovra_landmark), color='r', line_width=5)
    pl.add_mesh(clipped_centerlines, color='yellow', line_width=5)
    pl.add_title(pid)
    pl.show()

#-----------------------------------------------------------------------------------------------------------------------
## Clip descending aorta
#-----------------------------------------------------------------------------------------------------------------------
cut_mm_desc = 10
length_max = parentCenterline['Abscissas'][-1]
cutoff_idx = np.max(np.where(parentCenterline['Abscissas'] < length_max - cut_mm_desc))
step = float(np.mean(np.linalg.norm(np.diff(parentCenterline.points, axis=0), axis=1)))
cyl_h = max(3.0, 3.0 * step)

t = np.asarray(parentCenterline["FrenetTangent"][cutoff_idx], dtype=float)
t /= (np.linalg.norm(t) + 1e-12)

cyl_center = parentCenterline.points[cutoff_idx] - t * (cyl_h / 2.0)
cut_cyl = pv.Cylinder(
    center=cyl_center,
    direction=t,
    radius=2.8 * float(parentCenterline["MaximumInscribedSphereRadius"][cutoff_idx]),
    height=cyl_h
)

ao_surf_clipped_all = ao_surf_clipped_branches.clip_surface(cut_cyl, invert=False).connectivity("largest")
clipped_parentCenterline = clipped_parentCenterline.clip_surface(cut_cyl, invert=False).clean().connectivity("largest")

if show_plot:
    pl = pv.Plotter()
    pl.add_mesh(ao_surf_clipped_all, color="w", opacity=0.8)
    pl.add_mesh(clipped_parentCenterline, color="k", line_width=6)
    pl.add_mesh(clipped_centerlines, color="r", line_width=4)
    pl.show_axes()
    pl.show()

#-----------------------------------------------------------------------------------------------------------------------
## Reordering of the centerlines
#-----------------------------------------------------------------------------------------------------------------------
branches_cl = [cl1, cl2, cl3]

parent_pts = np.asarray(clipped_parentCenterline.points)
parent_ab  = np.asarray(clipped_parentCenterline["Abscissas"], dtype=float)
parent_tree_bif = cKDTree(parent_pts)

bif_pts = np.asarray(bif_landmarks)
d, idx = parent_tree_bif.query(bif_pts, k=1)
bif_ab = parent_ab[idx]

order = np.argsort(bif_ab)
bif_ab = bif_ab[order]
cut_cyl_sovra = [cut_cyl_sovra[i] for i in order]
branches_cl = [branches_cl[i] for i in order]

cl1_n = pv.wrap(branches_cl[0]).clip_surface(cut_cyl_sovra[0], invert=False).connectivity('largest')
cl2_n = pv.wrap(branches_cl[1]).clip_surface(cut_cyl_sovra[1], invert=False).connectivity('largest')
cl3_n = pv.wrap(branches_cl[2]).clip_surface(cut_cyl_sovra[2], invert=False).connectivity('largest')

## Save data
clipped_parentCenterline.save(osp.join(out_dir, f'cl0.vtp'))
pv.wrap(cl1_n).save(osp.join(out_dir, f'cl1.vtp'))
pv.wrap(cl2_n).save(osp.join(out_dir, f'cl2.vtp'))
pv.wrap(cl3_n).save(osp.join(out_dir, f'cl3.vtp'))
ao_surf_clipped_all = mut.fill_small_holes(ao_surf_clipped_all, nbe=10)
ao_surf_clipped_all.save(os.path.join(out_dir,'surf.vtp'))
print('End processing')
