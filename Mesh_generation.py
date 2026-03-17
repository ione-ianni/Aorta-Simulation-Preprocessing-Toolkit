import sys
sys.path.append('path to the utils folder')
import os.path as osp
import numpy as np
import os
import pyvista as pv
from scipy.spatial import KDTree
from scipy.interpolate import RBFInterpolator
from collections import defaultdict
from scipy.spatial import cKDTree
import mesh_utils as mut


#-----------------------------------------------------------------------------------------------------------------------
## OPTIONS
#-----------------------------------------------------------------------------------------------------------------------
pid = ''
data_dir = '' # where to save mesh data
os.makedirs(data_dir, exist_ok=True)
model_dir = '' # path to the aortic surface model
out_dir = fr'{data_dir}\mesh'
lumen_out_dir = osp.join(out_dir, 'lumen_mesh')
lumen_surf_out_dir = osp.join(lumen_out_dir, 'mesh-surfaces')
solid_out_dir = osp.join(out_dir, 'solid_mesh')
solid_surf_out_dir = osp.join(solid_out_dir, 'mesh-surfaces')
os.makedirs(out_dir, exist_ok=True)
os.makedirs(lumen_out_dir, exist_ok=True)
os.makedirs(lumen_surf_out_dir, exist_ok=True)
os.makedirs(solid_out_dir, exist_ok=True)
os.makedirs(solid_surf_out_dir, exist_ok=True)
show_plot = False
thickness_region = True

#-----------------------------------------------------------------------------------------------------------------------
## Guarantee flat faces at the inlet and outlets by extrusion
#-----------------------------------------------------------------------------------------------------------------------
mesh = pv.read(osp.join(model_dir, f'aorta_surf.vtp'))
mesh = mut.mmg_remesh(mesh, hausd=0.3, hmax=1.2, hmin=0.8, max_aspect_ratio=5, max_iter=3, verbose=True)
feature_edges_p = mesh.extract_feature_edges(80, boundary_edges=True, feature_edges=False,manifold_edges=False).connectivity()
merged_points = mesh.points
merged_faces = mesh.faces
control = mut.meshfix(mesh)

for i in np.unique(feature_edges_p['RegionId']):
    if i == 0:
        extrusion_length = 2.0
    else:
        extrusion_length = 1.0
    fn = feature_edges_p.threshold([i, i + 0.5], scalars='RegionId').delaunay_2d()
    n = np.mean(fn.cell_normals, axis=0)
    g = np.mean(fn.points, axis=0)
    fn = feature_edges_p.threshold([i, i + 0.5], scalars='RegionId').extract_surface()
    if pv.PolyData(g+5*n).select_enclosed_points(control)['SelectedPoints'].sum()==0:
        extruded = mut.fill_small_holes(fn.extrude(n * extrusion_length, capping=False), nbe=10).clean()
    else:
        extruded = mut.fill_small_holes(fn.extrude(-n * extrusion_length, capping=False), nbe=10).clean()

    if show_plot:
        pl = pv.Plotter()
        pl.add_mesh(mesh, color='w', opacity=1, show_edges=True)
        pl.add_points(fn.points, color='black')
        pl.add_mesh(extruded, color='b', opacity=1, show_edges=True)
        pl.add_points(extruded.points, color='red')
        pl.show()

    # Find coinciding points between `extruded` and `mesh`
    t = 1e-3
    distances = np.linalg.norm(extruded.points[:, None, :] - mesh.points[None, :, :], axis=2)
    idx = np.argmin(distances, axis=1)
    mask = np.min(distances, axis=1) < t
    coinciding_indices = idx[mask]

    # Create a mapping from `extruded` points to `mesh` points and update the connectivity matrix
    point_map = {np.where(mask)[0][i]: coinciding_indices[i] for i in range(len(coinciding_indices))}
    point_map_not = {np.where(~mask)[0][i]: len(merged_points)+i for i in range(np.size(extruded.points, axis=0) - len(coinciding_indices))}
    id_map = np.arange(len(merged_points) + len(extruded.points))
    id_map[list(point_map.keys())] = list(point_map.values())
    id_map[list(point_map_not.keys())] = list(point_map_not.values())
    new_connectivity = id_map[extruded.faces.reshape(-1, 4)[:, 1:]]
    new_connectivity = np.hstack([[3] + list(face) for face in new_connectivity])

    # Create merged mesh points and faces
    merged_points = np.vstack([merged_points, extruded.points[~mask]])
    merged_faces = np.hstack([merged_faces, new_connectivity])

# Guarantee high-quality mesh to ensure no errors in 3D meshing
merged_mesh = pv.PolyData(merged_points, merged_faces)
target_ar = 3.0
max_cycles = 15
cycle = 0
curr_ar = mut.get_max_aspect_ratio(merged_mesh)
while curr_ar > target_ar and cycle < max_cycles:
    merged_mesh = mut.fill_small_holes(merged_mesh, nbe=5)
    print('ok')

    # Remesh (gmsh) + (optional) smoothing
    merged_mesh = mut.gmsh_remesh1(merged_mesh,size=0.5,dim=2,is_open=True,max_aspect_ratio=5, max_iter=1, verbose=True)
    merged_mesh = mut.mmg_remesh(merged_mesh, hausd=0.3, hmax=0.8, hmin=0.6, max_aspect_ratio=2, max_iter=3,verbose=True)
    curr_ar = mut.get_max_aspect_ratio(merged_mesh)
    cycle += 1
merged_mesh = pv.wrap(mut.smoothVtk(merged_mesh, nIter=500, passband=0.8, featureAngle=120))

if show_plot:
    pl = pv.Plotter()
    pl.add_mesh(merged_mesh, color='r', opacity=1, show_edges=True)
    pl.show()

#-----------------------------------------------------------------------------------------------------------------------
## Mesh the aorta for the FSI simulation
#-----------------------------------------------------------------------------------------------------------------------
size = 1.2           # target elmennt size
lumen_bl = 4         # number of boundary layer in the lumen domain
wall_bl = 2          # number of layers in aortic wall
lumen_t = 1
wall_t = 0.9         # set this value to 1 if variable thickness, otherwise set to the thickness value that you want
lumen_el = 'tetra'
wall_el = 'tetra'
mesh_complete = mut.gmsh_remesh_fsi(merged_mesh, size=size, lumen_bl=lumen_bl,  wall_bl=wall_bl, lumen_t=lumen_t, wall_t=wall_t, lumen_el=lumen_el, wall_el=wall_el)
mesh_complete.save(osp.join(out_dir, 'mesh-complete.mesh.vtu'))
mesh_complete = pv.read(osp.join(out_dir, 'mesh-complete.mesh.vtu'))

mesh_complete.save(f'{pid}.vtu')
cl0 = pv.read(osp.join(model_dir, f'cl0.vtp'))
cl1 = pv.read(osp.join(model_dir, f'cl1.vtp'))
cl2 = pv.read(osp.join(model_dir, f'cl2.vtp'))
cl3 = pv.read(osp.join(model_dir, f'cl3.vtp'))

#-----------------------------------------------------------------------------------------------------------------------
## Create the vtp and vtu files for the lumen and solid meshes
#-----------------------------------------------------------------------------------------------------------------------
# project fields on mesh-complete
mesh_complete.points = np.floor(mesh_complete.points*10000)/10000
mask_lumen = np.where(np.logical_or(mesh_complete.cell_data['gmsh:physical'] == 2, mesh_complete.cell_data['gmsh:physical'] == 3))
mask_solid = np.where(mesh_complete.cell_data['gmsh:physical'] == 1)
mesh_complete.clear_data()
mesh_complete.cell_data['GlobalElementID'] = np.zeros(mesh_complete.n_cells).astype(int)
e_id_1 = np.arange(1, np.size(mask_lumen,1) + 1).astype(int)
e_id_2 = np.arange(np.size(mask_lumen,1) + 1, np.size(mask_lumen,1) + np.size(mask_solid,1) + 1).astype(int)
mesh_complete.cell_data['GlobalElementID'][mask_lumen] = e_id_1
mesh_complete.cell_data['GlobalElementID'][mask_solid] = e_id_2
mesh_complete.cell_data['ModelRegionID'] = np.zeros(mesh_complete.n_cells)
mesh_complete.cell_data['ModelRegionID'][mask_lumen] = 1
mesh_complete.cell_data['ModelRegionID'][mask_solid] = 2

# lumen mesh: mesh-complete.mesh
lumen_mesh_complete = mesh_complete.extract_cells(mask_lumen)
lumen_mesh_complete.point_data['GlobalNodeID'] = np.arange(1, lumen_mesh_complete.n_points + 1)

# solid mesh: mesh-complete.mesh
solid_mesh_complete = mesh_complete.extract_cells(mask_solid)
solid_mesh_complete.point_data['GlobalNodeID'] = np.arange(1, solid_mesh_complete.n_points + 1)

# lumen mesh surfaces
skin_tot = mesh_complete.extract_surface()
surfaces = skin_tot.extract_cells(np.where(skin_tot.cell_data['ModelRegionID'] == 1)).connectivity()
tree = KDTree(lumen_mesh_complete.points)
_, common_node_ids = tree.query(surfaces.points)
surfaces.point_data['GlobalNodeID'] = common_node_ids + 1

# identify the surfaces: project the barycenter of the surfaces on the main centerline
region_ids = np.unique(surfaces['RegionId'])
s = [surfaces.threshold([i, i + 0.5], scalars='RegionId') for i in region_ids]
g = np.array([np.mean(i.points, axis=0) for i in s])
tree = KDTree(cl0.points)
_, ids = tree.query(g)
sort = np.argsort(ids)
inlet_id = sort[-1]
outlet_disc_id = sort[0]
outlet_bct_id = sort[1]
outlet_car_id = sort[2]
outlet_suc_id = sort[3]

inlet_cell_id = np.where(surfaces.cell_data['RegionId']==inlet_id)
outlet_disc_cell_id = np.where(surfaces.cell_data['RegionId']==outlet_disc_id)
outlet_bct_cell_id = np.where(surfaces.cell_data['RegionId']==outlet_bct_id)
outlet_car_cell_id = np.where(surfaces.cell_data['RegionId']==outlet_car_id)
outlet_suc_cell_id = np.where(surfaces.cell_data['RegionId']==outlet_suc_id)

if show_plot:
    pl = pv.Plotter()
    pl.add_mesh(skin_tot, color='w', opacity=0.5)
    pl.add_mesh(cl0, color='k', line_width=2)
    pl.add_mesh(surfaces.threshold([inlet_id, inlet_id + 0.5], scalars='RegionId'), color='blue', opacity=1)
    pl.add_points(cl0.points[ids[inlet_id]], color='blue', point_size=10, render_points_as_spheres=True)
    pl.add_mesh(surfaces.threshold([outlet_disc_id, outlet_disc_id + 0.5], scalars='RegionId'), color='red', opacity=1)
    pl.add_points(cl0.points[ids[outlet_disc_id]], color='red', point_size=10, render_points_as_spheres=True)
    pl.add_mesh(surfaces.threshold([outlet_bct_id, outlet_bct_id + 0.5], scalars='RegionId'), color='green', opacity=1)
    pl.add_points(cl0.points[ids[outlet_bct_id]], color='green', point_size=10, render_points_as_spheres=True)
    pl.add_mesh(surfaces.threshold([outlet_car_id, outlet_car_id + 0.5], scalars='RegionId'), color='orange', opacity=1)
    pl.add_points(cl0.points[ids[outlet_car_id]], color='orange', point_size=10, render_points_as_spheres=True)
    pl.add_mesh(surfaces.threshold([outlet_suc_id, outlet_suc_id + 0.5], scalars='RegionId'), color='purple', opacity=1)
    pl.add_points(cl0.points[ids[outlet_suc_id]], color='purple', point_size=10, render_points_as_spheres=True)
    pl.show()

# lumen mesh: inlet
lumen_inlet = mut.extract_patch(surfaces, inlet_id, face_id=1)

# lumen mesh: outlets
outlets = [("bct",  outlet_bct_id),("car",  outlet_car_id), ("suc",  outlet_suc_id),("disc", outlet_disc_id),]
lumen_outlets = {}
for i, (name, rid) in enumerate(outlets, start=2):   # 2..N+1
    lumen_outlets[name] = mut.extract_patch(surfaces, rid, face_id=i)

lumen_bct_outlet  = lumen_outlets.get("bct")
lumen_car_outlet  = lumen_outlets.get("car")
lumen_suc_outlet  = lumen_outlets.get("suc")
lumen_disc_outlet = lumen_outlets.get("disc")

# lumen mesh: wall
lumen_mesh_complete_exterior = lumen_mesh_complete.extract_surface()

if lumen_el == 'tetra':
    inlet_cell_ids = lumen_inlet.cell_data['GlobalElementID']
    outlet_bct_cell_ids = lumen_bct_outlet.cell_data['GlobalElementID']
    outlet_car_cell_ids = lumen_car_outlet.cell_data['GlobalElementID']
    outlet_suc_cell_ids = lumen_suc_outlet.cell_data['GlobalElementID']
    outlet_disc_cell_ids = lumen_disc_outlet.cell_data['GlobalElementID']
    remove = np.concatenate([inlet_cell_ids, outlet_bct_cell_ids, outlet_car_cell_ids, outlet_suc_cell_ids, outlet_disc_cell_ids])
    mask = ~np.isin(lumen_mesh_complete_exterior.cell_data['GlobalElementID'], remove)
    lumen_wall = lumen_mesh_complete_exterior.extract_cells(mask).extract_surface()
else:
    lumen_wall = lumen_mesh_complete_exterior.extract_cells_by_type(pv.CellType.TRIANGLE).extract_surface()

# Crete filed necessary for the SimVascular pipeline
tree = KDTree(lumen_mesh_complete.points)
_, common_node_ids = tree.query(lumen_wall.points)
lumen_wall.point_data['GlobalNodeID'] = common_node_ids + 1
lumen_wall.cell_data['ModelFaceID'] = np.ones(lumen_wall.n_cells)*6
lumen_wall.cell_data['Normals'] = lumen_wall.cell_normals

# lumen mesh: mesh-complete.exterior
tree = KDTree(lumen_mesh_complete.points)
_, common_node_ids = tree.query(lumen_mesh_complete_exterior.points)
lumen_mesh_complete_exterior.point_data['GlobalNodeID'] = common_node_ids + 1

inlet_cell_ids = lumen_inlet.point_data['GlobalNodeID']
outlet_bct_cell_ids = lumen_bct_outlet.point_data['GlobalNodeID']
outlet_car_cell_ids = lumen_car_outlet.point_data['GlobalNodeID']
outlet_suc_cell_ids = lumen_suc_outlet.point_data['GlobalNodeID']
outlet_disc_cell_ids = lumen_disc_outlet.point_data['GlobalNodeID']

regions = {
    1: inlet_cell_ids,
    2: outlet_bct_cell_ids,
    3: outlet_car_cell_ids,
    4: outlet_suc_cell_ids,
    5: outlet_disc_cell_ids,
}

mut.assign_model_face_ids(lumen_mesh_complete_exterior,regions=regions, lumen_el=lumen_el,default_id=6, global_id_key="GlobalNodeID")
lumen_mesh_complete_exterior.cell_data['Normals'] = lumen_mesh_complete_exterior.cell_normals

# solid mesh: walls_combined_connected_region_1
solid_mesh_complete_exterior = solid_mesh_complete.extract_surface()
tree = KDTree(solid_mesh_complete_exterior.points)
_, solid_wall_1_id = tree.query(lumen_wall.points)
solid_wall_1 = solid_mesh_complete_exterior.extract_points(solid_wall_1_id,  adjacent_cells=False).extract_surface()
solid_wall_1.cell_data['GlobalElementID'] = solid_wall_1.cell_data['GlobalElementID']-lumen_mesh_complete.n_cells
solid_wall_1.cell_data['ModelFaceID'] = np.ones(solid_wall_1.n_cells)*6
solid_wall_1.cell_data['Normals'] = solid_wall_1.cell_normals

# identify the surfaces: find the nodes belonging to the lateral boundary layer
mask = ~np.isin(np.arange(0,solid_mesh_complete_exterior.n_points), solid_wall_1_id)
temp = solid_mesh_complete_exterior.extract_points(mask).extract_surface()
feature_edges_p = temp.extract_feature_edges(80, boundary_edges=True, feature_edges=False,manifold_edges=False)
tree = KDTree(temp.points)
_, bl_points = tree.query(feature_edges_p.points)
for _ in range(max(1,wall_bl)):
    neighbors = temp.extract_points(bl_points, adjacent_cells=True)
    _, bl_points = tree.query(neighbors.points)
surfaces = temp.extract_points(bl_points, adjacent_cells=False).connectivity()
surfaces.cell_data['GlobalElementID'] = surfaces.cell_data['GlobalElementID']-lumen_mesh_complete.n_cells

# identify the surfaces: project the barycenter of the surfaces on the main centerline
region_ids = np.unique(surfaces['RegionId'])
s = [surfaces.threshold([i, i + 0.5], scalars='RegionId') for i in region_ids]
g = np.array([np.mean(i.points, axis=0) for i in s])
tree = KDTree(cl0.points)
_, ids = tree.query(g)
sort = np.argsort(ids)
inlet_id = sort[-1]
outlet_disc_id = sort[0]
outlet_bct_id = sort[1]
outlet_car_id = sort[2]
outlet_suc_id = sort[3]

inlet_cell_id = np.where(surfaces.cell_data['RegionId'] == inlet_id)
outlet_disc_cell_id = np.where(surfaces.cell_data['RegionId'] == outlet_disc_id)
outlet_bct_cell_id = np.where(surfaces.cell_data['RegionId'] == outlet_bct_id)
outlet_car_cell_id = np.where(surfaces.cell_data['RegionId'] == outlet_car_id)
outlet_suc_cell_id = np.where(surfaces.cell_data['RegionId'] == outlet_suc_id)

if show_plot:
    pl = pv.Plotter()
    pl.add_mesh(solid_mesh_complete_exterior, color='w', opacity=0.5, show_edges=False)
    pl.add_mesh(cl0, color='k', line_width=2)
    pl.add_mesh(surfaces.threshold([inlet_id, inlet_id + 0.5], scalars='RegionId'), color='blue', opacity=1)
    pl.add_points(cl0.points[ids[inlet_id]], color='blue', point_size=10, render_points_as_spheres=True)
    pl.add_mesh(surfaces.threshold([outlet_disc_id, outlet_disc_id + 0.5], scalars='RegionId'), color='red', opacity=1)
    pl.add_points(cl0.points[ids[outlet_disc_id]], color='red', point_size=10, render_points_as_spheres=True)
    pl.add_mesh(surfaces.threshold([outlet_bct_id, outlet_bct_id + 0.5], scalars='RegionId'), color='green', opacity=1)
    pl.add_points(cl0.points[ids[outlet_bct_id]], color='green', point_size=10, render_points_as_spheres=True)
    pl.add_mesh(surfaces.threshold([outlet_car_id, outlet_car_id + 0.5], scalars='RegionId'), color='orange', opacity=1)
    pl.add_points(cl0.points[ids[outlet_car_id]], color='orange', point_size=10, render_points_as_spheres=True)
    pl.add_mesh(surfaces.threshold([outlet_suc_id, outlet_suc_id + 0.5], scalars='RegionId'), color='purple', opacity=1)
    pl.add_points(cl0.points[ids[outlet_suc_id]], color='purple', point_size=10, render_points_as_spheres=True)
    pl.show()

# solid mesh: inlet
solid_inlet = mut.extract_patch(surfaces, inlet_id, face_id=1)

outlets = [("bct",  outlet_bct_id),("car",  outlet_car_id),("suc",  outlet_suc_id),("disc", outlet_disc_id),]

solid_outlets = {}
for i, (name, rid) in enumerate(outlets, start=2):  # 2..N+1
    solid_outlets[name] = mut.extract_patch(surfaces, rid, face_id=i)
solid_bct_outlet  = solid_outlets.get("bct")
solid_car_outlet  = solid_outlets.get("car")
solid_suc_outlet  = solid_outlets.get("suc")
solid_disc_outlet = solid_outlets.get("disc")

# solid mesh: walls_combined_connected_region_0
inlet_cell_ids = solid_inlet.point_data['GlobalNodeID']
outlet_bct_cell_ids = solid_bct_outlet.point_data['GlobalNodeID']
outlet_car_cell_ids = solid_car_outlet.point_data['GlobalNodeID']
outlet_suc_cell_ids = solid_suc_outlet.point_data['GlobalNodeID']
outlet_disc_cell_ids = solid_disc_outlet.point_data['GlobalNodeID']
wall_1_ids = solid_wall_1.point_data['GlobalNodeID']
remove = np.concatenate([inlet_cell_ids, outlet_bct_cell_ids, outlet_car_cell_ids, outlet_suc_cell_ids, outlet_disc_cell_ids, wall_1_ids])
mask = ~np.isin(solid_mesh_complete_exterior.point_data['GlobalNodeID'], remove)
wall_0_ids = np.where(mask)[0]
solid_wall_0 = solid_mesh_complete_exterior.extract_points(mask, adjacent_cells=True).extract_surface()

solid_wall_0.cell_data['ModelFaceID'] = np.ones(solid_wall_0.n_cells)*6
solid_wall_0.cell_data['Normals'] = solid_wall_0.cell_normals

if show_plot:
    pl = pv.Plotter()
    pl.add_mesh(solid_wall_1, color='white', opacity=1, show_edges=False)
    pl.add_mesh(solid_wall_0, color='blue', opacity=1, show_edges=True)
    pl.show()



#-----------------------------------------------------------------------------------------------------------------------
## variable thickness in the aortic wall
#-----------------------------------------------------------------------------------------------------------------------
if thickness_region:
    # Precompute cell-centers KDTree for region membership test
    cell_centers = solid_mesh_complete.cell_centers().points  # (N_cells, 3)
    centers_tree = cKDTree(cell_centers)

    thick_asc = 1.5
    thick_disc = 1.5

    # Assign thickness to nodes on the inner surface
    feature_edges_p = solid_wall_1.extract_feature_edges(80, boundary_edges=True, feature_edges=False,
                                                         manifold_edges=False).connectivity()
    region_ids = np.unique(feature_edges_p['RegionId'])
    s = [feature_edges_p.threshold([i, i + 0.5], scalars='RegionId') for i in region_ids]
    g = np.array([np.mean(i.points, axis=0) for i in s])
    tree = KDTree(cl0.points)
    _, ids = tree.query(g)
    sort = np.argsort(ids)
    inlet_id = sort[-1]
    outlet_disc_id = sort[0]
    outlet_bct_id = sort[3]
    outlet_car_id = sort[2]
    outlet_suc_id = sort[1]

    global_id_inlet = feature_edges_p.threshold([inlet_id, inlet_id + 0.5])['GlobalNodeID']
    global_id_outlet_disc = feature_edges_p.threshold([outlet_disc_id, outlet_disc_id + 0.5])['GlobalNodeID']
    global_id_outlet_bct = feature_edges_p.threshold([outlet_bct_id, outlet_bct_id + 0.5])['GlobalNodeID']
    global_id_outlet_car = feature_edges_p.threshold([outlet_car_id, outlet_car_id + 0.5])['GlobalNodeID']
    global_id_outlet_suc = feature_edges_p.threshold([outlet_suc_id, outlet_suc_id + 0.5])['GlobalNodeID']

    # find the final part of the ascending aorta as the central part of the ZoneIds = -1 in the ascending aorta
    asc_end_id = np.max(np.where(cl0['ZoneIds'] != -1)[0])  # start of landing zone
    asc_end_id = asc_end_id + int((len(cl0.points) - asc_end_id) / 2)

    # slice the aorta
    normal = cl0['FrenetTangent'][asc_end_id]
    pt = cl0.points[asc_end_id]
    sections = solid_wall_1.slice(normal=normal, origin=pt).connectivity()
    section_asc = sections.threshold([0, 0.5], scalars='RegionId')
    section_desc = sections.threshold([0.5, 1], scalars='RegionId')
    if np.linalg.norm(section_desc.center - pt) < np.linalg.norm(section_asc.center - pt):
        section_asc, section_desc = section_desc, section_asc

    # find the closest nodes between this section and the inlet
    tree = KDTree(solid_wall_1.points)
    _, id_asc_end = tree.query(section_asc.points)
    _, id_desc_start = tree.query(section_desc.points)

    if show_plot:
        pl = pv.Plotter()
        pl.add_mesh(solid_wall_1, color='w', opacity=0.5)
        pl.add_mesh(cl0, color='k', line_width=2)
        pl.add_mesh(feature_edges_p.threshold([inlet_id, inlet_id + 0.5], scalars='RegionId'), color='blue', opacity=1,
                    line_width=5)
        pl.add_points(cl0.points[ids[inlet_id]], color='blue', point_size=10, render_points_as_spheres=True)
        pl.add_mesh(feature_edges_p.threshold([outlet_disc_id, outlet_disc_id + 0.5], scalars='RegionId'), color='red',
                    opacity=1, line_width=5)
        pl.add_points(cl0.points[ids[outlet_disc_id]], color='red', point_size=10, render_points_as_spheres=True)
        pl.add_mesh(feature_edges_p.threshold([outlet_bct_id, outlet_bct_id + 0.5], scalars='RegionId'), color='green',
                    opacity=1, line_width=5)
        pl.add_points(cl0.points[ids[outlet_bct_id]], color='green', point_size=10, render_points_as_spheres=True)
        pl.add_mesh(feature_edges_p.threshold([outlet_car_id, outlet_car_id + 0.5], scalars='RegionId'), color='orange',
                    opacity=1, line_width=5)
        pl.add_points(cl0.points[ids[outlet_car_id]], color='orange', point_size=10, render_points_as_spheres=True)
        pl.add_mesh(feature_edges_p.threshold([outlet_suc_id, outlet_suc_id + 0.5], scalars='RegionId'), color='purple',
                    opacity=1, line_width=5)
        pl.add_points(cl0.points[ids[outlet_suc_id]], color='purple', point_size=10, render_points_as_spheres=True)
        pl.add_mesh(section_asc, color='blue', opacity=1, line_width=5)
        pl.add_mesh(section_desc, color='red', opacity=1, line_width=5)
        pl.show()

    solid_wall_1['Thickness'] = np.zeros(solid_wall_1.n_points)
    solid_wall_1['Thickness'][np.isin(solid_wall_1['GlobalNodeID'], global_id_inlet)] = thick_asc
    solid_wall_1['Thickness'][np.isin(solid_wall_1['GlobalNodeID'], global_id_outlet_disc)] = thick_disc
    solid_wall_1['Thickness'][np.isin(solid_wall_1['GlobalNodeID'], global_id_outlet_bct)] = 0.5
    solid_wall_1['Thickness'][np.isin(solid_wall_1['GlobalNodeID'], global_id_outlet_car)] = 0.5
    solid_wall_1['Thickness'][np.isin(solid_wall_1['GlobalNodeID'], global_id_outlet_suc)] = 0.5
    solid_wall_1['Thickness'][id_asc_end] = thick_asc
    solid_wall_1['Thickness'][id_desc_start] = thick_disc


    # fit RBF to interpolate the thickness values on the entire inner surface
    fixed_mask = solid_wall_1['Thickness'] > 0  # nodes with assigned thickness
    known_coords = solid_wall_1.points[fixed_mask]
    known_values = solid_wall_1['Thickness'][fixed_mask]
    rbf = RBFInterpolator(known_coords, known_values, kernel="thin_plate_spline")
    node_thickness = rbf(solid_wall_1.points)
    solid_wall_1['Thickness'] = node_thickness
    solid_wall_1.save(f'ciao_{pid}.vtp')
    gids = solid_mesh_complete.point_data["GlobalNodeID"]

    # Assume GlobalNodeID is integer array-like
    # Build dict: id_val -> list of indices in volume mesh
    gid_to_vids = {}
    for vid, gid in enumerate(gids):
        gid_to_vids.setdefault(int(gid), []).append(vid)

    # Also build maps for inner and outer surfaces: GlobalNodeID -> point index in surface
    inner_gids = solid_wall_1.point_data["GlobalNodeID"]
    outer_gids = solid_wall_0.point_data["GlobalNodeID"]

    inner_gid_to_idx = {int(g): idx for idx, g in enumerate(inner_gids)}
    outer_gid_to_idx = {int(g): idx for idx, g in enumerate(outer_gids)}

    vol_pts = np.array(solid_mesh_complete.points)
    gids_vol = np.array(solid_mesh_complete.point_data["GlobalNodeID"], dtype=int)
    gids_inner = np.array(solid_wall_1.point_data["GlobalNodeID"], dtype=int)
    normals_inner = np.array(solid_wall_1.point_normals)
    pts_inner = np.array(solid_wall_1.points)
    connectivity = solid_mesh_complete.cells.reshape(-1, 5)[:, 1:]  # (N_cells, 4) connectivity of volume mesh

    # Find the adjacency list of the volume mesh (neighboring nodes for each node)
    adjacency = defaultdict(set)
    for tet in connectivity:
        for i in range(4):
            for j in range(i + 1, 4):
                adjacency[tet[i]].add(tet[j])
                adjacency[tet[j]].add(tet[i])

    # Prepare arrays for vectorized processing
    pts_inner = np.array(solid_wall_1.points)  # (N_inner, 3)
    normals_inner = np.array(solid_wall_1.point_normals)  # (N_inner, 3)
    inner_gids = np.array(solid_wall_1.point_data["GlobalNodeID"], dtype=int) - 1
    vol_pts = np.array(solid_mesh_complete.points)

    n_inner = len(inner_gids)
    idx1_all = np.empty(n_inner, dtype=int)
    idx2_all = np.empty(n_inner, dtype=int)

    # Vectorized neighbor selection
    for k, gid in enumerate(inner_gids):
        Pi = pts_inner[k]
        n = normals_inner[k]

        # 1) candidates around inner node
        neighs = list(adjacency[gid])
        if not neighs:
            idx1_all[k] = gid
            idx2_all[k] = gid
            continue

        diffs = vol_pts[neighs] - Pi  # shape (n_neighs, 3)
        diffs /= np.linalg.norm(diffs, axis=1)[:, None]
        best1 = np.argmax(np.abs(diffs @ n))
        idx1 = neighs[best1]

        # 2) candidates around mid node
        neighs2 = list(adjacency[idx1] - {gid})
        if not neighs2:
            idx1_all[k] = idx1
            idx2_all[k] = idx1
            continue

        diffs2 = vol_pts[neighs2] - Pi
        diffs2 /= np.linalg.norm(diffs2, axis=1)[:, None]
        best2 = np.argmax(np.abs(diffs2 @ n))
        idx2 = neighs2[best2]

        idx1_all[k] = idx1
        idx2_all[k] = idx2

    # Ensure normals point outward consistently
    dirs = vol_pts[idx1_all] - pts_inner
    flip = (np.einsum("ij,ij->i", normals_inner, dirs) < 0)
    normals_fixed = normals_inner.copy()
    normals_fixed[flip] *= -1

    new_mid = pts_inner + normals_fixed * (node_thickness[:, None] * 0.5)
    new_outer = pts_inner + normals_fixed * node_thickness[:, None]

    # 9) Update mesh in bulk --------------------------------------
    solid_mesh_complete.points[idx1_all] = new_mid
    solid_mesh_complete.points[idx2_all] = new_outer

    # remove degenerate tetra
    solid_mesh_complete_exterior = solid_mesh_complete.extract_surface()

    tree = KDTree(solid_mesh_complete_exterior.points)
    _, solid_wall_1_id = tree.query(lumen_wall.points)
    solid_wall_1 = solid_mesh_complete_exterior.extract_points(solid_wall_1_id, adjacent_cells=False).extract_surface()

    solid_wall_1.cell_data['GlobalElementID'] = solid_wall_1.cell_data['GlobalElementID'] - lumen_mesh_complete.n_cells
    solid_wall_1.cell_data['ModelFaceID'] = np.ones(solid_wall_1.n_cells) * 6
    solid_wall_1.cell_data['Normals'] = solid_wall_1.cell_normals

    # identify the surfaces: find the nodes belonging to the lateral boundary layer
    mask = ~np.isin(np.arange(0, solid_mesh_complete_exterior.n_points), solid_wall_1_id)
    temp = solid_mesh_complete_exterior.extract_points(mask).extract_surface()
    feature_edges_p = temp.extract_feature_edges(80, boundary_edges=True, feature_edges=False, manifold_edges=False)
    tree = KDTree(temp.points)
    _, bl_points = tree.query(feature_edges_p.points)
    for _ in range(max(1, wall_bl)):
        neighbors = temp.extract_points(bl_points, adjacent_cells=True)
        _, bl_points = tree.query(neighbors.points)
    surfaces = temp.extract_points(bl_points, adjacent_cells=False).connectivity()

    surfaces.cell_data['GlobalElementID'] = surfaces.cell_data['GlobalElementID'] - lumen_mesh_complete.n_cells

    # identify the surfaces: project the barycenter of the surfaces on the main centerline
    region_ids = np.unique(surfaces['RegionId'])
    s = [surfaces.threshold([i, i + 0.5], scalars='RegionId') for i in region_ids]
    g = np.array([np.mean(i.points, axis=0) for i in s])
    tree = KDTree(cl0.points)
    _, ids = tree.query(g)
    sort = np.argsort(ids)
    inlet_id = sort[-1]
    outlet_disc_id = sort[0]
    outlet_bct_id = sort[3]
    outlet_car_id = sort[2]
    outlet_suc_id = sort[1]

    inlet_cell_id = np.where(surfaces.cell_data['RegionId'] == inlet_id)
    outlet_disc_cell_id = np.where(surfaces.cell_data['RegionId'] == outlet_disc_id)
    outlet_bct_cell_id = np.where(surfaces.cell_data['RegionId'] == outlet_bct_id)
    outlet_car_cell_id = np.where(surfaces.cell_data['RegionId'] == outlet_car_id)
    outlet_suc_cell_id = np.where(surfaces.cell_data['RegionId'] == outlet_suc_id)

    if show_plot:
        pl = pv.Plotter()
        pl.add_mesh(solid_mesh_complete_exterior, color='w', opacity=0.5, show_edges=False)
        pl.add_mesh(cl0, color='k', line_width=2)
        pl.add_mesh(surfaces.threshold([inlet_id, inlet_id + 0.5], scalars='RegionId'), color='blue', opacity=1)
        pl.add_points(cl0.points[ids[inlet_id]], color='blue', point_size=10, render_points_as_spheres=True)
        pl.add_mesh(surfaces.threshold([outlet_disc_id, outlet_disc_id + 0.5], scalars='RegionId'), color='red',
                    opacity=1)
        pl.add_points(cl0.points[ids[outlet_disc_id]], color='red', point_size=10, render_points_as_spheres=True)
        pl.add_mesh(surfaces.threshold([outlet_bct_id, outlet_bct_id + 0.5], scalars='RegionId'), color='green',
                    opacity=1)
        pl.add_points(cl0.points[ids[outlet_bct_id]], color='green', point_size=10, render_points_as_spheres=True)
        pl.add_mesh(surfaces.threshold([outlet_car_id, outlet_car_id + 0.5], scalars='RegionId'), color='orange',
                    opacity=1)
        pl.add_points(cl0.points[ids[outlet_car_id]], color='orange', point_size=10, render_points_as_spheres=True)
        pl.add_mesh(surfaces.threshold([outlet_suc_id, outlet_suc_id + 0.5], scalars='RegionId'), color='purple',
                    opacity=1)
        pl.add_points(cl0.points[ids[outlet_suc_id]], color='purple', point_size=10, render_points_as_spheres=True)
        pl.show()

    # solid mesh: inlet
    solid_inlet = mut.extract_patch(surfaces, inlet_id, face_id=1)

    outlets = [("bct", outlet_bct_id),("car", outlet_car_id),("suc", outlet_suc_id),("disc", outlet_disc_id),]

    solid_outlets = {
        name: mut.extract_patch(surfaces, rid, face_id=i)
        for i, (name, rid) in enumerate(outlets, start=2)  # 2..N+1
    }
    solid_bct_outlet = solid_outlets.get("bct")
    solid_car_outlet = solid_outlets.get("car")
    solid_suc_outlet = solid_outlets.get("suc")
    solid_disc_outlet = solid_outlets.get("disc")

    # solid mesh: walls_combined_connected_region_0
    inlet_cell_ids = solid_inlet.point_data['GlobalNodeID']
    outlet_bct_cell_ids = solid_bct_outlet.point_data['GlobalNodeID']
    outlet_car_cell_ids = solid_car_outlet.point_data['GlobalNodeID']
    outlet_suc_cell_ids = solid_suc_outlet.point_data['GlobalNodeID']
    outlet_disc_cell_ids = solid_disc_outlet.point_data['GlobalNodeID']
    wall_1_ids = solid_wall_1.point_data['GlobalNodeID']

    remove = np.concatenate(
        [inlet_cell_ids, outlet_bct_cell_ids, outlet_car_cell_ids, outlet_suc_cell_ids, outlet_disc_cell_ids,
         wall_1_ids])
    mask = ~np.isin(solid_mesh_complete_exterior.point_data['GlobalNodeID'], remove)
    wall_0_ids = np.where(mask)[0]
    solid_wall_0 = solid_mesh_complete_exterior.extract_points(mask, adjacent_cells=True).extract_surface()

    solid_wall_0.cell_data['GlobalElementID'] = solid_wall_0.cell_data['GlobalElementID'] - lumen_mesh_complete.n_cells #riga aggiunta da me
    solid_wall_0.cell_data['ModelFaceID'] = np.ones(solid_wall_0.n_cells) * 6
    solid_wall_0.cell_data['Normals'] = solid_wall_0.cell_normals

    if show_plot:
        pl = pv.Plotter()
        pl.add_mesh(solid_wall_1, color='white', opacity=1, show_edges=False)
        pl.add_mesh(solid_wall_0, color='blue', opacity=1, show_edges=True)
        pl.show()

    # Update global mesh
    mesh_complete.points[np.unique(mesh_complete.cells.reshape(-1, 5)[:, 1:][mesh_complete['ModelRegionID'] == 2])] = solid_mesh_complete.points


mesh_complete.save(osp.join(out_dir, 'mesh-complete.mesh.vtu'))
mesh_complete = pv.read(osp.join(out_dir, 'mesh-complete.mesh.vtu'))
solid_wall = solid_wall_0.merge(solid_wall_1)

regions = {
    1: inlet_cell_ids,
    2: outlet_bct_cell_ids,
    3: outlet_car_cell_ids,
    4: outlet_suc_cell_ids,
    5: outlet_disc_cell_ids,
}

mut.assign_model_face_ids(solid_mesh_complete_exterior, regions=regions,lumen_el=wall_el,default_id=6, global_id_key="GlobalNodeID",)
solid_mesh_complete_exterior.cell_data['Normals'] = solid_mesh_complete_exterior.cell_normals

# save
mesh_complete.save(osp.join(out_dir, 'mesh-complete.mesh.vtu'))

lumen_mesh_complete.save(osp.join(lumen_out_dir, 'mesh-complete.mesh.vtu'))
lumen_mesh_complete_exterior.save(osp.join(lumen_out_dir, 'mesh-complete.exterior.vtp'))
lumen_inlet.save(osp.join(lumen_surf_out_dir, 'inlet.vtp'))
lumen_bct_outlet.save(osp.join(lumen_surf_out_dir, 'outlet_1.vtp'))
lumen_car_outlet.save(osp.join(lumen_surf_out_dir, 'outlet_2.vtp'))
lumen_suc_outlet.save(osp.join(lumen_surf_out_dir, 'outlet_3.vtp'))
lumen_disc_outlet.save(osp.join(lumen_surf_out_dir, 'outlet_4.vtp'))
lumen_wall.save(osp.join(lumen_surf_out_dir, 'wall.vtp'))
lumen_wall.save(osp.join(lumen_out_dir, 'walls_combined.vtp'))
lumen_mesh_complete_exterior.save(osp.join(lumen_out_dir, 'mesh-complete.exterior.vtp'))

solid_mesh_complete.save(osp.join(solid_out_dir, 'mesh-complete.mesh.vtu'))
solid_wall_0.save(osp.join(solid_out_dir, 'walls_combined_connected_region_0.vtp'))
solid_wall_1.save(osp.join(solid_out_dir, 'walls_combined_connected_region_1.vtp'))
solid_mesh_complete_exterior.save(osp.join(solid_out_dir, 'mesh-complete.exterior.vtp'))
solid_inlet.save(osp.join(solid_surf_out_dir, 'inlet.vtp'))
solid_bct_outlet.save(osp.join(solid_surf_out_dir, 'outlet_1.vtp'))
solid_car_outlet.save(osp.join(solid_surf_out_dir, 'outlet_2.vtp'))
solid_suc_outlet.save(osp.join(solid_surf_out_dir, 'outlet_3.vtp'))
solid_disc_outlet.save(osp.join(solid_surf_out_dir, 'outlet_4.vtp'))
solid_wall.save(osp.join(solid_surf_out_dir, 'wall.vtp'))

