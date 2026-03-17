import pymeshfix as mf
import subprocess
import networkx as nx
import scipy
import math
import gmsh
import trimesh
from pymeshfix._meshfix import PyTMesh
from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import vmtk_utils as vmtkut
from pyFM.mesh import TriMesh
from vtk_utils import *
import pyacvd
import time
from scipy.signal import savgol_filter
from scipy.interpolate import splev, splder, splrep
import SimpleITK as sitk
import meshio

def meshfix(mesh):
    meshfix = mf.MeshFix(pv.PolyData(mesh))
    meshfix.repair(verbose=False)
    mesh = meshfix.mesh
    return mesh


def mmg_remesh(input_mesh, hausd=0.3, hmax=2, hmin=1.5, max_aspect_ratio=None, max_iter=3, verbose=False):
    #crea un idetificatore univoco per la mesh
    mesh_id = os.getpid()
    input_mesh.clear_data()
    #salva la mesh in formato mesh
    pv.save_meshio(f'{mesh_id}.mesh', input_mesh)
    #esegue il comando mmg con i paramteri specificati per eseguire il remeshing
    subprocess.run([r"path to .. \utils\mmg\bin\mmgs_O3.exe",
                f'{mesh_id}.mesh',
                '-hausd', str(hausd),
                '-hmax', str(hmax),
                '-hmin', str(hmin),
                '-nr',
                '-nreg',
                '-xreg'
                '-optim',
                f'{mesh_id}_remeshed.mesh'], stdout=subprocess.DEVNULL)

    new_mesh = meshio.read(f'{mesh_id}_remeshed.mesh')
    pvmesh = pv.utilities.from_meshio(new_mesh)
    #controlla qualità del remeshing
    if max_aspect_ratio is not None:
        qual = pvmesh.compute_cell_quality(quality_measure='aspect_ratio')
        it = 0
        while np.max(qual['CellQuality']) > max_aspect_ratio and it < max_iter:
            it += 1

            subprocess.run([r"path to .. \utils\mmg\bin\mmgs_O3.exe",
            f'{mesh_id}.mesh',
            '-hausd', str(hausd * 2),
            '-hmax', str(hmax),
            '-hmin', str(hmin),
            '-nr',
            '-nreg',
            '-xreg'
            '-optim',
            f'{mesh_id}_remeshed.mesh'], stdout=subprocess.DEVNULL)

            new_mesh = meshio.read(f'{mesh_id}_remeshed.mesh')
            pvmesh = pv.utilities.from_meshio(new_mesh)
            qual = pvmesh.compute_cell_quality(quality_measure='aspect_ratio')

        if verbose: print('Max aspect ratio:', np.max(qual['CellQuality']))
    os.remove(f'{mesh_id}.mesh')
    os.remove(f'{mesh_id}_remeshed.mesh')
    os.remove(f'{mesh_id}_remeshed.sol')
    return pvmesh.extract_surface()


def smoothVtk(surface, nIter=1000, passband=0.001, featureAngle=120.0):
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputData(surface)
    smoother.SetNumberOfIterations(nIter)
    smoother.BoundarySmoothingOn()
    smoother.FeatureEdgeSmoothingOff()
    smoother.SetFeatureAngle(featureAngle)
    smoother.SetPassBand(passband)
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()
    return smoother.GetOutput()


def fill_small_holes(input_mesh, nbe=100):
    prid = os.getpid()
    #salva la mesh formato stl
    input_mesh.save(f'ao_{prid}.stl')

    mfix = PyTMesh(False)  # False removes extra verbose output
    mfix.load_file(f'ao_{prid}.stl')
    os.remove(f'ao_{prid}.stl')
    #riempimento dei piccoli buchi utlizzando l'algoritmo specificato
    mfix.fill_small_boundaries(nbe=nbe, refine=True)
    vert, faces = mfix.return_arrays()
    triangles = np.empty((faces.shape[0], 4), dtype=faces.dtype)
    triangles[:, -3:] = faces
    triangles[:, 0] = 3

    mesh = pv.PolyData(vert, triangles)
    return mesh

def find_closest_point_on_surface(ref_pt, surface):
    tree = vtk.vtkKdTreePointLocator()
    tree.SetDataSet(surface)
    tree.BuildLocator()
    pt_id = tree.FindClosestPoint(ref_pt)
    pt_coord = np.array(surface.GetPoint(pt_id))
    return pt_coord, pt_id



#this function compute the laplace beltrami eigen vector ans stored teh in an array in the polydata surface
def compute_LB_eigenvectors(ao_surf, num_eigs=5, normalize=True, fn='out.off'):
    write_polydata_to_off(ao_surf, fn) #scrittura superfice su file OFF
    mesh = TriMesh(fn, area_normalize=True, center=True) #crea oggeto trimesh della sup salvata e processa la mesh per caloclare i primi autovettori di Laplace-Beltrami
    mesh.process(k=num_eigs, intrinsic=False, verbose=False)
    os.remove(fn)
    ao_surf = pv.wrap(ao_surf)
    #if normlize is set to true the eignvector are normalize between -1 and +1
    if normalize:
        for e in range(num_eigs):
            ao_surf[f'eigvec{e}'] = (mesh.eigenvectors[:, e] - np.min(mesh.eigenvectors[:, e])) / \
                             (np.max(mesh.eigenvectors[:, e]) - np.min(mesh.eigenvectors[:, e]))
    else:
        for e in range(num_eigs):
            ao_surf[f'eigvec{e}'] = mesh.eigenvectors[:, e]

    if ao_surf.points[np.argmax(ao_surf['eigvec1']), 2] > ao_surf.points[np.argmin(ao_surf['eigvec1']), 2]:
        evec = np.max(ao_surf['eigvec1']) - ao_surf['eigvec1']
        ao_surf.clear_data()
        ao_surf['eigvec1'] = evec
    return ao_surf



#this funxtion compute the heat kernel signature for a given surface
def compute_hks_v2(surfaceAo, t=50, fn='ao.vtp'):
    write_polydata(surfaceAo, fn)
    subprocess.run(['python.exe', r" path to hks.py",'--input', fn, '--t', str(t), '--output', fn])
    surfaceAo = pv.read(fn)
    os.remove(fn)
    #normaliza i valori Hsk per essere nel range [0,1]
    surfaceAo['hks'] = (surfaceAo['hks'] - np.min(surfaceAo['hks'])) / (np.max(surfaceAo['hks']) - np.min(surfaceAo['hks'])) #superfice contiene anche iondo HKS
    return surfaceAo


def pick_3_points_max_triangle(points: np.ndarray, min_area=1e-6):
    p1 = points[0]
    d1 = np.linalg.norm(points - p1, axis=1)
    i2 = int(np.argmax(d1))
    p2 = points[i2]

    d2 = np.linalg.norm(points - p2, axis=1)
    perim = d1 + d2

    order = np.argsort(perim)[::-1]
    p3 = points[order[0]]
    for idx in order[:200]:
        cand = points[idx]
        area2 = np.linalg.norm(np.cross(p2 - p1, cand - p1))
        if area2 > min_area:
            p3 = cand
            break

    return p1, p2, p3


def convert_triangle_mesh_to_graph(mesh):
    node_list = [list(mesh.points[j]) for j in range(mesh.n_points)]
    node_dict = dict()
    for j, n in enumerate(node_list):
        node_dict[j] = n
    faces = mesh.faces.reshape(-1, 4)
    edges0 = faces[:, 1:3]  # edges AB
    edges1 = faces[:, (1, 3)]  # edges AC
    edges2 = faces[:, 2:]  # edges BC
    edges = np.vstack((edges0, edges1, edges2))
    G = nx.Graph()
    G.add_nodes_from(node_dict)
    G.add_edges_from(edges)
    return G


def calculate_frenet(sampled_indices, xcoefs_3D, ycoefs_3D, zcoefs_3D):

    # First derivative (tangent)
    dxcoefs_3D = splder(xcoefs_3D)
    dycoefs_3D = splder(ycoefs_3D)
    dzcoefs_3D = splder(zcoefs_3D)

    # Evaluate at location
    tangent_x = splev(sampled_indices, dxcoefs_3D)
    tangent_y = splev(sampled_indices, dycoefs_3D)
    tangent_z = splev(sampled_indices, dzcoefs_3D)

    # Second derivative (curvature)
    ddxcoefs_3D = splder(xcoefs_3D,2)
    ddycoefs_3D = splder(ycoefs_3D,2)
    ddzcoefs_3D = splder(zcoefs_3D,2)

    # Evaluate at location
    curvature_x = splev(sampled_indices, ddxcoefs_3D)
    curvature_y = splev(sampled_indices, ddycoefs_3D)
    curvature_z = splev(sampled_indices, ddzcoefs_3D)

    # Compute Frenet tern for points
    t = np.column_stack([tangent_x, tangent_y, tangent_z])
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    curvatures = np.column_stack([curvature_x, curvature_y, curvature_z])
    b = np.cross(t, curvatures)
    b /= np.linalg.norm(b, axis=1, keepdims=True)

    # Fix binormals
    '''for i in range(1, len(b)):
        if np.dot(b[i], b[i - 1]) < 0:

            b[i] = -b[i]'''

    n = np.cross(b, t)
    return t,b,n


def filter_centerline(cl, window_size, polyorder, resampling):
    radius = cl['MaximumInscribedSphereRadius']
    show_plot = False
    skel_new = np.array(cl.points)
    # Savitzky-Golay filter
    if window_size % 2 == 0:

        window_size += 1
    skel_new_filt = savgol_filter(skel_new, window_length=window_size, polyorder=polyorder, axis=0)

    # Cumulative abscissa
    skel_new_filt_abscissa = np.concatenate(([0], np.cumsum(np.sqrt(np.sum(np.diff(skel_new_filt, axis=0) ** 2, axis=1)))))

    # Fitting (cubic)
    xcoefs_3D = splrep(skel_new_filt_abscissa, skel_new_filt[:, 0], k=3)
    ycoefs_3D = splrep(skel_new_filt_abscissa, skel_new_filt[:, 1], k=3)
    zcoefs_3D = splrep(skel_new_filt_abscissa, skel_new_filt[:, 2], k=3)

    # Resampling
    #t_3D = np.arange(skel_new_filt_abscissa[0], skel_new_filt_abscissa[-1] + 1, resampling)
    tt_3D = skel_new_filt_abscissa

    # Evaluation at resample points
    skel_new_filt_x = splev(tt_3D, xcoefs_3D)
    skel_new_filt_y = splev(tt_3D, ycoefs_3D)
    skel_new_filt_z = splev(tt_3D, zcoefs_3D)


    skel_new = np.column_stack([skel_new_filt_x, skel_new_filt_y, skel_new_filt_z])
    smoothed_centerline_poly = pv.PolyData(skel_new)
    smoothed_centerline_poly.lines = np.hstack(([len(skel_new)], np.arange(len(skel_new))))
    smoothed_centerline_poly = vmtkut.add_centerline_geometry(smoothed_centerline_poly)
    smoothed_centerline_poly = pv.wrap(smoothed_centerline_poly)

    # Compute Frenet csys
    tangents, binormals, normals = calculate_frenet(tt_3D, xcoefs_3D, ycoefs_3D, zcoefs_3D)
    smoothed_centerline_poly.point_data['FrenetTangent'] = tangents
    smoothed_centerline_poly.point_data['FrenetBinormal'] = binormals
    smoothed_centerline_poly.point_data['FrenetNormal'] = normals
    smoothed_centerline_poly.point_data['MaximumInscribedSphereRadius'] = radius

    if show_plot==True:

        pl = pv.Plotter()
        pl.add_mesh(cl, color='red', line_width=2,
                    label='Original centerline')  # Centerline originale in rosso

        pl.add_mesh(smoothed_centerline_poly, color='green', line_width=2,
                    label='Smoothed centerline')  # Centerline levigata in verde

        # Aggiungi le tangenti al plot
        for i, point in enumerate(skel_new):
            tangent = tangents[i] * 2
            pl.add_arrows(np.array([point]), np.array([tangent]), mag=0.6, color='blue')
            binormal = binormals[i]*2
            pl.add_arrows(np.array([point]), np.array([binormal]), mag=0.6, color='red')

        pl.add_legend()
        pl.show()

    return smoothed_centerline_poly



def orient_centerline_by_seeds(cl: pv.PolyData, source_pt):
    cl = pv.wrap(cl).copy(deep=True)
    p0 = np.asarray(cl.points[0], dtype=float)
    pN = np.asarray(cl.points[-1], dtype=float)
    s  = np.asarray(source_pt, dtype=float)

    d0 = np.linalg.norm(p0 - s)
    dN = np.linalg.norm(pN - s)

    reverse = dN < d0
    if reverse:
        cl.points = cl.points[::-1].copy()
        for name in list(cl.point_data.keys()):
            arr = np.asarray(cl.point_data[name])
            if arr.shape[0] == cl.n_points:
                cl.point_data[name] = arr[::-1].copy()

    return cl


def extractPoints(source):
    """
    Return points from a polydata as a list of tuples.
    """
    points = source.GetPoints()
    indices = range(points.GetNumberOfPoints())
    pointAccessor = lambda i: points.GetPoint(i)
    return list(map(pointAccessor, indices))


def findClosestSurfacePoint(source, point):
    locator = vtk.vtkKdTreePointLocator()
    locator.SetDataSet(source)
    locator.BuildLocator()

    pId = locator.FindClosestPoint(point)
    return pId

def stitch (edge1,edge2):
    # Extract points along the edge line (in correct order).
    # The following further assumes that the polyline has the
    # same orientation (clockwise or counterclockwise).
    start = time.time()
    points1 = extractPoints(edge1)
    points2 = extractPoints(edge2)
    n1 = len(points1)
    n2 = len(points2)

    # Prepare result containers.
    # Variable points concatenates points1 and points2.
    # Note: all indices refer to this targert container!
    points = vtk.vtkPoints()
    cells = vtk.vtkCellArray()
    points.SetNumberOfPoints(n1+n2)
    for i, p1 in enumerate(points1):
        points.SetPoint(i, p1)
    for i, p2 in enumerate(points2):
        points.SetPoint(i+n1, p2)

    # The following code stitches the curves edge1 with (points1) and
    # edge2 (with points2) together based on a simple growing scheme.

    # Pick a first stitch between points1[0] and its closest neighbor
    # of points2.
    i1Start = 0
    i2Start = findClosestSurfacePoint(edge2, points1[i1Start])
    i2Start += n1 # offset to reach the points2

    # Initialize
    i1 = i1Start
    i2 = i2Start
    p1 = np.asarray(points.GetPoint(i1))
    p2 = np.asarray(points.GetPoint(i2))
    mask = np.zeros(n1+n2, dtype=bool)
    count = 0
    while not np.all(mask):
        count += 1
        i1Candidate = (i1+1)%n1
        i2Candidate = (i2+1-n1)%n2+n1
        p1Candidate = np.asarray(points.GetPoint(i1Candidate))
        p2Candidate = np.asarray(points.GetPoint(i2Candidate))
        diffEdge12C = np.linalg.norm(p1-p2Candidate)
        diffEdge21C = np.linalg.norm(p2-p1Candidate)

        mask[i1] = True
        mask[i2] = True
        if diffEdge12C < diffEdge21C:
            triangle = vtk.vtkTriangle()
            triangle.GetPointIds().SetId(0,i1)
            triangle.GetPointIds().SetId(1,i2)
            triangle.GetPointIds().SetId(2,i2Candidate)
            cells.InsertNextCell(triangle)
            i2 = i2Candidate
            p2 = p2Candidate
        else:
            triangle = vtk.vtkTriangle()
            triangle.GetPointIds().SetId(0,i1)
            triangle.GetPointIds().SetId(1,i2)
            triangle.GetPointIds().SetId(2,i1Candidate)
            cells.InsertNextCell(triangle)
            i1 = i1Candidate
            p1 = p1Candidate

    # Add the last triangle.
    i1Candidate = (i1+1)%n1
    i2Candidate = (i2+1-n1)%n2+n1
    if (i1Candidate <= i1Start) or (i2Candidate <= i2Start):
        if i1Candidate <= i1Start:
            iC = i1Candidate
        else:
            iC = i2Candidate
        triangle = vtk.vtkTriangle()
        triangle.GetPointIds().SetId(0,i1)
        triangle.GetPointIds().SetId(1,i2)
        triangle.GetPointIds().SetId(2,iC)
        cells.InsertNextCell(triangle)

    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetPolys(cells)
    poly.BuildLinks()

    end = time.time()
    return poly


def pyacvdq_remesh_surface(surface, nOfNodes=5000):
    print('Remeshing...')
    clus = pyacvd.Clustering(pv.wrap(surface))
    clus.subdivide(3)
    clus.cluster(nOfNodes)
    surface = clus.create_mesh()
    return surface

def create_tube0(cl, resamplingStep=0.2, radiusFactor=1., N=80, nodes=10000, remesh=True):
    cl = vmtkut.resample_centerline(cl, step=resamplingStep)
    pt = cl.GetNumberOfPoints()
    print(pt)
    combo = vtk.vtkAppendPolyData()

    c = cl.GetPoint(pt - 1)
    r = np.multiply(cl.GetPointData().GetArray('MaximumInscribedSphereRadius').GetTuple1(pt - 1), radiusFactor)
    n = cl.GetPointData().GetArray('FrenetTangent').GetTuple(pt - 1)
    polygonSource = vtk.vtkRegularPolygonSource()
    polygonSource.SetNormal(n)
    polygonSource.SetCenter(c)
    polygonSource.SetRadius(r)
    polygonSource.SetNumberOfSides(N)
    polygonSource.Update()
    polygonSource.GetGeneratePolyline()
    circle = polygonSource.GetOutput()
    combo.AddInputData(circle)
    edge1 = vtk.vtkFeatureEdges()
    edge1.SetInputData(circle)
    edge1.SetBoundaryEdges(1)
    edge1.SetFeatureEdges(1)
    edge1.SetNonManifoldEdges(0)
    edge1.SetManifoldEdges(0)
    edge1.Update()
    boundaryStrips = vtk.vtkStripper()
    boundaryStrips.SetInputConnection(edge1.GetOutputPort())
    boundaryStrips.Update()
    edge1 = boundaryStrips.GetOutput()

    print('\nstitching')
    for i in range(pt - 1):
        c = cl.GetPoint(pt - 2 - i)
        r = np.multiply(cl.GetPointData().GetArray('MaximumInscribedSphereRadius').GetTuple1(pt - i - 2), radiusFactor)
        n = cl.GetPointData().GetArray('FrenetTangent').GetTuple(pt - i - 2)
        polygonSource = vtk.vtkRegularPolygonSource()
        polygonSource.SetNormal(n)
        polygonSource.SetCenter(c)
        polygonSource.SetRadius(r)
        polygonSource.SetNumberOfSides(N)
        polygonSource.Update()
        polygonSource.GetGeneratePolyline()
        circle = polygonSource.GetOutput()

        edge2 = vtk.vtkFeatureEdges()
        edge2.SetInputData(circle)
        edge2.SetBoundaryEdges(1)
        edge2.SetFeatureEdges(0)
        edge2.SetNonManifoldEdges(0)
        edge2.SetManifoldEdges(0)
        edge2.Update()
        boundaryStrips = vtk.vtkStripper()
        boundaryStrips.SetInputConnection(edge2.GetOutputPort())
        boundaryStrips.Update()
        edge2 = boundaryStrips.GetOutput()

        stitching = stitch(edge1, edge2)

        combo.AddInputData(stitching)
        edge1 = edge2

    combo.AddInputData(circle)
    combo.Update()
    cleanFilter = vtk.vtkCleanPolyData()
    cleanFilter.SetInputConnection(combo.GetOutputPort())
    cleanFilter.Update()
    pd = cleanFilter.GetOutput()

    #write_polydata(pd, 'cache/.recc.vtp')

    if remesh:
        pd = pyacvdq_remesh_surface(pd, nOfNodes=5000)
        # reduced = decimate_surface(pd, reduction=0.5)

    return pd



def create_tubeall_finale(cl, resamplingStep=0.2, radiusFactor=1., N=80, nodes=5000, remesh=True):
    combo = vtk.vtkAppendPolyData()
    cl = vmtkut.resample_centerline(cl, step=resamplingStep)
    pt = cl.GetNumberOfPoints()

    c = cl.GetPoint(pt - 1)
    r = np.multiply(cl.GetPointData().GetArray('MaximumInscribedSphereRadius').GetTuple1(pt - 1), radiusFactor)
    n = cl.GetPointData().GetArray('FrenetTangent').GetTuple(pt - 1)
    polygonSource = vtk.vtkRegularPolygonSource()
    polygonSource.SetNormal(n)
    polygonSource.SetCenter(c)
    polygonSource.SetRadius(r)
    polygonSource.SetNumberOfSides(N)
    polygonSource.Update()
    polygonSource.GetGeneratePolyline()
    circle = polygonSource.GetOutput()
    combo.AddInputData(circle)
    edge1 = vtk.vtkFeatureEdges()
    edge1.SetInputData(circle)
    edge1.SetBoundaryEdges(1)
    edge1.SetFeatureEdges(0)
    edge1.SetNonManifoldEdges(0)
    edge1.SetManifoldEdges(0)
    edge1.Update()
    # write_polydata(edge1.GetOutput(), 'feature_Edge.vtp')
    boundaryStrips = vtk.vtkStripper()
    boundaryStrips.SetInputConnection(edge1.GetOutputPort())
    boundaryStrips.Update()
    edge1 = boundaryStrips.GetOutput()
    # write_polydata(edge1, 'stipper.vtp')
    print('\nStitching.')
    for i in range(pt - 1):
        c = cl.GetPoint(pt - 2 - i)
        r = np.multiply(cl.GetPointData().GetArray('MaximumInscribedSphereRadius').GetTuple1(pt - i - 2), radiusFactor)
        n = cl.GetPointData().GetArray('FrenetTangent').GetTuple(pt - i - 2)
        polygonSource = vtk.vtkRegularPolygonSource()
        polygonSource.SetNormal(n)
        polygonSource.SetCenter(c)
        polygonSource.SetRadius(r)
        polygonSource.SetNumberOfSides(N)
        polygonSource.Update()
        polygonSource.GetGeneratePolyline()
        circle = polygonSource.GetOutput()

        edge2 = vtk.vtkFeatureEdges()
        edge2.SetInputData(circle)
        edge2.SetBoundaryEdges(1)
        edge2.SetFeatureEdges(0)
        edge2.SetNonManifoldEdges(0)
        edge2.SetManifoldEdges(0)
        edge2.Update()
        boundaryStrips = vtk.vtkStripper()
        boundaryStrips.SetInputConnection(edge2.GetOutputPort())
        boundaryStrips.Update()
        edge2 = boundaryStrips.GetOutput()

        stitching = stitch(edge1, edge2)

        combo.AddInputData(stitching)
        edge1 = edge2

    combo.AddInputData(circle)
    combo.Update()
    cleanFilter = vtk.vtkCleanPolyData()
    cleanFilter.SetInputConnection(combo.GetOutputPort())
    cleanFilter.Update()
    pd = cleanFilter.GetOutput()

    #write_polydata(pd, '.recc.vtp')

    if remesh:
        # reduced = decimate_surface(pd, reduction=0.5)
        pd = pyacvdq_remesh_surface(pd, nOfNodes=5000)

    return pd



def extract_branch_edge(tubeAo, tubeB):
    inter = vtk.vtkIntersectionPolyDataFilter()
    #inter.CheckMeshOff()
    #inter.CheckInputOff()
    #inter.SetTolerance(0.1)
    inter.SetInputData(0, tubeAo)
    inter.SetInputData(1, tubeB)
    print('Computing intersection.')
    inter.Update()

    print('Computing connectivity.')
    connectivityFilter = vtk.vtkPolyDataConnectivityFilter()
    connectivityFilter.SetInputConnection(inter.GetOutputPort())
    connectivityFilter.SetExtractionModeToClosestPointRegion()
    connectivityFilter.SetClosestPoint(200,0,0)
    connectivityFilter.Update()

    edge = connectivityFilter.GetOutput()
    return  edge



def sort_edges_modify2(structures, cl0):
    absMean = np.empty(len(structures))
    cl0_abscissas = cl0.GetPointData().GetArray('Abscissas')
    for e in range(len(structures)):
        edge = structures[e]['Edge']
        closest_point_abscissas = np.empty(edge.GetNumberOfPoints())
        for p in range(edge.GetNumberOfPoints()):
            point = edge.GetPoint(p)
            dist = np.empty(cl0.GetNumberOfPoints())
            for i in range(cl0.GetNumberOfPoints()):
                d = math.sqrt(vtk.vtkMath.Distance2BetweenPoints(point, cl0.GetPoint(i)))
                dist[i] = d
            closest_point_on_cl0 = np.argmin(dist)
            closest_point_abscissas[p] = cl0_abscissas.GetValue(closest_point_on_cl0)
        absMean[e] = np.mean(np.unique(closest_point_abscissas))
        structures[e]['AbsMean'] = absMean[e]
    sortedTargets = sorted(structures, key=lambda k: k['AbsMean'])

    return sortedTargets

def distanza(p1, p2):
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

def trova_distanza_massima(punti):
    punto1, punto2 = punti[0], punti[1]
    max_distanza = distanza(punto1, punto2)

    for i in range(len(punti)):
        for j in range(i + 1, len(punti)):
            d = distanza(punti[i], punti[j])
            if d > max_distanza:
                max_distanza = d
                punto1, punto2 = punti[i], punti[j]

    return [punto1, punto2]


def calculate_minimum_distance_to_polydata(pt, pd):
    dist = np.empty(pd.GetNumberOfPoints())
    for i in range(pd.GetNumberOfPoints()):
        d = math.sqrt(vtk.vtkMath.Distance2BetweenPoints(pt, pd.GetPoint(i)))
        dist[i] = d
    minDist = np.min(dist)
    minDistPointId = np.argmin(dist)
    return minDist, minDistPointId


def find_bifurcation_pt(cl0, cl1, aorticCenterline, thr=0.8):
    for i in range(cl1.GetNumberOfPoints()):
        pt = cl1.GetPoint(i)
        distanceToCl2, _ = calculate_minimum_distance_to_polydata(pt, cl0)
        if distanceToCl2 > thr:
            #print(i)
            break

    bifPT = pt
    plane = vtk.vtkPlane()
    plane.SetOrigin(bifPT)
    plane.SetNormal((0, 1, 0))
    cutter = vtk.vtkCutter()
    cutter.SetCutFunction(plane)
    cutter.SetInputData(aorticCenterline)
    cutter.Update()
    pTg = cutter.GetOutput().GetPoint(0)

    aotg, idTg = find_closest_point_on_surface(pTg, aorticCenterline)
    tgVec = aorticCenterline.GetPointData().GetArray('FrenetTangent').GetTuple(idTg)

    return bifPT, pTg, aotg, tgVec



def get_branch_landmarks(edge, cl0, surface):
    cl0_abscissas = cl0.GetPointData().GetArray('Abscissas')
    closest_point_abscissas = np.empty(edge.GetNumberOfPoints())
    for p in range(edge.GetNumberOfPoints()):
        point = edge.GetPoint(p)
        dist = np.empty(cl0.GetNumberOfPoints())
        for i in range(cl0.GetNumberOfPoints()):
            d = math.sqrt(vtk.vtkMath.Distance2BetweenPoints(point, cl0.GetPoint(i)))
            dist[i] = d
        closest_point_on_cl0 = np.argmin(dist)
        closest_point_abscissas[p] = cl0_abscissas.GetValue(closest_point_on_cl0)
    landmark_start = dict()
    landmark_start['coord'], _ = find_closest_point_on_surface(edge.GetPoint(np.argmin(closest_point_abscissas)), surface)
    landmark_start['abscissa'] = np.min(closest_point_abscissas)
    landmark_end = dict()
    landmark_end['coord'], _ = find_closest_point_on_surface(edge.GetPoint(np.argmax(closest_point_abscissas)), surface)
    landmark_end['abscissa'] = np.max(closest_point_abscissas)

    return landmark_start, landmark_end

def get_max_aspect_ratio(mesh: pv.PolyData) -> float:
    q = mesh.compute_cell_quality(quality_measure="aspect_ratio")
    ar = np.asarray(q["CellQuality"])
    ar = ar[np.isfinite(ar)]
    return float(ar.max()) if ar.size else np.inf



def gmsh_remesh1(surf, size, dim, is_open, max_aspect_ratio=5, max_iter=3, verbose=True):
    # INPUT
    # - surf = polydata mesh
    # - size = characteristic dimension of the element [mm]
    # - dim = 2 2D mesh; dim = 3D mesh
    # - is_open = True open geometry (eg: cropped aorta); is_open = False closed geometry (eg: calcium, not cropped aorta)

    # OUTPUT
    # surf_remesh = Polydata remeshed mesh

    show_plot = False

    fin = 'surf.stl'
    if dim == 2:
        fout = 'surf_remeshed.stl'
    else:
        fout = 'surf_remeshed.vtu'
    surf.save(fin)

    gmsh.initialize()
    # gmsh.option.setNumber("General.Verbosity", 0)

    # Create ONELAB parameters with remeshing options:
    gmsh.onelab.set("""[
          {
            "type":"number",
            "name":"Parameters/Angle for surface detection",
            "values":[180],
            "min":20,
            "max":120,
            "step":1
          },
          {
            "type":"number",
            "name":"Parameters/Create surfaces guaranteed to be parametrizable",
            "values":[1],
            "choices":[0, 1]
          }
        ]""")

    '''try:'''
    # Clear all models and merge an STL mesh that we would like to remesh
    gmsh.clear()
    gmsh.merge(fin)
    # We first classify the surfaces by splitting the original surface
    # along sharp geometrical features. This will create new discrete surfaces,
    # curves and points.
    # Angle between two triangles above which an edge is considered as sharp,
    # retrieved from the ONELAB database (see below):
    angle = gmsh.onelab.getNumber('Parameters/Angle for surface detection')[0]
    # For complex geometries, patches can be too complex, too elongated or too
    # large to be parametrized; setting the following option will force the
    # creation of patches that are amenable to reparametrization:
    forceParametrizablePatches = gmsh.onelab.getNumber(
        'Parameters/Create surfaces guaranteed to be parametrizable')[0]
    # For open surfaces include the boundary edges in the classification
    # process:
    if is_open:
        # Aorta, coronaries and pulmonary are open geometries
        includeBoundary = True
    else:
        # Calcium is a closed volume
        includeBoundary = False
    # Force curves to be split on given angle:
    curveAngle = 180
    gmsh.model.mesh.classifySurfaces(angle * math.pi / 180., includeBoundary,
                                     forceParametrizablePatches,
                                     curveAngle * math.pi / 180.)
    # Create a geometry for all the discrete curves and surfaces in the mesh, by
    # computing a parametrization for each one
    gmsh.model.mesh.createGeometry()
    '''except:
        # Clear all models and merge an STL mesh that we would like to remesh 
        gmsh.clear()
        gmsh.merge(filename)
        gmsh.model.mesh.createTopology()
        gmsh.model.mesh.createGeometry()'''
    if dim == 2:
        # Create triangular surface mesh
        size = str(size)
        f = gmsh.model.mesh.field.add("MathEval")
        gmsh.model.mesh.field.setString(f, "F", size)
        gmsh.model.mesh.field.setAsBackgroundMesh(f)
        # Define the physical group of the surfaces to create only the 2D mesh
        s = gmsh.model.getEntities(2)
        gmsh.model.add_physical_group(2, [e[1] for e in s], name="fluid")
        # Use as 2D-meshing algorithm the MeshAdapt
        gmsh.option.setNumber('Mesh.Algorithm', 1)
        gmsh.model.mesh.generate()
    elif dim == 3:
        # Create volumes from all the surfaces
        s = gmsh.model.getEntities(2)
        l = gmsh.model.geo.addSurfaceLoop([e[1] for e in s])
        gmsh.model.geo.addVolume([l])
        gmsh.model.geo.synchronize()
        # Create tetrahedral volume mesh
        size = str(size)
        f = gmsh.model.mesh.field.add("MathEval")
        gmsh.model.mesh.field.setString(f, "F", size)
        gmsh.model.mesh.field.setAsBackgroundMesh(f)
        # Define the physical group of the volumes to create only the 3D mesh
        v = gmsh.model.getEntities(3)
        gmsh.model.add_physical_group(3, [e[1] for e in v])
        # Use as 3D-meshing algorithm the Frontal algorithm
        # gmsh.option.setNumber('Mesh.Algorithm3D', 4)
        gmsh.model.mesh.generate()

    # Write the files
    gmsh.write(fout)
    gmsh.finalize()

    surf_remeshed = meshio.read(fout)
    pvmesh = pv.utilities.from_meshio(surf_remeshed)
    if max_aspect_ratio is not None:
        qual = pvmesh.compute_cell_quality(quality_measure='aspect_ratio')
        it = 0
        while np.max(qual['CellQuality']) > max_aspect_ratio and it < max_iter:
            it += 1

            fin = 'surf.stl'
            fout = 'surf_remeshed.stl'
            surf.save(fin)

            gmsh.initialize()
            gmsh.option.setNumber("General.Verbosity", 0)

            gmsh.onelab.set("""[
                     {
                       "type":"number",
                       "name":"Parameters/Angle for surface detection",
                       "values":[180],
                       "min":20,
                       "max":120,
                       "step":1
                     },
                     {
                       "type":"number",
                       "name":"Parameters/Create surfaces guaranteed to be parametrizable",
                       "values":[1],
                       "choices":[0, 1]
                     }
                   ]""")

            '''try:'''
            gmsh.clear()
            gmsh.merge(fin)
            angle = gmsh.onelab.getNumber('Parameters/Angle for surface detection')[0]
            forceParametrizablePatches = gmsh.onelab.getNumber(
                'Parameters/Create surfaces guaranteed to be parametrizable')[0]
            if is_open:
                includeBoundary = True
            else:
                includeBoundary = False
            curveAngle = 180
            gmsh.model.mesh.classifySurfaces(angle * math.pi / 180., includeBoundary,
                                             forceParametrizablePatches,
                                             curveAngle * math.pi / 180.)
            gmsh.model.mesh.createGeometry()
            '''except:
                # Clear all models and merge an STL mesh that we would like to remesh 
                gmsh.clear()
                gmsh.merge(filename)
                gmsh.model.mesh.createTopology()
                gmsh.model.mesh.createGeometry()'''
            if dim == 2:
                size = str(size)
                f = gmsh.model.mesh.field.add("MathEval")
                gmsh.model.mesh.field.setString(f, "F", size)
                gmsh.model.mesh.field.setAsBackgroundMesh(f)
                s = gmsh.model.getEntities(2)
                gmsh.model.add_physical_group(2, [e[1] for e in s])
                gmsh.option.setNumber('Mesh.Algorithm', 1)
                gmsh.model.mesh.generate()
            elif dim == 3:
                s = gmsh.model.getEntities(2)
                l = gmsh.model.geo.addSurfaceLoop([e[1] for e in s])
                gmsh.model.geo.addVolume([l])
                gmsh.model.geo.synchronize()
                size = str(size)
                f = gmsh.model.mesh.field.add("MathEval")
                gmsh.model.mesh.field.setString(f, "F", size)
                gmsh.model.mesh.field.setAsBackgroundMesh(f)
                v = gmsh.model.getEntities(3)
                gmsh.model.add_physical_group(3, [e[1] for e in v])
                # gmsh.option.setNumber('Mesh.Algorithm3D', 4)
                gmsh.model.mesh.generate()

            gmsh.write(fout)
            gmsh.finalize()

            surf_remeshed = meshio.read(fout)
            pvmesh = pv.utilities.from_meshio(surf_remeshed)
            qual = pvmesh.compute_cell_quality(quality_measure='aspect_ratio')

        if verbose: print('Max aspect ratio:', np.max(qual['CellQuality']))
    if show_plot:
        # Load the initial and final STL files
        mesh_initial = pv.read(fin)
        mesh_final = pv.read(fout)
        plotter = pv.Plotter(shape=(1, 2))
        # Add the initial mesh to the left subplot
        plotter.subplot(0, 0)
        plotter.add_mesh(mesh_initial, color=[0.4, 0.4, 0.4], show_edges=True, opacity=1, specular=1,
                         smooth_shading=True)
        plotter.add_text('Before remeshing', position='upper_edge', font_size=24)
        # Add the final mesh to the right subplot
        plotter.subplot(0, 1)
        plotter.add_mesh(mesh_final, color=[0.4, 0.4, 0.4], show_edges=True, opacity=1, specular=1, smooth_shading=True)
        plotter.add_text('After remeshing', position='upper_edge', font_size=24)
        # Link the camera positions and up vectors
        plotter.link_views()
        # Display the plot
        plotter.show()
    os.remove(fin)
    os.remove(fout)

    return pvmesh.extract_surface()



def gmsh_remesh_fsi(surf, size, lumen_bl=4, wall_bl=2, lumen_t=0.5, wall_t=1.5, lumen_el='tetra', wall_el='wedge'):
    fin = 'surf.stl'
    fout = 'surf_remeshed.msh'
    surf.save(fin)

    gmsh.initialize()
    gmsh.clear()
    gmsh.merge(fin)
    gmsh.model.mesh.classifySurfaces(angle=math.pi, curveAngle=math.pi, boundary=False, forReparametrization=True)
    gmsh.model.mesh.createGeometry()

    # make extrusions only return "top" surfaces and volumes, not lateral surfaces
    gmsh.option.setNumber('Geometry.ExtrudeReturnLateralEntities', 0)
    e1 = []
    e2 = []
    el_type = {'tetra': False, 'wedge': True}

    # lumen boundary layer
    if lumen_bl > 0:
        e2 = gmsh.model.geo.extrudeBoundaryLayer(gmsh.model.getEntities(2), numElements=[lumen_bl], heights=[-lumen_t], recombine=el_type[lumen_el])
    # wall boundary layer
    if wall_bl > 0:
        e1 = gmsh.model.geo.extrudeBoundaryLayer(gmsh.model.getEntities(2), numElements=[wall_bl], heights=[wall_t], recombine=el_type[wall_el], second=True)

    # get "top" surfaces created by extrusion
    top_ent = [s for s in e2 if s[0] == 2] if lumen_bl > 0 else gmsh.model.getEntities(2)
    top_surf = [s[1] for s in top_ent]
    # get boundary of top surfaces, i.e. boundaries of holes
    gmsh.model.geo.synchronize()
    bnd_ent = gmsh.model.getBoundary(top_ent)
    bnd_curv = [c[1] for c in bnd_ent]

    # cap the holes
    loops = gmsh.model.geo.addCurveLoops(bnd_curv)
    bnd_surf = []
    for l in loops:
        bnd_surf.append(gmsh.model.geo.addPlaneSurface([l]))
    # create the inner volume for the lumen
    vf = gmsh.model.geo.addVolume([gmsh.model.geo.addSurfaceLoop(top_surf + bnd_surf)])
    gmsh.model.geo.synchronize()

    # create physical groups for the different volumetric parts
    if e1:
        gmsh.model.addPhysicalGroup(3, [v[1] for v in e1 if v[0] == 3], name="solid")
    if e2:
        gmsh.model.addPhysicalGroup(3, [v[1] for v in e2 if v[0] == 3], name="fluid bl")
    gmsh.model.addPhysicalGroup(3, [vf], name="fluid")


    # Create tetrahedral volume mesh
    size = str(size)
    f = gmsh.model.mesh.field.add("MathEval")
    gmsh.model.mesh.field.setString(f, "F", size)
    gmsh.model.mesh.field.setAsBackgroundMesh(f)

    gmsh.option.setNumber('Mesh.Algorithm', 1)
    gmsh.model.mesh.generate()

    gmsh.write(fout)
    gmsh.finalize()

    surf_remeshed = meshio.read(fout)
    pvmesh = pv.utilities.from_meshio(surf_remeshed)
    pvmesh.save('prova.vtu')
    os.remove(fin)
    os.remove(fout)
    return pvmesh

def compute_tri_quad_global_face_nodes(mesh, global_id_key="GlobalNodeID"):
    """
    Outputs:
      tri_cell_idx:  indices of triangular cells in the mesh
      tri_gids:      [n_tri,3] GlobalNodeID of the verteces of triangualr faces
      quad_cell_idx: indices of quads cells in the mesh
      quad_gids:     [n_quad,4] GlobalNodeID of the verteces of triangualr faces
    """
    face_node_ids = mesh.regular_faces
    global_node_ids = np.asarray(mesh.point_data[global_id_key])

    face_lens = np.fromiter((len(f) for f in face_node_ids), dtype=np.int32)

    tri_cell_idx  = np.where(face_lens == 3)[0]
    quad_cell_idx = np.where(face_lens == 4)[0]

    tri_conn = np.asarray([face_node_ids[i] for i in tri_cell_idx], dtype=np.int64) if tri_cell_idx.size else np.empty((0,3), dtype=np.int64)
    quad_conn = np.asarray([face_node_ids[i] for i in quad_cell_idx], dtype=np.int64) if quad_cell_idx.size else np.empty((0,4), dtype=np.int64)

    # mappa local point ids -> GlobalNodeID
    tri_gids  = global_node_ids[tri_conn]  if tri_conn.size  else np.empty((0,3), dtype=global_node_ids.dtype)
    quad_gids = global_node_ids[quad_conn] if quad_conn.size else np.empty((0,4), dtype=global_node_ids.dtype)

    return tri_cell_idx, tri_gids, quad_cell_idx, quad_gids


def region_face_indices(region_node_ids, tri_cell_idx, tri_gids, quad_cell_idx, quad_gids, lumen_el):
    region_node_ids = np.asarray(region_node_ids)
    tri_hits = tri_cell_idx[
        np.all(np.isin(tri_gids, region_node_ids), axis=1)
    ] if tri_gids.size else np.empty((0,), dtype=np.int64)

    if lumen_el == "tetra" or quad_gids.size == 0:
        return tri_hits

    quad_hits = quad_cell_idx[
        np.all(np.isin(quad_gids, region_node_ids), axis=1)
    ]
    return np.concatenate([quad_hits, tri_hits])

def extract_patch(surfaces, region_id, face_id, scalar_name="RegionId"):
    patch = surfaces.threshold([region_id, region_id + 0.5], scalars=scalar_name).extract_surface()
    patch.cell_data["ModelFaceID"] = np.full(patch.n_cells, face_id, dtype=np.int32)
    patch.cell_data["Normals"] = patch.cell_normals
    return patch

def assign_model_face_ids(mesh, regions, lumen_el, default_id=6, global_id_key="GlobalNodeID"):
    tri_cell_idx, tri_gids, quad_cell_idx, quad_gids = compute_tri_quad_global_face_nodes(mesh, global_id_key)
    model_face_ids = np.full(mesh.n_cells, default_id, dtype=np.int32)
    # assign face id
    for face_id, node_ids in regions.items():
        idx = region_face_indices(node_ids, tri_cell_idx, tri_gids, quad_cell_idx, quad_gids, lumen_el)
        model_face_ids[idx] = int(face_id)

    mesh.cell_data["ModelFaceID"] = model_face_ids
    return model_face_ids

def find_sovraortic_vessels(ao_surf, sav_surf):
    dist_aorta = ao_surf.compute_implicit_distance(ao_surf)
    dist_supra = ao_surf.compute_implicit_distance(sav_surf)
    indices = np.where(dist_aorta['implicit_distance'] < dist_supra['implicit_distance'], 1, 2)
    ao_surf.point_data['region_index'] = indices
    return ao_surf

def mmg_remesh_adaptive_aorta(tot_surf, sec_surf, hausd=0.3, main_size=1, sec_size=0.4,  max_aspect_ratio=None, max_iter=3, verbose=False):
    tot_surf = find_sovraortic_vessels(tot_surf, sec_surf)
    size_map = np.where(tot_surf['region_index'] == 1, main_size, sec_size)

    sol_filename = "aorta.sol"
    with open(sol_filename, "w") as f:
        f.write("MeshVersionFormatted 2\n")
        f.write("\nDimension 3\n\n")
        f.write("SolAtVertices\n")
        f.write(f"{len(tot_surf.points)}\n")
        f.write("1 1\n\n")
        for size in size_map:
            f.write(f"{size}\n")
        f.write("\nEnd\n")

    tot_surf.clear_data()
    pv.save_meshio(f'aorta.mesh', tot_surf)
    subprocess.run([r"path to .. utils\mmg\bin\mmgs_O3.exe",
                f'aorta.mesh',
                '-hausd', str(hausd),
                '-sol', sol_filename,
                '-nr',
                '-nreg',
                '-xreg'
                '-optim',
                f'aorta_remeshed.mesh'], stdout=subprocess.DEVNULL)
    new_mesh = meshio.read(f'aorta_remeshed.mesh')
    pvmesh = pv.utilities.from_meshio(new_mesh)

    if max_aspect_ratio is not None:
        qual = pvmesh.compute_cell_quality(quality_measure='aspect_ratio')
        it = 0
        while np.max(qual['CellQuality']) > max_aspect_ratio and it < max_iter:
            it += 1

            pvmesh = find_sovraortic_vessels(pvmesh, sec_surf)
            size_map = np.where(pvmesh['region_index'] == 1, main_size, sec_size)

            sol_filename = "aorta.sol"
            with open(sol_filename, "w") as f:
                f.write("MeshVersionFormatted 2\n")
                f.write("\nDimension 3\n\n")
                f.write("SolAtVertices\n")
                f.write(f"{len(pvmesh.points)}\n")
                f.write("1 1\n\n")
                for size in size_map:
                    f.write(f"{size}\n")
                f.write("\nEnd\n")

            subprocess.run([r"path to .. mmg\bin\mmgs_O3.exe",
                            f'aorta_remeshed.mesh',
                            '-hausd', str(hausd),
                            '-sol', sol_filename,
                            '-nr',
                            '-nreg',
                            '-xreg'
                            '-optim',
                            f'aorta_remeshed.mesh'], stdout=subprocess.DEVNULL)

            new_mesh = meshio.read(f'aorta_remeshed.mesh')
            pvmesh = pv.utilities.from_meshio(new_mesh)
            qual = pvmesh.compute_cell_quality(quality_measure='aspect_ratio')

        if verbose: print('Max aspect ratio:', np.max(qual['CellQuality']))
    os.remove(f'aorta.mesh')
    os.remove(f'aorta_remeshed.mesh')
    os.remove(f'aorta_remeshed.sol')
    os.remove(f'aorta.sol')
    return pvmesh.extract_surface()