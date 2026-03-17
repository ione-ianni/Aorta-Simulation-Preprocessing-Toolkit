from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from vmtk import vmtkscripts
from vtk_utils import *
import mesh_utils as mut
import math


def extract_centerline(surface, sourceCoords, targetCoords, resampling=0.05, appendEndPoints=0, project2surface=False):
    cl_filter = vmtkscripts.vmtkCenterlines()
    cl_filter.Surface = surface
    cl_filter.SeedSelectorName = "pointlist"
    cl_filter.SourcePoints = sourceCoords
    cl_filter.TargetPoints = targetCoords
    #cl_filter.CapDisplacement = 1
    cl_filter.AppendEndPoints = appendEndPoints
    cl_filter.Resampling = 1
    cl_filter.ResamplingStepLength = resampling
    cl_filter.Execute()

    attr = vmtkscripts.vmtkCenterlineAttributes()
    attr.Centerlines = cl_filter.Centerlines
    attr.Execute()

    geo = vmtkscripts.vmtkCenterlineGeometry()
    geo.Centerlines = attr.Centerlines
    geo.LineSmoothing = 0
    geo.OutputSmoothingLines = 0
    geo.Execute()
    cl = geo.Centerlines

    if project2surface:
        projecter = vmtkscripts.vmtkSurfaceCenterlineProjection()
        projecter.Surface = surface
        projecter.Centerlines = cl
        projecter.Execute()
        return pv.wrap(cl), pv.wrap(projecter.Surface)
    else:
        return pv.wrap(cl)


def resample_centerline(cl, step=0.001):
    samp = vmtkscripts.vmtkCenterlineResampling()
    samp.Centerlines = cl
    samp.Length = step
    samp.Execute()
    return samp.Centerlines


def add_centerline_geometry(cl):
    attr = vmtkscripts.vmtkCenterlineAttributes()
    attr.Centerlines = cl
    attr.Execute()
    geo = vmtkscripts.vmtkCenterlineGeometry()
    geo.Centerlines = attr.Centerlines
    geo.LineSmoothing = 1
    geo.OutputSmoothingLines = 1
    geo.Execute()
    return geo.Centerlines


def smooth_centerline(cl, n_iter=50, factor=0.5):
    smoo = vmtkscripts.vmtkCenterlineSmoothing()
    smoo.Centerlines = cl
    smoo.NumberOfSmoothingIterations = n_iter
    smoo.SmoothingFactor = factor
    smoo.Execute()
    return smoo.Centerlines

def extract_PT_centerline(surface, sourceCoord, targetLeftCoord, targetRightCoord):
    newSourceCoord, newSourceId = find_closest_point_on_surface(sourceCoord, surface)
    newTargetLeftCoord, newTargetLeftId = find_closest_point_on_surface(targetLeftCoord, surface)
    newTargetRightCoord, newTargetRightId = find_closest_point_on_surface(targetRightCoord, surface)

    allTargetCoord = list(newTargetLeftCoord)[:] + list(newTargetRightCoord)[:]

    cl_filter = vmtkscripts.vmtkCenterlines()
    cl_filter.Surface = surface
    #cl_filter.SeedSelectorName = "idlist"
    #cl_filter.SourceIds = [newSourceId]
    #cl_filter.TargetIds = [newTargetId]
    cl_filter.SeedSelectorName = "pointlist"
    cl_filter.SourcePoints = list(newSourceCoord)
    cl_filter.TargetPoints = allTargetCoord
    #cl_filter.CapDisplacement = 1
    #cl_filter.AppendEndPoints = 1
    cl_filter.Resampling = 1
    cl_filter.ResamplingStepLength = 0.5
    cl_filter.Execute()

    attr = vmtkscripts.vmtkCenterlineAttributes()
    attr.Centerlines = cl_filter.Centerlines
    attr.Execute()

    geo = vmtkscripts.vmtkCenterlineGeometry()
    geo.Centerlines = attr.Centerlines
    geo.LineSmoothing = 1
    geo.OutputSmoothingLines = 1
    geo.Execute()
    return geo.Centerlines, cl_filter

def compute_distance2cl0(surface, parentCenterline):
    print('Computing distances to parent centerline.')
    distArr = vtk.vtkDoubleArray()
    distArr.SetNumberOfComponents(1)
    distArr.SetName('DistanceToParentCenterline')
    distArr.SetNumberOfTuples(surface.GetNumberOfPoints())
    distArr.FillComponent(0, 1)
    surface.GetPointData().AddArray(distArr)

    for i in range(surface.GetNumberOfPoints()):
        # print(surface.GetNumberOfPoints()-i)
        pt = surface.GetPoint(i)
        dist = np.empty(parentCenterline.GetNumberOfPoints())
        for j in range(parentCenterline.GetNumberOfPoints()):
            dist[j] = math.sqrt(vtk.vtkMath.Distance2BetweenPoints(pt, parentCenterline.GetPoint(j)))
        distArr.SetValue(i, np.min(dist))

    surface.GetPointData().AddArray(distArr)

    return surface


def find_cl_branch_targets(surface, parentCenterline, delta=0.1, arrayName='hks'):
    '''
    dist = vmtkscripts.vmtkDistanceToCenterlines()
    dist.Surface = surface
    dist.Centerlines = parentCenterline
    dist.UseRadiusInformation = 1
    dist.EvaluateTubeFunction = 1
    dist.EvaluateCenterlineRadius = 1
    dist.UseCombinedDistance = 1
    dist.Execute()
    surface = dist.Surface
    '''
    surface = compute_distance2cl0(surface, parentCenterline)
    np_arr = vtk_to_numpy(surface.GetPointData().GetArray(arrayName))
    norm_dist = numpy_to_vtk(np_arr / max(np_arr))
    normArrName = 'Normalized' + arrayName
    norm_dist.SetName(normArrName)
    surface.GetPointData().AddArray(norm_dist)

    coor_arr = np.empty((surface.GetNumberOfPoints(),3))
    for i in range(surface.GetNumberOfPoints()):
        coor_arr[i] = surface.GetPoint(i)
    coords = numpy_to_vtk(coor_arr)
    coords.SetName('Coordinates')
    surface.GetPointData().AddArray(coords)

    pt2cell = vtk.vtkPointDataToCellData()
    pt2cell.SetInputData(surface)
    pt2cell.PassPointDataOn()
    pt2cell.Update()
    surface = pt2cell.GetOutput()

    clcoord = np.empty((parentCenterline.GetNumberOfPoints(), 3))
    for i in range(parentCenterline.GetNumberOfPoints()):
        clcoord[i,:] = parentCenterline.GetPoint(i)
    #cl0Top = clcoord[np.argmax(clcoord[:,2]), :] - 50
    cl0Top = clcoord[np.argmin(clcoord[:, 1]), :] - 50

    plane = vtk.vtkPlane()
    plane.SetOrigin(cl0Top[0], cl0Top[1], cl0Top[2])
    plane.SetNormal(0, 0, 1)
    clipFilter = vtk.vtkClipPolyData()
    clipFilter.SetInputData(surface)
    clipFilter.SetClipFunction(plane)
    clipFilter.Update()
    pd = clipFilter.GetOutput()
    #pd2 = pv.PolyData(pd)
    #pd2.plot()

    #pd = surface
    _, branchTargetIds = mut.recursive_find_endpoints(pd, delta=delta, dist_thr=1.0, normArrName=normArrName)
    #if len(branchTargetIds) > 3: branchTargetIds.pop(-1)
    branchTargetCoord = []
    for i in range(len(branchTargetIds)):
        branchTargetCoord.append(pd.GetPoint(branchTargetIds[i]))
        _, branchTargetIds[i] = find_closest_point_on_surface(pd.GetPoint(branchTargetIds[i]), surface)
    return branchTargetCoord, branchTargetIds, surface

def extract_branch_centerlines(surface, parentCenterline, branchTargetCoord):
    target_pt_coord, target_pt_id = find_closest_point_on_surface(parentCenterline.GetPoint(0), surface)
    source_pt_coord, source_pt_id = find_closest_point_on_surface(parentCenterline.GetPoint(parentCenterline.GetNumberOfPoints() - 1), surface)
    allTargetCoord = []
    for k in range(3):
        allTargetCoord.append(list(target_pt_coord)[k])
    for i in range(3):
        for j in range(3):
            allTargetCoord.append((branchTargetCoord[i][j]))


    cl_filter = vmtkscripts.vmtkCenterlines()
    cl_filter.Surface = surface
    cl_filter.SeedSelectorName = "pointlist"
    cl_filter.SourcePoints = list(source_pt_coord)
    cl_filter.TargetPoints = allTargetCoord
    cl_filter.Execute()

    attr = vmtkscripts.vmtkCenterlineAttributes()
    attr.Centerlines = cl_filter.Centerlines
    attr.Execute()

    geo = vmtkscripts.vmtkCenterlineGeometry()
    geo.Centerlines = attr.Centerlines
    geo.LineSmoothing = 0
    geo.OutputSmoothingLines = 0
    geo.Execute()
    return geo.Centerlines



def sort_branch_targets(branchTargetCoord, branchTargetIds, cl_0):
    newTargets = []
    sortedBranchTargetCoord = []
    sortedBranchTargetIds = []
    cl0_abscissas = cl_0.GetPointData().GetArray('Abscissas')
    dist = np.empty(cl_0.GetNumberOfPoints())
    for j in range(len(branchTargetIds)):
        for i in range(cl_0.GetNumberOfPoints()):
            d = math.sqrt(vtk.vtkMath.Distance2BetweenPoints(branchTargetCoord[j], cl_0.GetPoint(i)))
            dist[i] = d
        closest_point_on_cl0 = np.argmin(dist)
        br_end = dict()
        br_end['coord'] = branchTargetCoord[j]
        br_end['id'] = branchTargetIds[j]
        br_end['cl_0_abscissa'] = cl0_abscissas.GetValue(closest_point_on_cl0)
        newTargets.append(br_end)

    sortedTargets = sorted(newTargets, key=lambda k: k['cl_0_abscissa'])

    for i in range(len(sortedTargets)):
        sortedBranchTargetCoord.append(sortedTargets[i]['coord'])
        sortedBranchTargetIds.append(sortedTargets[i]['id'])

    return sortedBranchTargetCoord, sortedBranchTargetIds


def divide_surface(surface, centerlines):
    extractor = vmtkscripts.vmtkBranchExtractor()
    extractor.Surface = surface
    extractor.Centerlines = centerlines
    extractor.Execute()
    clipper = vmtkscripts.vmtkBranchClipper()
    clipper.Surface = surface
    clipper.Centerlines = extractor.Centerlines
    clipper.Execute()
    surface = clipper.Surface

    cell2pt = vtk.vtkCellDataToPointData()
    cell2pt.SetInputData(clipper.Centerlines)
    cell2pt.PassCellDataOn()
    cell2pt.Update()
    cl = cell2pt.GetOutput()
    centerlines = cl

    return surface, centerlines


def separate_centerlines(cl):
    thr = vtk.vtkThreshold()
    thr.SetInputData(cl)
    thr.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS_THEN_CELLS, 'CenterlineIds')
    thr.ThresholdBetween(0.0, 0.5)
    thr.Update()
    geo = vtk.vtkGeometryFilter()
    geo.SetInputData(thr.GetOutput())
    geo.Update()
    cl_0 = geo.GetOutput()

    thr.ThresholdBetween(0.5, 1.5)
    thr.Update()
    geo = vtk.vtkGeometryFilter()
    geo.SetInputData(thr.GetOutput())
    geo.Update()
    cl_1 = geo.GetOutput()

    thr.ThresholdBetween(1.5, 2.5)
    thr.Update()
    geo = vtk.vtkGeometryFilter()
    geo.SetInputData(thr.GetOutput())
    geo.Update()
    cl_2 = geo.GetOutput()

    thr.ThresholdBetween(2.5, 3.5)
    thr.Update()
    geo = vtk.vtkGeometryFilter()
    geo.SetInputData(thr.GetOutput())
    geo.Update()
    cl_3 = geo.GetOutput()

    return cl_0, cl_1, cl_2, cl_3


def modify_branch_centerline(tube0, cl, cl0):
    # trovo punto in cui tagliare la centerline, tolgo inizio se no trova sempre l'inizio, e taglio
    dist = np.empty(np.max([cl.GetNumberOfPoints() - 50, 1]))
    for i in range(cl.GetNumberOfPoints() - 50):
        dist[i], _ = mut.calculate_minimum_distance_to_polydata(cl.GetPoint(i + 50), tube0)
    pt_id = np.argmin(dist) + 50
    pt_coord_clbif = cl.GetPoint(pt_id)

    plane = vtk.vtkPlane()
    plane.SetOrigin(cl.GetPoint(pt_id))
    plane.SetNormal(cl.GetPointData().GetArray('FrenetTangent').GetTuple(pt_id))

    cut = vtk.vtkClipPolyData()
    cut.SetInputData(cl)
    cut.SetClipFunction(plane)
    cut.Update()
    cl_new = cut.GetOutput()

    # trovo punto a cui ricucire la centerline
    _, pt_id = mut.calculate_minimum_distance_to_polydata(cl.GetPoint(pt_id), cl0)
    pt_coord_cl0 = cl0.GetPoint(pt_id)

    # interpolo punti con spline
    pts = vtk.vtkPoints()
    pts.InsertNextPoint(cl0.GetPoint(pt_id))
    for k in range(cl_new.GetNumberOfPoints()):
        pts.InsertNextPoint(cl_new.GetPoint(k))

    spline = vtk.vtkParametricSpline()
    spline.SetPoints(pts)

    functionSource = vtk.vtkParametricFunctionSource()
    functionSource.SetParametricFunction(spline)
    functionSource.SetUResolution(cl_new.GetNumberOfPoints() + 1)
    functionSource.Update()
    cl_spline = functionSource.GetOutput()

    cl_spline = resample_centerline(cl_spline, step=0.1)

    array = vtk.vtkDoubleArray()
    array.SetName('MaximumInscribedSphereRadius')
    array.SetNumberOfComponents(0)
    array.SetNumberOfTuples(cl_spline.GetNumberOfPoints())
    for j in range(cl_spline.GetNumberOfPoints()):
        _, id = find_closest_point_on_surface(cl_spline.GetPoint(j), cl)
        radius = cl.GetPointData().GetArray('MaximumInscribedSphereRadius').GetTuple(id)
        array.SetTuple(j, radius)

    cl_spline.GetPointData().AddArray(array)

    cl_spline = add_centerline_geometry(cl_spline)

    pt_ref = dict()
    pt_ref['Ostium'] = pt_coord_clbif
    pt_ref['Ostium_proj'] = pt_coord_cl0

    return cl_spline, pt_ref


def get_top_cl_group(cl, pt_ref):
    _, pt_id = find_closest_point_on_surface(pt_ref['Ostium'], cl)

    plane = vtk.vtkPlane()
    plane.SetOrigin(cl.GetPoint(pt_id))
    plane.SetNormal(cl.GetPointData().GetArray('FrenetTangent').GetTuple(pt_id))

    cut = vtk.vtkClipPolyData()
    cut.SetInputData(cl)
    cut.SetClipFunction(plane)
    cut.Update()
    cl_new = cut.GetOutput()

    return cl_new