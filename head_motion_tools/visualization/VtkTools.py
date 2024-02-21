"""
Basic Tools to make our data structures work with Vtk (meshes and pointclouds)
"""


import vtk
import matplotlib
import numpy as np

from head_motion_tools import point_cloud_io, preprocessing


class VtkArrow:

    def __init__(self, startPoint, endPoint):
        assert(len(startPoint) == 3)
        assert(len(endPoint) == 3)


        colors = vtk.vtkNamedColors()

        # Set the background color.
        #colors.SetColor("BkgColor", [26, 51, 77, 255])

        # Create an arrow.
        arrowSource = vtk.vtkArrowSource()

        # Compute a basis
        normalizedX = [0] * 3
        normalizedY = [0] * 3
        normalizedZ = [0] * 3

        # The X axis is a vector from start to end
        vtk.vtkMath.Subtract(endPoint, startPoint, normalizedX)
        length = vtk.vtkMath.Norm(normalizedX)
        vtk.vtkMath.Normalize(normalizedX)

        # The Z axis is an arbitrary vector cross X
        arbitrary = [1, 1, 1]
        vtk.vtkMath.Cross(normalizedX, arbitrary, normalizedZ)
        vtk.vtkMath.Normalize(normalizedZ)

        # The Y axis is Z cross X
        vtk.vtkMath.Cross(normalizedZ, normalizedX, normalizedY)



        matrix = vtk.vtkMatrix4x4()

        # Create the direction cosine matrix
        matrix.Identity()

        for i in range(0, 3):
            matrix.SetElement(i, 0, normalizedX[i])
            matrix.SetElement(i, 1, normalizedY[i])
            matrix.SetElement(i, 2, normalizedZ[i])

        # Apply the transforms
        transform = vtk.vtkTransform()
        transform.Translate(startPoint)
        transform.Concatenate(matrix)
        transform.Scale(length, length, length)

        # Transform the polydata
        transformPD = vtk.vtkTransformPolyDataFilter()
        transformPD.SetTransform(transform)
        transformPD.SetInputConnection(arrowSource.GetOutputPort())

        # Create a mapper and actor for the arrow
        mapper = vtk.vtkPolyDataMapper()
        actor = vtk.vtkActor()
        # if USER_MATRIX:
        #     mapper.SetInputConnection(arrowSource.GetOutputPort())
        #     actor.SetUserMatrix(transform.GetMatrix())
        # else:
        mapper.SetInputConnection(transformPD.GetOutputPort())
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(colors.GetColor3d("Cyan"))

        # Create spheres for start and end point
        sphereStartSource = vtk.vtkSphereSource()
        sphereStartSource.SetCenter(startPoint)
        sphereStartSource.SetRadius(0.8)
        sphereStartMapper = vtk.vtkPolyDataMapper()
        sphereStartMapper.SetInputConnection(sphereStartSource.GetOutputPort())
        sphereStart = vtk.vtkActor()
        sphereStart.SetMapper(sphereStartMapper)
        sphereStart.GetProperty().SetColor(colors.GetColor3d("Yellow"))

        sphereEndSource = vtk.vtkSphereSource()
        sphereEndSource.SetCenter(endPoint)
        sphereEndSource.SetRadius(0.8)
        sphereEndMapper = vtk.vtkPolyDataMapper()
        sphereEndMapper.SetInputConnection(sphereEndSource.GetOutputPort())
        sphereEnd = vtk.vtkActor()
        sphereEnd.SetMapper(sphereEndMapper)
        sphereEnd.GetProperty().SetColor(colors.GetColor3d("Yellow"))

        self.vtkActor = actor
        self.vtkSphereStart = sphereStart
        self.vtkSphereEnd = sphereEnd 
        

class VtkPointCloud:
    """
    Representation of point cloud in Vtk
    """

    def __init__(self, pc, pt_min=None, pt_max=None, colors=None, colors_min=None, colors_max=None, color_map='gray', invert=False):
        self.vtkActor = vtk.vtkActor()
        self.clearPoints()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.vtkPolyData)

        if len(pc.shape) == 3:
            pc = point_cloud_io.to2dArray(pc)

        # generate dictionary of color names and rgb values
        rgb_colors = {}
        for name, hex in matplotlib.colors.cnames.items():
            rgb_colors[name] = matplotlib.colors.to_rgb(hex)

        if isinstance(colors, str) and colors in list(rgb_colors.keys()):
            mapper.SetScalarVisibility(0)
            self.vtkActor.GetProperty().SetColor(*rgb_colors[colors])

        else:  # process color mapping
            mapper.SetColorModeToDefault()

            if colors is None:  # set coloring to scale with Z axis
                if pt_min is None or pt_max is None:
                    pt_min, pt_max, _ = preprocessing.get_boundaries(pc)
                zMin = pt_min[2]
                zMax = pt_max[2]
                mapper.SetScalarRange(zMin, zMax)
            else:
                if colors_min is None:
                    zMin = np.min(colors)
                else:
                    zMin = colors_min
                if colors_max is None:
                    zMax = np.max(colors)
                else:
                    zMax = colors_max

                # Create a greyscale lookup table
                table = vtk.vtkLookupTable()
                table.SetRange(zMin, zMax)  # image intensity range
                #if invert:
                #    table.SetValueRange(1, 0)  # from white to black
                #else:
                #    table.SetValueRange(0, 1)  # from black to white

                if color_map == 'gray':
                    table.SetValueRange(0, 1)  # from white to black
                    table.SetSaturationRange(0.0, 0.0)  # no color saturation
                    table.SetRampToLinear()
                else:
                    #table.SetSaturationRange(1, 1)
                    table.SetHueRange(0, 0.26) # red to green
                    # Raingbow red to blue 0, 0.667
                    #Rainbow - blue to red: 0.667,0.0
                    table.SetRampToLinear()
                table.SetNumberOfTableValues(402)
                table.Build()
                mapper.SetLookupTable(table)
            mapper.SetScalarRange(zMin, zMax)
            mapper.SetScalarVisibility(1)

        self.vtkActor.SetMapper(mapper)


        if colors is None or isinstance(colors, str):
            self.addPoints(pc, self.vtkPoints, self.vtkDepth, self.vtkCells)
        else:
            assert(colors.shape[0] == pc.shape[0])
            self.addPointsColors(pc, colors, self.vtkPoints, self.vtkDepth, self.vtkCells)

        self.vtkDepth.Modified()
        self.vtkCells.Modified()
        self.vtkPoints.Modified()


    def addPoints(self, pc, vtkPoints, vtkDepth, vtkCells):
        for i in range(pc.shape[0]): #for pt in list(pc):
            vtkDepth.InsertNextValue(pc[i,2])
            vtkCells.InsertNextCell(1)
            vtkCells.InsertCellPoint(vtkPoints.InsertNextPoint(pc[i]))

    def addPointsColors(self, pc, colors, vtkPoints, vtkDepth, vtkCells):
        for i in range(pc.shape[0]): #enumerate(list(pc)):
            vtkDepth.InsertNextValue(colors[i])
            vtkCells.InsertNextCell(1)
            vtkCells.InsertCellPoint(vtkPoints.InsertNextPoint(pc[i]))

    def clearPoints(self):
        self.vtkPoints = vtk.vtkPoints()
        self.vtkCells = vtk.vtkCellArray()
        self.vtkDepth = vtk.vtkDoubleArray()
        self.vtkDepth.SetName('DepthArray')
        self.vtkPolyData = vtk.vtkPolyData()
        self.vtkPolyData.SetPoints(self.vtkPoints)
        self.vtkPolyData.SetVerts(self.vtkCells)
        self.vtkPolyData.GetPointData().SetScalars(self.vtkDepth)
        self.vtkPolyData.GetPointData().SetActiveScalars('DepthArray')


def mkVtkIdList(it):
    """
    Makes a vtkIdList from a Python iterable.

    it       A python iterable.
    return   A vtkIdList
    """
    vil = vtk.vtkIdList()
    for i in it:
        vil.InsertNextId(int(i))
    return vil


def createPoly(vertices,faces=None, colors=None):
    """
    Create vtk poly
    vertices    points
    faces       mesh faces if applicable

    return      filled vtkPolyData object
    """
    poly = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    scalars = vtk.vtkFloatArray()

    if colors is not None and len(colors) != len(vertices):
        raise ValueError('given intesities dont match numbers vertices')

    for i, xi in enumerate(vertices):
        points.InsertPoint(i, xi)
    if not faces is None:
        polys = vtk.vtkCellArray()
        for face in faces:
            polys.InsertNextCell(mkVtkIdList(face))

    if colors is not None:
        for i, c in enumerate(colors):
            scalars.InsertTuple1(i, c)
    else:
        for i, _ in enumerate(vertices):
            scalars.InsertTuple1(i, i)

    poly.SetPoints(points)
    if not faces is None:
        poly.SetPolys(polys)
    poly.GetPointData().SetScalars(scalars)
    return poly


def displayPoly(poly,size=(1000, 1000)):
    """
    Displays vtkPolyData object
    """
    colors = vtk.vtkNamedColors()
    polyMapper = vtk.vtkPolyDataMapper()
    polyMapper.SetInputData(poly)
    polyMapper.SetScalarRange(poly.GetScalarRange())
    polyMapper.SetColorModeToDefault()
    polyMapper.SetScalarVisibility(1)
    polyActor = vtk.vtkActor()
    polyActor.SetMapper(polyMapper)

    #camera = vtk.vtkCamera()
    #camera.SetPosition(1, 1, 1)
    #camera.SetFocalPoint(0, 0, 0)

    renderer = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(renderer)

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    renderer.AddActor(polyActor)
    #renderer.SetActiveCamera(camera)
    #renderer.ResetCamera()
    #renderer.SetBackVtkPointCloudground(colors.GetColor3d("Cornsilk"))
    renderer.SetBackground(1.0, 0.9688, 0.8594)

    renWin.SetSize(size)

    renWin.Render()
    iren.Start()


def displayMeshVtk(v,t=None):
    """
    displays mesh with vtk
    v    vertices
    t    face triangles if applicable

    NOTE: should work for arbitrary meshed not only triangle
    """
    mesh = createPoly(v,t)
    displayPoly(mesh)

