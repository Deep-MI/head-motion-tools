"""
Visualizer for Point Cloud seqeunces using Vtk library
"""

from argparse import Namespace
from gc import callbacks
import itertools
from typing import List

import vtk
import numpy as np

from head_motion_tools.visualization import VtkTools
from head_motion_tools import point_cloud_io, metadata_io



class PointCloudSequence:
    
    """
    calss for sequences of pointclouds
    enables displaying and saving of sequences and single pointclouds

    timestep     miliseconds
    name         title of sequence (for displaying and saving)
    no_pcs       number of expected pointclouds in sequence
    colors_min   minimum value on colormap
    colors_max   maximum value on colormap

    """
    def __init__(self, timestep=1000, name='Pointcloud Sequence', no_pcs=1, colors_min=None, colors_max=None):
        self.timestep = int(timestep)
        self.no_pcs = no_pcs
        self.name = name
        self.colors_max = colors_max
        self.colors_min = colors_min
        self.clear()
        
        
    def clear(self):
        self.sequenceList = []
        if self.no_pcs > 1:
            self.standard_cmap = []
            for i in range(self.no_pcs):
                self.sequenceList.append([])
                self.standard_cmap.append('gray')
                
        else:
            self.standard_cmap = 'gray'
    
    def showSingle(self, i):
        showPointCloud(self.sequenceList[i])
        
    def show(self, title='Pointcloud Sequence'):
        if self.no_pcs > 1:
            displayOverlaidPointCloudSequence(self.sequenceList, title="Pointcloud Sequence", ms=self.timestep)
        else:
            displayPointCloudSequence(self.sequenceList, title="Pointcloud Sequence", ms=self.timestep)
        
    def save(self, filename):
        if self.no_pcs > 1:
            print('calling Overlaid Pc Show')
            displayOverlaidPointCloudSequence(self.sequenceList, title="Pointcloud Sequence", ms=self.timestep, record=True, filename=filename)
        else:
            displayPointCloudSequence(self.sequenceList, title="Pointcloud Sequence", ms=self.timestep, record=True, filename=filename)
    
    def addPointCloud(self, pc, colors=None, cmap='gray'):
        if self.no_pcs != 1:
            raise ValueError('no_pointclouds set to >1, this disables addPointCloud')
        self.sequenceList.append(VtkTools.VtkPointCloud(pc, colors=colors, colors_min=self.colors_min, colors_max=self.colors_max, color_map=cmap))
        
    def addOverlaidPointClouds(self, pc_list, color_list, cmap=None):
        if cmap == None:
            cmap = self.standard_cmap
        
        if self.no_pcs == 1:
            pc = np.concatenate(pc_list, axis=0)
            color = np.array(list(itertools.chain.from_iterable(color_list)))
            self.sequenceList.append(VtkTools.VtkPointCloud(pc, colors=color, color_map=cmap, colors_min=self.colors_min, colors_max=self.colors_max))
        else:
            assert len(pc_list) == len(self.sequenceList), 'passed number of pointclouds doesnt match configuration'
            assert isinstance(cmap, list), 'pass list of color map for overlaid pointclouds'
            for i in range(len(pc_list)):
                self.sequenceList[i].append(VtkTools.VtkPointCloud(pc_list[i], colors=color_list[i], color_map=cmap[i], colors_min=self.colors_min, colors_max=self.colors_max))
            
    def getData(self):
        return self.sequenceList


class AddPointCloudTimerCallbackViewport():
    """
    Callback to update the displayed pointcloud in regular intervalls
    This one can render to multiple renderers
    used in displayPointCloudSequencesViewport
    """

    def __init__(self, renderers, pcs, record=False, moviewriter=None, imageFilter=None, recording_iteration=1):
        self.iterations = 1
        self.renderers = renderers
        self.prev = 0
        self.record = record
        self.moviewriter = moviewriter
        self.imageFilter = imageFilter
        self.movie_finished = False
        self.loop_number = 0
        self.RECORDING_ITERATION = recording_iteration
        self.frame_number = 0
        
        self.pcs = []

        for pcs_in_viewport in pcs:
            self.pcs.append(np.array(pcs_in_viewport))

    
    def hide_actors(self):
        for i,r in enumerate(self.renderers):
            if self.pcs[i].ndim == 2:
                for j in range(len(self.pcs[i][self.prev])):
                    r.RemoveActor(self.pcs[i][self.prev,j].vtkActor)
            else:
                r.RemoveActor(self.pcs[i][self.prev].vtkActor)


    def next(self, iren, event):
        # ----- used to set initial camera position -----
        # make camera global
        # print('position:    ',camera.GetPosition())
        # print('focal Point: ',camera.GetFocalPoint())
        # print('roll:        ',camera.GetRoll())
        # -----------------------------------------------
        selected_pc = (self.iterations % len(self.pcs[0])) -1

        # singal that the loop restarts
        if selected_pc == -1:
            self.hide_actors()
                

            if self.record and not self.movie_finished and self.loop_number == self.RECORDING_ITERATION:
                print('finishing')

                # Finish movie
                self.movie_finished = True

            self.loop_number += 1

        else:
            # TODO: could make this into numpy arrays for easier indexing and shape determination
            for i,r in enumerate(self.renderers):
                if self.pcs[i].ndim == 2:
                    for j in range(len(self.pcs[i][self.prev])):
                        r.RemoveActor(self.pcs[i][self.prev,j].vtkActor)
                        r.AddActor(self.pcs[i][selected_pc,j].vtkActor)
                else:
                    r.RemoveActor(self.pcs[i][self.prev].vtkActor)
                    r.AddActor(self.pcs[i][selected_pc].vtkActor)

            self.prev = selected_pc
            

        iren.GetRenderWindow().Render()

        if self.record and not self.movie_finished and self.loop_number == self.RECORDING_ITERATION:
            # Export a single frame
            self.frame_number += 1
            print('writing frame', self.frame_number, end='\r')
            self.imageFilter.Modified()
            self.moviewriter.Write()

        self.iterations += 1

    def previous(self, iren: vtk.vtkRenderWindowInteractor, event):
        if self.iterations <= 0:
            return

        selected_pc = (self.iterations % len(self.pcs[0])) -1

        if selected_pc == -1:
            self.hide_actors()
        else:
            for i,r in enumerate(self.renderers):
                if self.pcs[i].ndim == 2:
                    for j in range(len(self.pcs[i][self.prev])):
                        r.RemoveActor(self.pcs[i][self.prev,j].vtkActor)
                        r.AddActor(self.pcs[i][selected_pc,j].vtkActor)
                else:
                    r.RemoveActor(self.pcs[i][self.prev].vtkActor)
                    r.AddActor(self.pcs[i][selected_pc].vtkActor)

            self.prev = selected_pc

        #self.update_frame_displays(selected_pc)
        iren.GetRenderWindow().Render()

        self.iterations -= 1

            

            

        


class AddMeshCallbackViewport():
    """
    Callback to update the displayed pointcloud in regular intervalls, or on key press
    This one can render to multiple renderers, and handle multiple actors per viewport
    """

    def __init__(self, renderers: List[vtk.vtkRenderer], meshes: List[List[Namespace]], record=False, moviewriter=None, imageFilter=None, recording_iteration=1, frame_updates=[]):
        self.iterations = 1
        self.renderers = renderers
        self.record = record
        self.moviewriter = moviewriter
        self.imageFilter = imageFilter
        self.frame_updates = frame_updates
        self.movie_finished = False
        self.loop_number = 0
        self.RECORDING_ITERATION = recording_iteration
        self.frame_number = 0

        self.mov_mesh = meshes
        self.blank_frame = False

    
    def update_frame_displays(self, i):
        for update_func in self.frame_updates:
            update_func(i)

    def hide_actors(self):
        if not self.blank_frame:
            for r in self.renderers:
                for r, to_render in zip(self.renderers, self.mov_mesh):
                    for m_mesh in to_render:
                        r.RemoveActor(m_mesh.vtkActor)
        self.blank_frame = True


    def show_actors(self):
        if self.blank_frame:
            for r in self.renderers:
                for r, to_render in zip(self.renderers, self.mov_mesh):
                    for m_mesh in to_render:
                        r.AddActor(m_mesh.vtkActor)
            self.blank_frame = False

    def next(self, iren: vtk.vtkRenderWindowInteractor, event):
        # ----- used to set initial camera position -----
        # make camera global
        # camera = self.renderers[0].GetActiveCamera()
        # print('position:    ',camera.GetPosition())
        # print('focal Point: ',camera.GetFocalPoint())
        # print('roll:        ',camera.GetRoll())
        # -----------------------------------------------
        self.iterations += 1
        selected_trans = (self.iterations % self.mov_mesh[0][0].no_transforms) -1
        #print(selected_trans)

        # singal that the loop restarts
        if selected_trans == -1:
            self.hide_actors()

            if self.record and not self.movie_finished and self.loop_number == self.RECORDING_ITERATION:
                print('finishing')
                self.movie_finished = True

            self.loop_number += 1
        else:
            self.show_actors()
            for to_render in self.mov_mesh:
                for m_mesh in to_render:
                    m_mesh.callback_transform(selected_trans)

        self.update_frame_displays(selected_trans)
        iren.GetRenderWindow().Render()

        if self.record and not self.movie_finished and self.loop_number == self.RECORDING_ITERATION:
            # Export a single frame
            self.frame_number += 1
            print('writing frame', self.frame_number, end='\r')
            self.imageFilter.Modified()
            self.moviewriter.Write()

    def previous(self, iren: vtk.vtkRenderWindowInteractor, event):
        if self.iterations <= 0:
            print('reached start')
            return

        self.iterations -= 1
        selected_trans = (self.iterations % self.mov_mesh[0][0].no_transforms) -1
        #print(selected_trans)
        if selected_trans == -1:
            self.hide_actors()
        else:
            self.show_actors()
            for to_render in self.mov_mesh:
                for m_mesh in to_render:
                    m_mesh.callback_transform(selected_trans)

        self.update_frame_displays(selected_trans)
        iren.GetRenderWindow().Render()

        
def updateErrorFactory(txt_actor, measure_list):
    def update_error(frame):
        txt_actor.SetInput('RMS Error:' + str(np.round(measure_list[frame],2)))
    return update_error

def displayMeshesViewport(meshes, descriptions, title='Pointcloud Comparison', fps=30, window_size=(1000,1000), record=False, filename="movie.ogv", error_measures=None):


    camera = vtk.vtkCamera()
    setCameraPosition(camera, position='RAS')
    

    # Render Window
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.SetSize(*window_size)
    renderWindow.SetWindowName(title)

    # Interactor
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)
    renderWindowInteractor.Initialize()

    # Define viewport ranges
    xmins, xmaxs, ymins, ymaxs = getViewportLayouts(len(meshes))

    frame_updates = []

    renderers = []
    for i in range(len(meshes)):
        renderer = vtk.vtkRenderer()
        renderer.SetBackground(.2, .3, .4) # blue
        renderer.SetActiveCamera(camera)
        renderWindow.AddRenderer(renderer)
        renderer.SetViewport(xmins[i], ymins[i], xmaxs[i], ymaxs[i])

        txt = vtk.vtkTextActor()
        txt.SetInput(descriptions[i])
        renderer.AddActor(txt)

        if error_measures is not None:
            error_txt = vtk.vtkTextActor()
            error_txt.SetInput('RMS Error:' + str(np.round(error_measures[i][0],2)))
            error_txt.SetPosition(20, 50)
            #error_txt.GetTextProperty().SetJustificationToRight()

            renderer.AddActor(error_txt)

            # def update_error(frame):
            #     error_txt.SetInput('RMS Error:' + str(error_measures[i][frame]))
    
            frame_updates.append(updateErrorFactory(error_txt , error_measures[i]))

        for mesh in meshes[i]:
            renderer.AddActor(mesh.vtkActor)

        renderers.append(renderer)

    frame_counter = vtk.vtkTextActor()
    frame_counter.SetDisplayPosition(20, 30)
    frame_counter.SetInput('Frame number: 1')
    renderers[0].AddActor(frame_counter)
    frame_updates.append(lambda frame: frame_counter.SetInput('Frame number: ' + str(frame+1)))
    
    # make video from output
    if record:
        imageFilter, moviewriter = setupRecording(filename, renderWindow ,1/fps/1000)
    else:
        imageFilter = None
        moviewriter = None

    # Initialize a timer for the animation
    addPointCloudTimerCallback = AddMeshCallbackViewport(renderers, meshes, record=record, moviewriter=moviewriter, imageFilter=imageFilter, frame_updates=frame_updates)

    renderWindow.Render()
    style = RecordingClickInteractorStyle(renderWindowInteractor, renderWindow, addPointCloudTimerCallback, fps=fps)
    renderWindowInteractor.SetInteractorStyle(style)
    renderWindowInteractor.Start()

    if record:
        moviewriter.End()
        print('wrote',filename)


def displayPointCloudSequencesViewport(pcs, descriptions, title='Pointcloud Comparison', ms=100, window_size=(1000,1000), record=False, filename="movie.ogv"):
    """
    Display multiple point cloud sequnces in a vtk rendered window
    pcs             list of lists of pointclouds as 2d numpy arrays
    descriptions    list of captions for each sequence
    title           window title
    ms              number of ms each frame will be showed
    window_size     size of the display window in px

    no return
    """
    #colors = vtk.vtkNamedColors()

    # Camera
    # global camera
    camera =vtk.vtkCamera()

    # full view 
    # camera.SetPosition(195.27342323718858, 189.34504995328476, -146.12475972226036)
    # camera.SetFocalPoint(9.68813680451325, -0.440555007385222, 201.8577793162133)
    # camera.SetRoll(170.1225630033345)

    # close up
    setCameraPosition(camera)


    # Render Window
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.SetSize(*window_size)
    renderWindow.SetWindowName(title)

    # Interactor
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)
    renderWindowInteractor.Initialize()

    # Define viewport ranges
    xmins, xmaxs, ymins, ymaxs = getViewportLayouts(len(pcs))

    renderers = []
    for i in range(len(pcs)):
        renderer = vtk.vtkRenderer()
        renderer.SetBackground(.2, .3, .4) # blue
        #renderer.SetBackground(1.0, 0.9688, 0.8594) # beige
        #renderer.ResetCamera()
        renderer.SetActiveCamera(camera)
        renderWindow.AddRenderer(renderer)
        renderer.SetViewport(xmins[i], ymins[i], xmaxs[i], ymaxs[i])

        txt = vtk.vtkTextActor()
        txt.SetInput(descriptions[i])
        #txt.SetTextScaleModeToViewport()
        #txt.SetDisplayPosition(10, 30)
        #txtprop = txt.GetTextProperty()
        #txtprop.SetFontFamilyToArial()
        #txtprop.BoldOn()
        #txtprop.SetFontSize(36)
        #txtprop.ShadowOn()
        #txtprop.SetShadowOffset(4, 4)
        #txtprop.SetColor(colors.GetColor3d("Cornsilk"))

        renderer.AddActor(txt)



        renderers.append(renderer)

    # make video from output
    if record:
        imageFilter, moviewriter = setupRecording(filename, renderWindow ,ms)
    else:
        imageFilter = None
        moviewriter = None


    # Initialize a timer for the animation
    addPointCloudTimerCallback = AddPointCloudTimerCallbackViewport(renderers, pcs, record, moviewriter, imageFilter)
    if ms is None:
        renderWindow.Render()
        style = RecordingClickInteractorStyle(renderWindowInteractor, renderWindow, addPointCloudTimerCallback)
        renderWindowInteractor.SetInteractorStyle(style)
        renderWindowInteractor.Start()

    else:
        renderWindowInteractor.AddObserver('TimerEvent', addPointCloudTimerCallback.next)
        timerId = renderWindowInteractor.CreateRepeatingTimer(ms)
        addPointCloudTimerCallback.timerId = timerId

        renderWindow.Render()
        style = RecordingClickInteractorStyle(renderWindowInteractor, renderWindow, addPointCloudTimerCallback)
        renderWindowInteractor.SetInteractorStyle(style)
        renderWindowInteractor.Start()

    if record:
        moviewriter.End()
        print('wrote',filename)


def getViewportLayouts(viewports: int):
    # Define viewport ranges
    if viewports > 4:
        xmins = np.linspace(0,1-(1/viewports),viewports) #[0 ,  0.2, 0.4, 0.6, 0.8]
        xmaxs = np.linspace(0+(1/viewports),1,viewports)
        ymins = [0] * viewports
        ymaxs = [1] * viewports
    elif viewports == 4:
        xmins = [0 ,  .5, 0 , .5]
        xmaxs = [.5,  1 , .5, 1 ]
        ymins = [0 ,  0 , .5, .5]
        ymaxs = [.5,  .5, 1 , 1 ]
    elif viewports == 3:
        xmins = [0 , 0 , .5]
        xmaxs = [1 , .5, 1 ]
        ymins = [0 , .5, .5]
        ymaxs = [.5, 1 , 1 ]
    elif viewports == 2:
        xmins = [0 , .5]
        xmaxs = [.5, 1 ]
        ymins = [0 , 0 ]
        ymaxs = [1,  1 ]
    elif viewports == 1:
        xmins = [0]
        xmaxs = [1]
        ymins = [0]
        ymaxs = [1]
    else:
        raise NotImplementedError('Too many pointcloud sequences in input. Define new Viewport layout.')

    return xmins, xmaxs, ymins, ymaxs

def setCameraPosition(camera, position='MT'):
    # full view
    # camera.SetPosition(195.27342323718858, 189.34504995328476, -146.12475972226036)
    # camera.SetFocalPoint(9.68813680451325, -0.440555007385222, 201.8577793162133)
    # camera.SetRoll(170.1225630033345)

    # close up
    if position == 'MT':
        camera.SetPosition(26.037883189925843, 40.73903598533825, 14.492523029763333)
        camera.SetFocalPoint(9.68813680451325, -0.440555007385222, 201.8577793162133)
        camera.SetRoll(177.1399258571636)
    elif position == 'RAS':
        camera.SetPosition(-4.605113054626948, 619.970203549004, 6.183503548374842)
        camera.SetFocalPoint(-16.39183144879289, 15.927383417023258, -4.363313192438888)
        camera.SetRoll(-133.2019768161967)

def setupRecording(filename, renderWindow ,ms):
    imageFilter = vtk.vtkWindowToImageFilter()
    imageFilter.SetInput(renderWindow)
    imageFilter.SetInputBufferTypeToRGB()
    imageFilter.ReadFrontBufferOff()
    imageFilter.Update()

    #Setup movie writer
    moviewriter = vtk.vtkOggTheoraWriter() 
    moviewriter.SetInputConnection(imageFilter.GetOutputPort())
    moviewriter.SetFileName(filename)
    moviewriter.SetRate(int(round(1000/ms)))
    moviewriter.Start()

    return imageFilter, moviewriter


def displayPointCloudsViewport(pcs, descriptions, title='Pointcloud Comparison', ms=100, window_size=(1000,1000), record=False):
    """
    Displays pointcloud sequences next to eachother in viewports

    pcs             list of pointcloud sequences as list of vtk pointcloud objects
    descriptions    title of each viewport as list of string with the same length as pcs
    title           window title
    ms              miliseconds between frames
    window_size     window size
    record          when true, the first loop is used to create a video of the window content
    """

    colors = vtk.vtkNamedColors()
    
    # Camera
    # global camera  # uncomment when camera position should be printed
    camera =vtk.vtkCamera()
    setCameraPosition(camera)


    # Render Window
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.SetSize(*window_size)
    renderWindow.SetWindowName(title)

    # Interactor
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)
    renderWindowInteractor.Initialize()

    xmins, xmaxs, ymins, ymaxs = getViewportLayouts(len(pcs))

    # setup a renderer for each pc sequence
    renderers = []
    for i in range(len(pcs)):
        renderer = vtk.vtkRenderer()
        renderer.SetBackground(.2, .3, .4)
        #renderer.ResetCamera()
        renderer.SetActiveCamera(camera)
        renderWindow.AddRenderer(renderer)
        renderer.SetViewport(xmins[i], ymins[i], xmaxs[i], ymaxs[i])

        txt = vtk.vtkTextActor()
        txt.SetInput(descriptions[i])

        renderer.AddActor(txt)
        renderer.AddActor(pcs[i].vtkActor)


        renderers.append(renderer)

    renderWindow.Render()
    style = RecordingClickInteractorStyle(renderWindowInteractor, renderWindow)
    renderWindowInteractor.SetInteractorStyle(style)
    renderWindowInteractor.Start()


class AddPointCloudTimerCallback():

    def __init__(self, renderer, pcs, record=False, moviewriter=None, imageFilter=None, recording_iteration=1):
        self.iterations = 1
        self.renderer = renderer
        self.pcs = pcs
        self.prev = 0
        self.record = record
        self.moviewriter = moviewriter
        self.imageFilter = imageFilter
        self.movie_finished = False
        self.loop_number = 0
        self.RECORDING_ITERATION = recording_iteration


        if len(pcs[0]) == 0:
            raise ValueError('recieved empty pointcloud list')


        # assert that sequences have the same length
        lens = []
        for i in range(len(self.pcs)):
            lens.append(len(pcs[i]))
        assert(all(x == lens[0] for x in lens))

    def next(self, iren, event):
        # ----- used to set initial camera position -----
        # make camera global
        # print('position:    ',camera.GetPosition())
        # print('focal Point: ',camera.GetFocalPoint())
        # print('roll:        ',camera.GetRoll())
        # -----------------------------------------------

        selected_pc = (self.iterations % len(self.pcs[0])) -1

        # singal that the loop restarts
        if selected_pc == -1:
            for i in range(len(self.pcs)):
                self.renderer.RemoveActor(self.pcs[i][self.prev].vtkActor)

            if self.record and not self.movie_finished and self.loop_number == self.RECORDING_ITERATION:
                # Finish movie
                print('finishing')
                self.movie_finished = True

            self.loop_number += 1
        else:
            for i in range(len(self.pcs)):
                self.renderer.RemoveActor(self.pcs[i][self.prev].vtkActor)
                self.renderer.AddActor(self.pcs[i][selected_pc].vtkActor)

            self.prev = selected_pc

        iren.GetRenderWindow().Render()

        if self.record and not self.movie_finished and self.loop_number == self.RECORDING_ITERATION:
            # Export a single frame
            print('writing frame')
            self.imageFilter.Modified()
            self.moviewriter.Write()

        self.iterations += 1

def displayPointCloudSequence(pcs, title='Test', ms=100, window_size=(1000,1000), record=False, filename='movie.ogg'):
    """
    Display a point cloud sequence in a vtk rendered window
    pcs             list of pointclouds as 2d numpy arrays
    title           window title
    ms              number of ms each frame will be showed
    window_size     size of the display window in px

    no return
    """

    # Renderer
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(.2, .3, .4)
    renderer.ResetCamera()
    camera =vtk.vtkCamera()
    
    setCameraPosition(camera)
    renderer.SetActiveCamera(camera)

    # Render Window
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.SetSize(*window_size)
    renderWindow.SetWindowName(title)
    renderWindow.AddRenderer(renderer)

    # Interactor
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)
    renderWindowInteractor.Initialize()

    # make video from output
    if record:
        imageFilter, moviewriter = setupRecording(filename, renderWindow ,ms)
    else:
        imageFilter = None
        moviewriter = None


    # Initialize a timer for the animation
    addPointCloudTimerCallback = AddPointCloudTimerCallbackViewport([renderer], [pcs], record, moviewriter, imageFilter)
    renderWindow.Render()
    style = RecordingClickInteractorStyle(renderWindowInteractor, renderWindow, addPointCloudTimerCallback, fps=8)
    renderWindowInteractor.SetInteractorStyle(style)
    renderWindowInteractor.Start()

    if record:
        moviewriter.End()
        print('wrote',filename)




def displayOverlaidPointCloudSequence(pcs, title='Test', ms=100, window_size=(1000,1000), record=True, filename='movie.ogg'):
    """
    Display a point cloud sequence in a vtk rendered window
    pcs             list of pointclouds as 2d numpy arrays
    title           window title
    ms              number of ms each frame will be showed
    window_size     size of the display window in px

    no return
    """

    # Renderer
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(.2, .3, .4)
    renderer.ResetCamera()
    camera =vtk.vtkCamera()
    
    setCameraPosition(camera)
    renderer.SetActiveCamera(camera)

    # Render Window
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.SetSize(*window_size)
    renderWindow.SetWindowName(title);
    renderWindow.AddRenderer(renderer)

    # Interactor
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)
    renderWindowInteractor.Initialize()

    # make video from output
    if record:
        imageFilter, moviewriter = setupRecording(filename, renderWindow ,ms)
    else:
        imageFilter = None
        moviewriter = None


    # Initialize a timer for the animation
    addPointCloudTimerCallback = AddPointCloudTimerCallback(renderer, pcs, record, moviewriter, imageFilter)
    renderWindowInteractor.AddObserver('TimerEvent', addPointCloudTimerCallback.next)
    timerId = renderWindowInteractor.CreateRepeatingTimer(ms)
    addPointCloudTimerCallback.timerId = timerId

    renderWindow.Render()
    style = RecordingClickInteractorStyle(renderWindowInteractor, renderWindow)
    renderWindowInteractor.SetInteractorStyle(style)
    renderWindowInteractor.Start()

    if record:
        moviewriter.End()
        print('wrote',filename)



def displaySubject(input_dir):
    """
    Automatically loads a timeframe in a scan specified by subject ID and displays it
    input_dir       path to the directory containing the pointclouds

    no return
    """
    pc_filenames = metadata_io.load_pointcloud_paths(input_dir)

    pcs = []

    for i,f in enumerate(pc_filenames):
        if i % 5 == 0:
            print('read',i+1,'of',len(pc_filenames),'files', end='\r')
        pcs.append(VtkTools.VtkPointCloud(point_cloud_io.loadPcd(f)))

    print('read',i+1,'files       ')
    displayPointCloudSequence(pcs)


class MovieWriterCallback():

    def __init__(self, moviewriter, imageFilter):
        self.moviewriter = moviewriter
        self.imageFilter = imageFilter

    def execute(self, iren, event):
        print('write')
        self.imageFilter.Modified()
        self.moviewriter.Write()

class RecordingClickInteractorStyle(vtk.vtkInteractorStyleTrackballCamera): # vtk.vtkInteractorStyleTrackball, vtk.vtkInteractorStyleFlight, vtk.vtkInteractorStyleSwitch
    def __init__(self, interactor: vtk.vtkRenderWindowInteractor, render_window: vtk.vtkRenderWindow, window_manager: AddMeshCallbackViewport = None ,filename='output.ogg', fps=30):
        super().__init__()
        self.window_manager = window_manager
        self.render_window = render_window
        self.interactor = interactor
        if self.window_manager is not None:
            self.interactor.AddObserver('TimerEvent', self.window_manager.next)
        self.AddObserver("CharEvent",self.onKeyPressEvent)
        self.recording = False
        self.fps = fps
        self.image_filename = 'screenshot.png'
        timerId = self.interactor.CreateRepeatingTimer(int(round(1000/self.fps)))
        if self.window_manager is not None:
            self.window_manager.timerId = timerId
        self.playing_video = True

        # set up image filter to pipe into movie writer
        self.imageFilter = vtk.vtkWindowToImageFilter()
        self.imageFilter.SetInput(render_window)
        self.imageFilter.SetInputBufferTypeToRGB()
        self.imageFilter.ReadFrontBufferOff()
        self.imageFilter.Update()

        #Setup movie writer
        self.moviewriter = vtk.vtkOggTheoraWriter() 
        self.moviewriter.SetInputConnection(self.imageFilter.GetOutputPort())
        self.moviewriter.SetFileName(filename)
        self.moviewriter.SetRate(int(round(self.fps)))
        
        #Setup image writer
        self.png_writer = vtk.vtkPNGWriter()
        self.png_writer.SetFileName(self.image_filename)
        self.png_writer.SetInputConnection(self.imageFilter.GetOutputPort())


    def onKeyPressEvent(self, renderer, event):
        #super().onKeyPressEvent()
        key = self.GetInteractor().GetKeySym()
        if key == 's':
            if not self.recording:
                print('staring recording')
                self.recording = True
                self.moviewriter.Start()
                # Initialize a timer for the animation
                self.moviewriter_callback = MovieWriterCallback(self.moviewriter, self.imageFilter)
                self.interactor.AddObserver('TimerEvent', self.moviewriter_callback.execute)
                self.interactor.AddObserver("InteractionEvent",self.moviewriter_callback.execute)
                timerId = self.interactor.CreateRepeatingTimer(int(round(1000/self.fps)))
                self.moviewriter_callback.timerId = timerId
            else:
                print('stop recording')
                self.recording = False
                self.interactor.DestroyTimer()
                self.interactor.RemoveAllObservers()
                self.moviewriter.End()
        elif key == 'c':
            self.imageFilter.Modified()
            self.png_writer.Write()
            print('saved screenshot to', self.image_filename)
        elif key == 'Left' or key == 'comma':
            if self.window_manager is not None:
                self.window_manager.previous(self.interactor,None)
        elif key == 'Right' or key == 'period':
            if self.window_manager is not None:
                self.window_manager.next(self.interactor,None)
        elif key == 'space':
            if self.playing_video:
                self.interactor.DestroyTimer()
                #self.interactor.RemoveAllObservers()
                self.playing_video = False
            else:
                timerId = self.interactor.CreateRepeatingTimer(int(round(1000/self.fps)))
                self.window_manager.timerId = timerId
                self.playing_video = True
        else:
            print(key)

        self.OnChar()

"""
wrapper for show point cloud to conform to new naming convention
"""
def displayPointCloud(pointCloud, record=False, window_size=(1500,1500), title='PointCloud View'):
    showPointCloud(pointCloud, record=record, window_size=window_size, title=title)


"""
displays a point cloud in a vtk rendering window

pointcloud      Nx3 array
record          bool
window_size     size of render window
"""
def showPointCloud(pointCloud, record=False, window_size=(1500,1500), title='PointCloud View', background_color=(.2,.3,.4), pt_size=1):
    pointCloud.vtkActor.GetProperty().SetPointSize(pt_size)

    # Renderer
    renderer = vtk.vtkRenderer()
    renderer.AddActor(pointCloud.vtkActor)
    renderer.SetBackground(*background_color)
    renderer.ResetCamera()

    # Render Window
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.SetSize(*window_size)
    renderWindow.SetWindowName(title)
    renderWindow.AddRenderer(renderer)

    # Interactor
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)

    # Begin Interaction
    renderWindow.Render()
    style = RecordingClickInteractorStyle(renderWindowInteractor, renderWindow)
    renderWindowInteractor.SetInteractorStyle(style)

    if record:    
        style.SetCurrentRenderer(renderer)
        print('press S key to start recording')

    renderWindowInteractor.Start()

    #close_window(renderWindowInteractor)


def displayOverlaidPointCloud(pointCloud1, pointCloud2, record=True, title=None):
    displayOverlaidPointClouds([pointCloud1.vtkActor, pointCloud2.vtkActor], record, title=title)

def displayOverlaidPointClouds(pointCloudList, record=True, title=None, set_background=(.2, .3, .4)):
    # Renderer
    renderer = vtk.vtkRenderer()
    for i in pointCloudList:
        renderer.AddActor(i)
    renderer.SetBackground(*set_background)
    renderer.ResetCamera()

    # Render Window
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    if title is not None:
        renderWindow.SetWindowName(title)

    # Interactor
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)

    # Begin Interaction
    renderWindow.Render()
    style = RecordingClickInteractorStyle(renderWindowInteractor, renderWindow)
    renderWindowInteractor.SetInteractorStyle(style)

    if record:    
        style.SetCurrentRenderer(renderer)
        print('press S key to start recording')

    renderWindowInteractor.Start()


    for i in pointCloudList:
        renderer.RemoveActor(i)
    renderWindow.Finalize()
    renderWindowInteractor.TerminateApp()
    del renderWindow, renderWindowInteractor

    #close_window(renderWindowInteractor)


def close_window(iren):
    print('closing')
    render_window = iren.GetRenderWindow()
    render_window.Finalize()
    iren.TerminateApp()
    del render_window, iren
