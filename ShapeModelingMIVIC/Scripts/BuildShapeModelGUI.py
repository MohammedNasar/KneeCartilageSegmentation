#!/usr/bin/env python

import sys
import vtk

from PyQt4 import Qt, QtGui, QtCore
# from PyQt4.QtGui import *
from vtk.qt4.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

class MainWindow(Qt.QMainWindow):
    """Test class"""
    def __init__(self, parent = None):
        Qt.QMainWindow.__init__(self, parent)
        self.initUI()
        
    def initUI(self):
        MainWindow = self
        MainWindow.setObjectName("MainWindow")
        MainWindow.setWindowTitle("NIA 3T Test")
        MainWindow.resize(600, 600)
        self.workspace = QtGui.QWorkspace()
        MainWindow.setCentralWidget(self.workspace)
        self.frame = QtGui.QFrame(self.workspace) 
        self.hbox = QtGui.QHBoxLayout()

        # ToDo: add functionality to 'file open' button.
        self.pushButton = QtGui.QPushButton(MainWindow)
        self.pushButton.setGeometry(QtCore.QRect(70, 200, 115, 26))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.setText("Open List")
#        self.gridlayout.addWidget(self.pushButton, 0, 0, 1, 1)

        self.listWidget = QtGui.QListWidget(MainWindow)
        self.listWidget.setGeometry(QtCore.QRect(120, 120, 256, 192))
        self.listWidget.setObjectName("listWidget")

        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)

        self.hbox.addWidget(self.pushButton)
        self.hbox.addWidget(self.listWidget)
        self.hbox.addWidget(self.vtkWidget)

        self.connect(self.pushButton,Qt.SIGNAL("clicked()"),self.pushButtonCallback)
        
        self.frame.setLayout(self.hbox)
        self.workspace.addWindow(self.frame)
        self.displayGraphics2()

    def pushButtonCallback(self):
        print "Hello!"
        

    def displayGraphics(self):
        widget = self.vtkWidget
        widget.Initialize()
        widget.Start()

##       widget.AddObserver("ExitEvent", lambda o, e, a=app: a.quit())

        ren = vtk.vtkRenderer()
        widget.GetRenderWindow().AddRenderer(ren)

        cone = vtk.vtkConeSource()
        cone.SetResolution(24)

        coneMapper = vtk.vtkPolyDataMapper()
        coneMapper.SetInput(cone.GetOutput())

        coneActor = vtk.vtkActor()
        coneActor.SetMapper(coneMapper)
        ren.AddActor(coneActor)
        widget.show()
        
        
        
    def displayGraphics2(self):
        # Input
        VTK_DATA_ROOT = "/home/sokratis/Workspace/VTKData"
        reader = vtk.vtkImageReader2()
        reader.SetFilePrefix(VTK_DATA_ROOT + "/Data/headsq/quarter")
        reader.SetDataExtent(0, 63, 0, 63, 1, 93)
        # reader.SetDataSpacing(3.2, 3.2, 1.5)
        reader.SetDataOrigin(0.0, 0.0, 0.0)
        reader.SetDataScalarTypeToUnsignedShort()
        reader.UpdateWholeExtent()        
        
        # Calculate the center of the volume
        reader.GetOutput().UpdateInformation()
        (xMin, xMax, yMin, yMax, zMin, zMax) = reader.GetOutput().GetWholeExtent()
        (xSpacing, ySpacing, zSpacing) = reader.GetOutput().GetSpacing()
        (x0, y0, z0) = reader.GetOutput().GetOrigin()

        center = [x0 + xSpacing * 0.5 * (xMin + xMax),
                  y0 + ySpacing * 0.5 * (yMin + yMax),
              z0 + zSpacing * 0.5 * (zMin + zMax)]

        # Matrices for axial, coronal, sagittal, oblique view orientations
        axial = vtk.vtkMatrix4x4()
        axial.DeepCopy((1, 0, 0, center[0],
                        0, 1, 0, center[1],
                        0, 0, 1, center[2],
                        0, 0, 0, 1))
    
        # Extract a slice in the desired orientation
        reslice = vtk.vtkImageReslice()
        reslice.SetInputConnection(reader.GetOutputPort())
        reslice.SetOutputDimensionality(2)
        reslice.SetResliceAxes(axial)
        reslice.SetInterpolationModeToLinear()        
        
        # Mapper
        # Create a greyscale lookup table
        table = vtk.vtkLookupTable()
        table.SetRange(0, 2000) # image intensity range
        table.SetValueRange(0.0, 1.0) # from black to white
        table.SetSaturationRange(0.0, 0.0) # no color saturation
        table.SetRampToLinear()
        table.Build()

        # Map the image through the lookup table
        color = vtk.vtkImageMapToColors()
        color.SetLookupTable(table)
        color.SetInputConnection(reslice.GetOutputPort())        

        # Actor
        actor = vtk.vtkImageActor()
        actor.SetInput(color.GetOutput())
        
        # Create the renderer, the render window, and the interactor. The
        # renderer draws into the render window, the interactor enables mouse-
        # and keyboard-based interaction with the scene.
        widget = self.vtkWidget
        widget.Initialize()
        widget.Start()

        renderer = vtk.vtkRenderer()
        renwin = widget.GetRenderWindow()
        renwin.AddRenderer(renderer)
  
        # Set up the interaction
        interactorStyle = vtk.vtkInteractorStyleImage()
        interactor = renwin.GetInteractor()
#        if isinstance(interactor, vtk.vtkRenderWindowInteractor):
#            print "interactor is vtkRenderWindowInteractor instance"
#        else:
#            print "interactor is " + interactor.GetClassName() + " instance"
        interactor.SetInteractorStyle(interactorStyle)
        interactor.SetRenderWindow(renwin)
        renwin.SetInteractor(interactor)

        
        # Create callbacks for slicing the image
        actions = {}
        actions["Slicing"] = 0
        
        def ButtonCallback(obj, event):
            if event == "LeftButtonPressEvent":
                actions["Slicing"] = 1
            else:
                actions["Slicing"] = 0
        
        def MouseMoveCallback(obj, event):
            (lastX, lastY) = interactor.GetLastEventPosition()
            (mouseX, mouseY) = interactor.GetEventPosition()
            if actions["Slicing"] == 1:
                deltaY = mouseY - lastY
                reslice.GetOutput().UpdateInformation()
                sliceSpacing = reslice.GetOutput().GetSpacing()[2]
                matrix = reslice.GetResliceAxes()
                # move the center point that we are slicing through
                center = matrix.MultiplyPoint((0, 0, sliceSpacing*deltaY, 1))
                matrix.SetElement(0, 3, center[0])
                matrix.SetElement(1, 3, center[1])
                matrix.SetElement(2, 3, center[2])
                renwin.Render()
            else:
                interactorStyle.OnMouseMove()
                
        
        interactorStyle.AddObserver("MouseMoveEvent", MouseMoveCallback)
        interactorStyle.AddObserver("LeftButtonPressEvent", ButtonCallback)
        interactorStyle.AddObserver("LeftButtonReleaseEvent", ButtonCallback)    

        renderer.AddActor(actor)
        interactor.Start()
        widget.show()
 
 


if __name__ == "__main__":
    app = Qt.QApplication(sys.argv)
    form = MainWindow()
    form.show()
    sys.exit(app.exec_())
