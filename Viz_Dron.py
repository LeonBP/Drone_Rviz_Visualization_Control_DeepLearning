#!/usr/bin/env python

## BEGIN_TUTORIAL
##
## Imports
## ^^^^^^^
##
## First we start with the standard ros Python import line:
import roslib; roslib.load_manifest('rviz_python_tutorial')
from std_msgs.msg import String
## Then load sys to get sys.argv.
import sys
import rospy
import time
from Config import *
from visualization_msgs.msg import Marker
## Next import all the Qt bindings into the current namespace, for
## convenience.  This uses the "python_qt_binding" package which hides
## differences between PyQt and PySide, and works if at least one of
## the two is installed.  The RViz Python bindings use
## python_qt_binding internally, so you should use it here as well.
from python_qt_binding.QtGui import *
from python_qt_binding.QtCore import *
from python_qt_binding.QtWidgets import *

## Finally import the RViz bindings themselves.
import rviz

## The MyViz class is the main container widget.
class MyViz( QWidget ):

    ## MyViz Constructor
    ## ^^^^^^^^^^^^^^^^^
    ##
    ## Its constructor creates and configures all the component widgets:
    ## frame, thickness_slider, top_button, and side_button, and adds them
    ## to layouts.

    def callback(self,data):
        points = data.points
        if len(points) < 2:
            print "ERROR"
        else:
            p1 = points[0]
            p2 = points[1]
            string = "Movement from ("+p1.x.__str__()+","+p1.y.__str__()+","+p1.z.__str__()+") to ("+p2.x.__str__()+","+p2.y.__str__()+","+p2.z.__str__()+")"
            self.text.setText(string)
            self.text.update()

    def __init__(self):
        QWidget.__init__(self)
        rospy.init_node("Vis")
        self.pub = rospy.Publisher("/Output",String,latch=False,queue_size = 1)
        self.pos_sub = rospy.Subscriber("/suggested_movement", Marker,self.callback,queue_size = 1)
        ## rviz.VisualizationFrame is the main container widget of the
        ## regular RViz application, with menus, a toolbar, a status
        ## bar, and many docked subpanels.  In this example, we
        ## disable everything so that the only thing visible is the 3D
        ## render window.
        self.frame = rviz.VisualizationFrame()

        ## The "splash path" is the full path of an image file which
        ## gets shown during loading.  Setting it to the empty string
        ## suppresses that behavior.
        self.frame.setSplashPath( "" )

        ## VisualizationFrame.initialize() must be called before
        ## VisualizationFrame.load().  In fact it must be called
        ## before most interactions with RViz classes because it
        ## instantiates and initializes the VisualizationManager,
        ## which is the central class of RViz.
        self.frame.initialize()

        ## The reader reads config file data into the config object.
        ## VisualizationFrame reads its data from the config object.
        reader = rviz.YamlConfigReader()
        config = rviz.Config()
        reader.readFile( config, RVIZ_FILE )
        self.frame.load( config )

        ## You can also store any other application data you like in
        ## the config object.  Here we read the window title from the
        ## map key called "Title", which has been added by hand to the
        ## config file.
        self.setWindowTitle( config.mapGetChild( "Title" ).getValue() )

        ## Here we disable the menu bar (from the top), status bar
        ## (from the bottom), and the "hide-docks" buttons, which are
        ## the tall skinny buttons on the left and right sides of the
        ## main render window.
        self.frame.setMenuBar( None )
        self.frame.setStatusBar( None )
        self.frame.setHideButtonVisibility( False )

        ## frame.getManager() returns the VisualizationManager
        ## instance, which is a very central class.  It has pointers
        ## to other manager objects and is generally required to make
        ## any changes in an rviz instance.
        self.manager = self.frame.getManager()

        ## Since the config file is part of the source code for this
        ## example, we know that the first display in the list is the
        ## grid we want to control.  Here we just save a reference to
        ## it for later.
        self.grid_display = self.manager.getRootDisplayGroup().getDisplayAt( 0 )
        self.text = QLabel("Movement: ")
        ## Here we create the layout and other widgets in the usual Qt way.
        layout = QVBoxLayout()
        layout.addWidget( self.frame )
        h_layout = QHBoxLayout()
        top_button = QPushButton( "Acept Movement? (Green Arrow)" )
        top_button.clicked.connect( self.onTopButtonClick )
        h_layout.addWidget( top_button )
        layout.addWidget(self.text)
        No_button = QPushButton( "Reject Movement?" )
        No_button.clicked.connect( self.NoButtonClick )
        h_layout.addWidget( No_button )
        layout.addLayout(h_layout)
        self.setLayout( layout )

    ## Handle GUI events
    ## ^^^^^^^^^^^^^^^^^
    ##
    ## After the constructor, for this example the class just needs to
    ## respond to GUI events.  Here is the slider callback.
    ## rviz.Display is a subclass of rviz.Property.  Each Property can
    ## have sub-properties, forming a tree.  To change a Property of a
    ## Display, use the subProp() function to walk down the tree to
    ## find the child you need.
    def NoButtonClick( self ):
        print "Petition Denied"
        #time.sleep(2)
        self.text.setText("Petition Denied")
        self.text.update()
        self.pub.publish(String("No"))	
    ## The view buttons just call switchToView() with the name of a saved view.
    def onTopButtonClick( self ):
        print "Petition Acepted"
        self.text.setText("Petition Acepted")
        self.text.update()
        self.pub.publish(String("Ok"))
  

## Start the Application
## ^^^^^^^^^^^^^^^^^^^^^
##
## That is just about it.  All that's left is the standard Qt
## top-level application code: create a QApplication, instantiate our
## class, and start Qt's main event loop (app.exec_()).
if __name__ == '__main__':
    app = QApplication( sys.argv )

    myviz = MyViz()
    #myviz.resize( 500, 500 )
    myviz.show()

    app.exec_()
