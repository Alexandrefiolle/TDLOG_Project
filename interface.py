"This file aims at creating a simple interface which allows the user to open and display an image."

from __future__ import annotations 
import sys
import PyQt6
from PyQt6.QtCore import Qt, QPoint
import PyQt6.QtWidgets as widgets
import PyQt6.QtGui as gui
from PIL import Image
import dijkstra
import point_class as pc
import manipulation as ui

class Chargement(widgets.QProgressBar):

    def __init__(self, v_init = 0):
        super().__init__()
        self.maxi = v_init
        self.setValue(0)

    def reinitialise(self, v_init):
        self.maxi = v_init
        self.setValue(0)

    def update(self, value):
        if 100 - 100*value//self.maxi > self.value():
            self.setValue(int(100 - 100*value/self.maxi))

class Fenetre(widgets.QLabel):

    def __init__(self, parent):
        super().__init__(parent)
        self.ps = None
        self.pe = None

    def mousePressEvent(self, event):
        if self.underMouse():
            point = gui.QCursor.pos()
            print("clicked", point.x(), point.y())
            point = self.mapFromGlobal(point)
            print("mapped", point.x(), point.y())
            point = pc.Point(int(self.parent().ratio*point.x()), int(self.parent().ratio*point.y()))
            if self.parent()._menu.starting_point is None:
                self.ps = self.mapFromGlobal(gui.QCursor.pos())
                self.parent()._menu.starting_point = point
                print("Starting point set to:", point)
                self.parent().texte.setText("Select an ending point")
            elif self.parent()._menu.ending_point is None:
                self.pe = self.mapFromGlobal(gui.QCursor.pos())
                self.parent()._menu.ending_point = point
                print("Ending point set to:", point)
                self.parent()._menu._starting_and_ending_points_set = True
                self.parent().texte.setText("Compute a distance map")
            else:
                print("Both starting and ending points are already set.")
            self.update()
    
    def paintEvent(self, a0):
        painter = gui.QPainter()
        painter.begin(self)
        painter.drawPixmap(QPoint(), self.pixmap())
        if self.ps is not None:
            painter.setBrush(gui.QBrush(gui.QColorConstants.Green))
            painter.drawEllipse(self.ps, 5, 5)
        if self.pe is not None:
            painter.setBrush(gui.QBrush(gui.QColorConstants.Red))
            painter.drawEllipse(self.pe, 5, 5)
        painter.end()

class Vue(widgets.QGroupBox):
    """
    A view class to display an image and some text, 
    which will constitute one of the two parts of the graphic window.
    """
    def __init__(self) -> None:
        """Initializes the view with a label for text and an image display area."""
        super().__init__(None)
        self.setFixedWidth(1000)
        vertical = widgets.QVBoxLayout(self)
        self.texte = widgets.QLabel("Select a starting point", self)
        self.texte.setSizePolicy(widgets.QSizePolicy.Policy.Minimum, widgets.QSizePolicy.Policy.Fixed)
        vertical.addWidget(self.texte)
        self.image = Fenetre(self)
        vertical.addWidget(self.image)
        img = gui.QPixmap("Carte.png")
        self.ratio = img.width()/1000
        self.image.setPixmap(img.scaled(1000, 700, aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio, transformMode=Qt.TransformationMode.SmoothTransformation))
        self._menu = None
        self.bar = Chargement()
        self.bar.setFixedWidth(1000)
        self.bar.setSizePolicy(widgets.QSizePolicy.Policy.Minimum, widgets.QSizePolicy.Policy.Fixed)
        vertical.addWidget(self.bar)
        self.bar.hide()


    @property
    def menu(self) -> Menu:
        """Returns the menu associated with the view."""
        return self._menu
    @menu.setter
    def menu(self, menu: Menu) -> None:
        """Sets the menu associated with the view."""
        self._menu = menu

    def change_image(self, path) -> None:
        """Changes the displayed image to the one located at the given path."""
        self.image.setPixmap(gui.QPixmap(path).scaled(1000, 700, aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio, transformMode=Qt.TransformationMode.SmoothTransformation))
    
    def print_stocked_image(self, image_name: str) -> None:
        """Displays the image currently stored in the view."""
        self.image.setPixmap(gui.QPixmap(image_name).scaled(1000, 700, aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio, transformMode=Qt.TransformationMode.SmoothTransformation))
        
        
class Menu(widgets.QGroupBox):
    """
    A menu class with buttons to interact with the image view,
    to enable functionalities such as loading an image, 
    displaying different maps, and printing the optimal path.
    """ 
    def __init__(self, vue: Vue) -> None:
        """Initializes the menu with buttons linked to various functionalities."""
        super().__init__(None)
        self.setFixedWidth(500)
        self.select_button = widgets.QPushButton("Select an image", self)
        self.select_button.setGeometry(10, 10, 150, 30)
        self.select_button.clicked.connect(self.select_button_was_clicked)
        self.select_button_erase_points = widgets.QPushButton("Erase the points", self)
        self.select_button_erase_points.setGeometry(10, 50, 150, 30)
        self.select_button_erase_points.clicked.connect(self.erase_points_was_clicked)
        self.original_image_button = widgets.QPushButton("Original image", self)
        self.original_image_button.setGeometry(10, 90, 150, 30)
        self.original_image_button.clicked.connect(self.original_image_button_was_selected)
        self.distances_map_button = widgets.QPushButton("Distances map", self)
        self.distances_map_button.setGeometry(10, 130, 150, 30)
        self.distances_map_button.clicked.connect(self.distances_map_button_was_selected)
        self.gradients_map_button = widgets.QPushButton("Gradients map", self)
        self.gradients_map_button.setGeometry(10, 170, 150, 30)
        self.gradients_map_button.clicked.connect(self.gradients_map_button_was_clicked)
        self.path_button = widgets.QPushButton("Print the optimal path", self)
        self.path_button.setGeometry(10, 210, 150, 30)
        self.path_button.clicked.connect(self.path_button_was_clicked)
        self._vue = vue
        self._original_image_name = 'Carte.png'
        self._distances_map_image_name = 'distances_map.png'
        self._distances_map_computed = False
        self._gradients_map_image_name = 'gradients_map.png'
        self._gradients_map_computed = False
        self._starting_point = None
        self._ending_point = None
        self._starting_and_ending_points_set = False
        self.obs = Observer()
    
    @property
    def starting_point(self) -> pc.Point:
        """Returns the starting point for pathfinding."""
        return self._starting_point
    @starting_point.setter
    def starting_point(self, point: pc.Point) -> None:
        """Sets the starting point for pathfinding."""
        self._starting_point = point
    @property
    def ending_point(self) -> pc.Point:
        """Returns the ending point for pathfinding."""
        return self._ending_point
    @ending_point.setter
    def ending_point(self, point: pc.Point) -> None:
        """Sets the ending point for pathfinding."""
        self._ending_point = point

    def select_button_was_clicked(self) -> None:
        """Handles the button click event to open a file dialog and display the selected image."""
        file_name, _ = widgets.QFileDialog.getOpenFileName(self)
        self._vue.change_image(file_name)
        self._original_image_name = file_name
        self._distances_map_computed = False
        self._gradients_map_computed = False
        self.erase_points_was_clicked()
        self._vue.ratio = gui.QPixmap(file_name).width()/1000
    
    def erase_points_was_clicked(self) -> None:
        self._vue.image.ps = None
        self._vue.image.pe = None
        self._starting_point = None
        self._ending_point = None
        self._starting_and_ending_points_set = False
        self._distances_map_computed = False
        self._gradients_map_computed = False
        self._vue.image.update()
        self._vue.texte.setText("Select a starting point")


    def original_image_button_was_selected(self) -> None:
        """Handles the button click event to display the original image."""
        self._vue.print_stocked_image(self._original_image_name)

    def distances_map_creation(self, start: pc.Point, end: pc.Point) -> None:
        """Creates the distances map and stores it in the corresponding view."""
        self._vue.bar.reinitialise(start.norm(end))
        self._vue.bar.show()
        self.obs.add_observer(self._vue.bar)
        im = ui.GreyImage(self._original_image_name)
        print("Starting point set to:", start)
        print("Ending point set to:", end)
        distances_map_image = dijkstra.distances_map(start, end, im, self.obs)
        img = Image.fromarray(distances_map_image)
        img.save(self._distances_map_image_name)
        self._distances_map_computed = True
        self._vue.texte.setText("Compute a gradient map")
        self._vue.bar.hide()
        self.obs.del_observer(self._vue.bar)

    def distances_map_button_was_selected(self) -> None:
        """Handles the button click event to display the distances map."""
        if self._distances_map_computed:
            self._vue.print_stocked_image(self._distances_map_image_name)
        elif self._starting_and_ending_points_set:
            self.distances_map_creation(self._starting_point, self._ending_point)
            self._vue.print_stocked_image(self._distances_map_image_name)
        else:
            print("Please select starting and ending points by clicking on the image.")
        """
        if not self._distances_map_computed:
            start = pc.Point(10,10)
            end = pc.Point(400,400)
            self.distances_map_creation(start, end)
            self._vue.print_stocked_image(self._distances_map_image_name)
        else:
            self._vue.print_stocked_image(self._distances_map_image_name)
        """

    def gradients_map_creation(self, start: pc.Point, end: pc.Point) -> None:
        """
        Creates the gradients map and stores it in the corresponding view.
        Using the functions from dijkstra.py
        """
        #self._vue.bar.reinitialise(start.norm(end))
        #self._vue.bar.show()
        #self.obs.add_observer(self._vue.bar)
        im = ui.GreyImage(self._original_image_name)
        print("Starting point set to:", start)
        print("Ending point set to:", end)
        distances = dijkstra.distances_costs(start, end, im, self.obs)
        gradients_map_image = dijkstra.gradient_on_image(distances, im)
        img = Image.fromarray(gradients_map_image)
        img.save(self._gradients_map_image_name)
        self._gradients_map_computed = True
        self._vue.texte.setText("You can now print the optimal path")
        #self._vue.bar.hide()
        

    def gradients_map_button_was_clicked(self) -> None:
        """Handles the button click event to display the gradients map."""
        if self._gradients_map_computed:
            self._vue.print_stocked_image(self._gradients_map_image_name)
        elif self._starting_and_ending_points_set:
            self.gradients_map_creation(self._starting_point, self._ending_point)
            self._vue.print_stocked_image(self._gradients_map_image_name)
        else:
            print("Please select starting and ending points by clicking on the image.")
    
    def path_button_was_clicked(self) -> None:
        """Handles the button click event to print the optimal path."""
        pass

class Window(widgets.QMainWindow):
    """A simple window class to open and display an image."""
    def __init__(self) -> None:
        """Initializes the main window and its components."""
        super().__init__(None)
        central = widgets.QWidget()
        horizontal = widgets.QHBoxLayout()
        self.vue = Vue()
        self.menu = Menu(self.vue)
        self.vue.menu = self.menu
        horizontal.addWidget(self.menu)
        horizontal.addWidget(self.vue)
        central.setLayout(horizontal)
        self.setCentralWidget(central)
 
class Observer:

    def __init__(self):
        self.liste = list()

    def add_observer(self, ob):
        self.liste.append(ob)

    def del_observer(self, ob):
        if ob in self.liste:
            self.liste.remove(ob)

    def notify_observer(self, value):
        for ob in self.liste:
            ob.update(value)

if __name__ == "__main__":
    application = widgets.QApplication(sys.argv)
    main_window = Window()
    main_window.showMaximized()
    sys.exit(application.exec())

