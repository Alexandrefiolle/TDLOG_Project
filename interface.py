"This file aims at creating a simple interface which allows the user to open and display an image."

from __future__ import annotations 
import sys
import PyQt6
from PyQt6.QtCore import Qt
import PyQt6.QtWidgets as widgets
import PyQt6.QtGui as gui
from PIL import Image
import dijkstra
import point_class as pc
import manipulation as ui

class Vue(widgets.QGroupBox):
    """
    A view class to display an image and some text, 
    which will constitute one of the two parts of the graphic window.
    """
    def __init__(self) -> None:
        """Initializes the view with a label for text and an image display area."""
        super().__init__(None)
        vertical = widgets.QVBoxLayout(self)
        self.texte = widgets.QLabel("Lorem ipsum ", self)
        vertical.addWidget(self.texte)
        self.image = widgets.QLabel(self)
        vertical.addWidget(self.image)
        self.image.setPixmap(gui.QPixmap("Carte.png").scaledToWidth(1000, mode = Qt.TransformationMode.SmoothTransformation))

    def change_image(self, path) -> None:
        """Changes the displayed image to the one located at the given path."""
        self.image.setPixmap(gui.QPixmap(path).scaledToWidth(1000, mode = Qt.TransformationMode.SmoothTransformation))
    
    def print_stocked_image(self, image_name: str) -> None:
        """Displays the image currently stored in the view."""
        self.image.setPixmap(gui.QPixmap(image_name).scaledToWidth(1000, mode = Qt.TransformationMode.SmoothTransformation))

    def mousePressEvent(self, event):
        if self.image.underMouse():
            point = gui.QCursor.pos()
            print("clicked", point.x(), point.y())
        
class Menu(widgets.QGroupBox):
    """
    A menu class with buttons to interact with the image view,
    to enable functionalities such as loading an image, 
    displaying different maps, and printing the optimal path.
    """ 
    def __init__(self, vue: Vue) -> None:
        """Initializes the menu with buttons linked to various functionalities."""
        super().__init__(None)
        self.select_button = widgets.QPushButton("Select an image", self)
        self.select_button.setGeometry(10, 10, 150, 30)
        self.select_button.clicked.connect(self.select_button_was_clicked)
        self.original_image_button = widgets.QPushButton("Original image", self)
        self.original_image_button.setGeometry(10, 50, 150, 30)
        self.original_image_button.clicked.connect(self.original_image_button_was_selected)
        self.distances_map_button = widgets.QPushButton("Distances map", self)
        self.distances_map_button.setGeometry(10, 90, 150, 30)
        self.distances_map_button.clicked.connect(self.distances_map_button_was_selected)
        self.gradients_map_button = widgets.QPushButton("Gradients map", self)
        self.gradients_map_button.setGeometry(10, 130, 150, 30)
        self.gradients_map_button.clicked.connect(self.gradients_map_button_was_clicked)
        self.path_button = widgets.QPushButton("Print the optimal path", self)
        self.path_button.setGeometry(10, 170, 150, 30)
        self.path_button.clicked.connect(self.path_button_was_clicked)
        self._vue = vue
        self._original_image_name = 'Carte.png'
        self._distances_map_image_name = 'distances_map.png'
        self._distances_map_computed = False
        self._gradients_map_image_name = 'gradients_map.png'
        self._gradients_map_computed = False

    def select_button_was_clicked(self) -> None:
        """Handles the button click event to open a file dialog and display the selected image."""
        file_name, _ = widgets.QFileDialog.getOpenFileName(self)
        self._vue.change_image(file_name)
        self._original_image_name = file_name
        self._distances_map_computed = False
        self._gradients_map_computed = False

    def original_image_button_was_selected(self) -> None:
        """Handles the button click event to display the original image."""
        self._vue.print_stocked_image(self._original_image_name)

    def distances_map_creation(self, start: pc.Point, end: pc.Point) -> None:
        """Creates the distances map and stores it in the corresponding view."""
        im = ui.GreyImage(self._original_image_name)
        distances_map_image = dijkstra.distances_map(start, end, im)
        img = Image.fromarray(distances_map_image)
        img.save(self._distances_map_image_name)
        self._distances_map_computed = True

    def distances_map_button_was_selected(self) -> None:
        """Handles the button click event to display the distances map."""
        if not self._distances_map_computed:
            start = pc.Point(10,10)
            end = pc.Point(400,400)
            self.distances_map_creation(start, end)
            self._vue.print_stocked_image(self._distances_map_image_name)
        else:
            self._vue.print_stocked_image(self._distances_map_image_name)

    def gradients_map_button_was_clicked(self) -> None:
        """Handles the button click event to display the gradients map."""
        pass
    
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
        horizontal.addWidget(self.menu)
        horizontal.addWidget(self.vue)
        central.setLayout(horizontal)
        self.setCentralWidget(central)
 

if __name__ == "__main__":
    application = widgets.QApplication(sys.argv)
    main_window = Window()
    main_window.showMaximized()
    sys.exit(application.exec())

