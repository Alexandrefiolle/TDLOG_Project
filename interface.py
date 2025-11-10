"This file aims at creating a simple interface which allows the user to open and display an image."

import sys
import PyQt6
from PyQt6.QtCore import Qt
import PyQt6.QtWidgets as widgets
import PyQt6.QtGui as gui
from PIL import Image

<<<<<<< HEAD
class Menu(widgets.QMainWindow):
=======
class Menu(widgets.QGroupBox):
>>>>>>> 7163366 (Implémentation de Vue)
     
    def __init__(self) -> None:
        super().__init__(None)
        assert isinstance(self, widgets.QWidget), "problem"

        self.select_button = widgets.QPushButton("Select an image", self)
        self.select_button.clicked.connect(self.select_button_was_clicked)
        self.original_image_button = widgets.QPushButton("Original image", self)
        self.original_image_button.clicked.connect(self.original_image_button_was_selected)
        self.distances_map_button = widgets.QPushButton("Distances map", self)
        self.distances_map_button.clicked.connect(self.distances_map_button_was_selected)
        self.gradients_map_button = widgets.QPushButton("Gradients map", self)
        self.gradients_map_button.clicked.connect(self.gradients_map_button_was_clicked)
        self.path_button = widgets.QPushButton("Print the optimal path", self)
        self.path_button.clicked.connect(self.path_button_was_clicked)

    def select_button_was_clicked(self) -> None:
        """Handles the button click event to open a file dialog and display the selected image."""
        pass
        #file_name, _ = widgets.QFileDialog.getOpenFileName(self)
        #pic = widgets.QLabel(self)
        #pic.setGeometry(0, 50, 1325, 701)
        #pic.setPixmap(PyQt6.QtGui.QPixmap(file_name).scaledToHeight(500, mode = Qt.TransformationMode.SmoothTransformation))
        #pic.show()

    def original_image_button_was_selected(self) -> None:
        pass

    def distances_map_button_was_selected(self) -> None:
        pass

    def gradients_map_button_was_clicked(self) -> None:
        pass
    
    def path_button_was_clicked(self) -> None:
        pass

class Vue(widgets.QGroupBox):
    pass
    def __init__(self):
        super().__init__(None)
        vertical = widgets.QVBoxLayout(self)
        self.texte = widgets.QLabel("Lorem ipsum ", self)
        vertical.addWidget(self.texte)
        self.image = widgets.QLabel(self)
        vertical.addWidget(self.image)
        self.image.setPixmap(gui.QPixmap("Carte.png").scaledToWidth(1000, mode = Qt.TransformationMode.SmoothTransformation))

    def change_image(self, path):
        self.image.setPixmap(gui.QPixmap(path).scaledToWidth(1000, mode = Qt.TransformationMode.SmoothTransformation))

class Window(widgets.QMainWindow):
    """A simple window class to open and display an image."""
    def __init__(self) -> None:
        """Initializes the main window and its components."""
        super().__init__(None)
        central = widgets.QWidget()
        horizontal = widgets.QHBoxLayout()
        self.vue = Vue()
        self.menu = Menu(self.vue)
        horizontal.addWidget(Menu())
        horizontal.addWidget(Vue())
        central.setLayout(horizontal)
        self.setCentralWidget(central)
 

if __name__ == "__main__":
    application = widgets.QApplication(sys.argv)
    main_window = Window()
    main_window.showMaximized()
    sys.exit(application.exec())

