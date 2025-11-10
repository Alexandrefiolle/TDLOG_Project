"This file aims at creating a simple interface which allows the user to open and display an image."

import sys
import PyQt6
from PyQt6.QtCore import Qt
import PyQt6.QtWidgets as widgets
from PIL import Image

class Menu:
     
    def __init__(self):
        self.a_button = widgets.QPushButton("click me", self)
        self.a_button.clicked.connect(self.a_button_was_clicked)

    def a_button_was_clicked(self) -> None:
        """Handles the button click event to open a file dialog and display the selected image."""
        file_name, _ = widgets.QFileDialog.getOpenFileName(self)
        pic = widgets.QLabel(self)
        pic.setGeometry(0, 50, 1325, 701)
        pic.setPixmap(PyQt6.QtGui.QPixmap(file_name).scaledToHeight(500, mode = Qt.TransformationMode.SmoothTransformation))
        pic.show()

class Vue:

    def __init__():
        pass

class Window(widgets.QMainWindow):
    """A simple window class to open and display an image."""
    def __init__(self) -> None:
        """Initializes the main window and its components."""
        super().__init__(None)
        central = widgets.QWidget()
        horizontal = widgets.QHBoxLayout()
        horizontal.addWidget(Menu())
        horizontal.addWidget(Vue())
        central.setLayout(horizontal)
        self.setCentralWidget(central)

       
 

if __name__ == "__main__":
    application = widgets.QApplication(sys.argv)
    main_window = Window()
    main_window.showMaximized()
    sys.exit(application.exec())

