"This file aims at creating a simple interface which allows the user to open and display an image."

from __future__ import annotations 
import numpy as np
import sys
from PyQt6.QtCore import Qt, QPoint
import PyQt6.QtWidgets as widgets
import PyQt6.QtGui as gui
from PIL import Image
import dijkstra
import point_class as pc
import manipulation as ui
import edge_detection as edge
import observer as obs

class Chargement(widgets.QProgressBar):
    """A progress bar class to indicate the loading status during computations."""
    def __init__(self, v_init: int = 0) -> None:
        """Initializes the progress bar with a maximum value."""
        super().__init__()
        self.maxi = v_init
        self.setValue(0)

    def reinitialise(self, v_init: int) -> None:
        """Reinitializes the progress bar with a new given maximum value."""
        self.maxi = v_init
        self.setValue(0)

    def update(self, value: int) -> None:
        """Updates the progress bar based on the current value."""
        if 100 - 100*value//self.maxi > self.value():
            self.setValue(int(100 - 100*value/self.maxi))

class Fenetre(widgets.QLabel):
    """A window class to display an image and handle mouse click events"""
    def __init__(self, parent: Vue) -> None:
        """Initializes the window with a parent view."""
        super().__init__(parent)
        self.ps = None
        self.pe = None

    def mousePressEvent(self, event: gui.QMouseEvent) -> None:
        """Handles mouse press events to select starting and ending points on the image."""
        if self.underMouse():
            point = gui.QCursor.pos()
            point = self.mapFromGlobal(point)
            point = pc.Point(int(self.parent().ratio*point.x()), int(self.parent().ratio*point.y()))
            if self.parent()._menu.contour_mode:
                self.parent()._menu.contour_points.append(point)
                if len(self.parent()._menu.contour_points) == 1:
                    self.ps = self.mapFromGlobal(gui.QCursor.pos())  # first point
                else:
                    self.pe = self.mapFromGlobal(gui.QCursor.pos())  # second point
                self.update()
                self.parent()._menu._vue.texte.setText(f"{len(self.parent()._menu.contour_points)}/2 points chosen for the contour")
                return
            if self.parent()._menu.starting_point is None: # first point
                self.ps = self.mapFromGlobal(gui.QCursor.pos())
                self.parent()._menu.starting_point = point
                self.parent().texte.setText("Select an ending point")
            elif self.parent()._menu.ending_point is None: # second point
                self.pe = self.mapFromGlobal(gui.QCursor.pos())
                self.parent()._menu.ending_point = point
                self.parent()._menu._starting_and_ending_points_set = True
                self.parent().texte.setText("Compute a distance map")
            else: # both points are already set
                self.parent().texte.setText("Both starting and ending points are already set.")
            self.update()
    
    def paintEvent(self, a0:gui.QPaintEvent) -> None:
        """Handles the paint event to draw the selected points on the image."""
        painter = gui.QPainter()
        painter.begin(self)
        painter.drawPixmap(QPoint(), self.pixmap())
        if self.ps is not None: # first point
            painter.setBrush(gui.QBrush(gui.QColorConstants.Green))
            painter.drawEllipse(self.ps, 5, 5)
        if self.pe is not None: # second point
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
        self.texte.setSizePolicy(widgets.QSizePolicy.Policy.Minimum, 
                                 widgets.QSizePolicy.Policy.Fixed)
        vertical.addWidget(self.texte)
        self.image = Fenetre(self)
        vertical.addWidget(self.image)
        img = gui.QPixmap("images/Carte.png")
        self.ratio = max(img.width()/1000, img.height()/700)
        self.image.setPixmap(img.scaled(1000, 700, 
                                        aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio, 
                                        transformMode=Qt.TransformationMode.SmoothTransformation))
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
        self.image.setPixmap(gui.QPixmap(path).scaled(1000, 700, 
                                                      aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio, 
                                                      transformMode=Qt.TransformationMode.SmoothTransformation))
    
    def print_stocked_image(self, image_name: str) -> None:
        """Displays the image currently stored in the view."""
        self.image.setPixmap(gui.QPixmap(image_name).scaled(1000, 700, 
                                                            aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio, 
                                                            transformMode=Qt.TransformationMode.SmoothTransformation))
        
        
class Menu(widgets.QGroupBox):
    """
    A menu class with buttons to interact with the image view,
    to enable functionalities such as loading an image, 
    displaying different maps, and printing the optimal path.
    """ 
    def __init__(self, vue: Vue) -> None:
        """Initializes the menu with buttons linked to various functionalities."""
        super().__init__(None)
        self.setFixedWidth(200)
        # Buttons
        # Selection button
        self.select_button = widgets.QPushButton("Select an image", self)
        self.select_button.setGeometry(10, 10, 150, 30)
        self.select_button.clicked.connect(self.select_button_was_clicked)
        # Erase points button
        self.select_button_erase_points = widgets.QPushButton("Erase the points", self)
        self.select_button_erase_points.setGeometry(10, 50, 150, 30)
        self.select_button_erase_points.clicked.connect(self.erase_points_was_clicked)
        # Original image button
        self.original_image_button = widgets.QPushButton("Original image", self)
        self.original_image_button.setGeometry(10, 90, 150, 30)
        self.original_image_button.clicked.connect(self.original_image_button_was_selected)
        # Distances map button
        self.distances_map_button = widgets.QPushButton("Distances map", self)
        self.distances_map_button.setGeometry(10, 130, 150, 30)
        self.distances_map_button.clicked.connect(self.distances_map_button_was_selected)
        # Gradients map button
        self.gradients_map_button = widgets.QPushButton("Gradients map", self)
        self.gradients_map_button.setGeometry(10, 170, 150, 30)
        self.gradients_map_button.clicked.connect(self.gradients_map_button_was_clicked)
        # Path button
        self.path_button = widgets.QPushButton("Print the optimal path", self)
        self.path_button.setGeometry(10, 210, 150, 30)
        self.path_button.clicked.connect(self.path_button_was_clicked)
        # Edge detection button
        self.edge_detection_button = widgets.QPushButton("Edge detection", self)
        self.edge_detection_button.setGeometry(10, 250, 150, 30)
        self.edge_detection_button.clicked.connect(self.edge_detection_button_was_clicked)
        # Reset edge detection button
        self.reset_edge_button = widgets.QPushButton("Reset edge detection", self)
        self.reset_edge_button.setGeometry(10, 290, 150, 30)
        self.reset_edge_button.clicked.connect(self.reset_edge_detection)
        # Next edge image button
        self.next_edge_button = widgets.QPushButton("Next image →", self)
        self.next_edge_button.setGeometry(10, 330, 180, 40) 
        self.next_edge_button.clicked.connect(self.show_next_edge_image)
        self.next_edge_button.hide()
        # edge button
        self.contour_button = widgets.QPushButton("Draw the edge", self)
        self.contour_button.setGeometry(10, 370, 180, 40)
        self.contour_button.clicked.connect(self.trace_contour)
        self.contour_button.hide()

        self._vue = vue
        # starting image
        self._original_image_name = 'images/Carte.png'
        self._original_image_grey_level = ui.GreyImage(self._original_image_name)
        self._distances_map_image_name = 'distances_map.png'
        # distances map
        self._list_visited = []
        self._distances_map_computed = False
        self._distances_costs = None
        # gradients map
        self._gradients_map_image_name = 'gradients_map.png'
        self._grad_image = None
        self._gradients_map_computed = False
        # optimal path
        self._optimal_path_image_name = 'optimal_path.png'
        self._optimal_path_computed = False
        # edge detection images
        self._gradient_magnitude_name = 'gradient_magnitude.png'
        self._smoothed_gradient_name = 'smoothed_gradient.png'
        self._weight_map_name = 'weight_map.png'
        self._edge_images_computed = False
        self._smoothed_map = None
        self._weight_map = None
        self._weight_map_computed = False
        self.current_edge_step = 0
        self.edge_steps = []
        self._edge_detection = False
        # edge contour tracing
        self.contour_mode = False
        self.contour_points = []
        self._weight_map_float = None  # Carte W en float pour les calculs
        self._contour_result_name = "contour_result.png"
        # starting and ending points
        self._starting_point = None
        self._ending_point = None
        self._starting_and_ending_points_set = False
        # observer
        self.obs = obs.Observer()
    
    # Properties for starting and ending points
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

    # Select button functionality
    def select_button_was_clicked(self) -> None:
        """Handles the button click event to open a file dialog and display the selected image."""
        file_name, _ = widgets.QFileDialog.getOpenFileName(self)
        self._vue.change_image(file_name)
        self._original_image_name = file_name
        self._original_image_grey_level = ui.GreyImage(self._original_image_name)
        self._distances_map_computed = False
        self._gradients_map_computed = False
        self._optimal_path_computed = False
        self._edge_images_computed = False
        self.erase_points_was_clicked()
        self._vue.ratio = max(gui.QPixmap(file_name).width()/1000, gui.QPixmap(file_name).height()/700)
        self._vue.texte.setText("Image has been selected. Select a starting point")
    
    # Erase points button functionality
    def erase_points_was_clicked(self) -> None:
        """Handles the button click event to erase the selected starting and ending points."""
        self._vue.image.ps = None
        self._vue.image.pe = None
        self._starting_point = None
        self._ending_point = None
        self._starting_and_ending_points_set = False
        self._distances_map_computed = False
        self._gradients_map_computed = False
        self._optimal_path_computed = False
        self._vue.image.update()
        self._vue.texte.setText("Select a starting point")

    # Original image button functionality
    def original_image_button_was_selected(self) -> None:
        """Handles the button click event to display the original image."""
        self._vue.print_stocked_image(self._original_image_name)
        self._vue.texte.setText("Original image is displayed")

    # Distances map button functionality
    def distances_map_creation(self, start: pc.Point, end: pc.Point) -> None:
        """Creates the distances map and stores it in the corresponding view."""
        self._vue.bar.reinitialise(start.norm(end))
        self._vue.bar.show()
        self.obs.add_observer(self._vue.bar)
        im = self._original_image_grey_level
        self._distances_costs = dijkstra.distances_costs(start, end, im, self._list_visited, self._edge_detection, obs=self.obs)
        distances_map_image = dijkstra.coloration_map(self._distances_costs, im)
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
            self._vue.texte.setText("Please select starting and ending points by clicking on the image.")
    
    # Gradients map button functionality
    def gradients_map_creation(self, start: pc.Point, end: pc.Point) -> None:
        """
        Creates the gradients map and stores it in the corresponding view.
        Using the functions from dijkstra.py
        """
        self.obs.add_observer(self._vue.bar)
        im = self._original_image_grey_level
        self._vue.bar.reinitialise(2*im.width*im.height)
        self._vue.bar.show()
        self._grad_image = dijkstra.gradient_on_image(self._distances_costs, im, self.obs)
        img = Image.fromarray(self._grad_image)
        img.save(self._gradients_map_image_name)
        self._gradients_map_computed = True
        self._vue.texte.setText("You can now print the optimal path")
        self._vue.bar.hide()
        self.obs.del_observer(self._vue.bar)
        
    def gradients_map_button_was_clicked(self) -> None:
        """Handles the button click event to display the gradients map."""
        if self._gradients_map_computed:
            self._vue.print_stocked_image(self._gradients_map_image_name)
        elif self._starting_and_ending_points_set:
            self.gradients_map_creation(self._starting_point, self._ending_point)
            self._vue.print_stocked_image(self._gradients_map_image_name)
        else:
            self._vue.texte.setText("Please select starting and ending points by clicking on the image.")
    
    # Path button functionality
    def path_button_was_clicked(self) -> None:
        """Handles the button click event to print the optimal path."""
        if self._optimal_path_computed:
            self._vue.print_stocked_image(self._optimal_path_image_name)
        elif self._starting_and_ending_points_set:
            im = self._original_image_grey_level
            descent = dijkstra.amelioration_descent(self._distances_costs, im, self._starting_point, self._ending_point, self._list_visited)
            final_img = dijkstra.affiche_descent(descent, self._grad_image)
            img = Image.fromarray(final_img)
            img.save(self._optimal_path_image_name)
            self._optimal_path_computed = True
            self._vue.print_stocked_image(self._optimal_path_image_name)
        else:
            self._vue.texte.setText("Please select starting and ending points by clicking on the image.")
    
    # Edge detection button functionality
    def edge_detection_button_was_clicked(self) -> None:
        """Handles the button click event to perform edge detection 
        and display the three images sequentially."""
        self._edge_detection = True
        im = self._original_image_grey_level

        if not self._edge_images_computed:
            print("Computing edge detection maps...")
            magnitude = edge.compute_gradient_magnitude(im)
            smoothed = edge.smooth_gradient_magnitude(magnitude, sigma=1.5)
            weight_map = edge.compute_edge_weight_map(smoothed, epsilon=0.1)
            self._weight_map_float = weight_map

            # Function to normalize and save images
            def normalize_and_save(array: np.ndarray, filename: str):
                if array.size == 0 or array.max() == 0:
                    norm = np.zeros(array.shape, dtype=np.uint8)
                else:
                    norm = (array / array.max() * 255).astype(np.uint8)
                img = Image.fromarray(norm)
                img.save(filename)
        
            def normalize_and_save_weight(array: np.ndarray, filename: str):
                norm = array.astype(np.float32)
                norm = np.log(norm + 0.01)    
                norm = (norm - norm.min()) / (norm.max() - norm.min() + 1e-8) * 255
                norm = norm.astype(np.uint8)
                img = Image.fromarray(norm)
                img.save(filename)
            
            normalize_and_save(magnitude, self._gradient_magnitude_name)
            normalize_and_save(smoothed, self._smoothed_gradient_name)
            normalize_and_save_weight(weight_map, self._weight_map_name)

            self._edge_images_computed = True

        # Prepare for displaying edge detection images
        self.current_edge_step = 0
        self.edge_steps = [
            (self._original_image_name, "Edge detection: 1. Original image"),
            (self._smoothed_gradient_name, "Edge detection: 2. Smoothed gradient (Gσ * |∇f|)"),
            (self._weight_map_name, "Edge detection: 3. Weight map W(x,y) = 1/(ε + smoothed)")
        ]

        # Show the next edge image button
        self.next_edge_button.show()

        # Launch the display of the first image
        self.show_next_edge_image()
    
    # Reset edge detection button functionality
    def reset_edge_detection(self) -> None:
        self._edge_detection = False
        self._edge_images_computed = False
        self._vue.texte.setText("Edge detection reset.")
        self._vue.print_stocked_image(self._original_image_name)
        self.next_edge_button.hide()
        self.contour_button.hide()
        self.contour_mode = False
        self.contour_points = []
        self.current_edge_step = 0
        self._weight_map_float = None
        self._starting_and_ending_points_set = False
        self.erase_points_was_clicked()

    # Next edge image button functionality
    def show_next_edge_image(self) -> None:
        """Prints the next image in the edge detection sequence."""
        if self.current_edge_step < len(self.edge_steps):
            filename, text = self.edge_steps[self.current_edge_step]
            self._vue.texte.setText(text)
            self._vue.print_stocked_image(filename)
            self.current_edge_step += 1
        else:
            # End of the sequence
            self._vue.texte.setText("Edge detection completed.\nYou can now select two points on a contour.")
            self._vue.print_stocked_image(self._original_image_name)
            self.next_edge_button.hide()
            self.contour_button.show()
            self.contour_mode = True
            self.contour_points = []
            self.current_edge_step = 0
    
    # Contour tracing button functionality
    def trace_contour(self) -> None:
        if len(self.contour_points) != 2:
            self._vue.texte.setText("Error: select exactly two points.")
            return

        start, goal = self.contour_points
        im = self._original_image_grey_level
        weight_map = self._weight_map_float

        list_visited = []
        # Distances calculation with edge detection
        dist_dict = dijkstra.distances_costs(
            start=start,
            end=None,
            grey_levels=im,
            list_visited=list_visited,
            edge_detection=True,
            weight_map=weight_map
        )
        # Gradient descent from goal to start
        path = self.reconstruct_path(dist_dict, goal, start)
        # Saving image with contour
        result_img = self.draw_contour(path, im)
        result_img.save(self._contour_result_name)
        self._vue.print_stocked_image(self._contour_result_name)
        self._vue.texte.setText(f"Detected edge ! Length : {len(path)} pixels")
        # Reset contour mode
        self.contour_mode = False
        self.contour_button.hide()
        self.next_edge_button.hide()

    def reconstruct_path(self, dist: ui.NumpyDict, current: pc.Point, start: pc.Point) -> list[pc.Point]:
        path = [current]
        while current != start and dist[current] > 0:
            neighbors = self._original_image_grey_level.neighbors(current)
            best = min(neighbors, key=lambda n: dist[n])
            if dist[best] >= dist[current]:
                break
            current = best
            path.append(current)
        path.reverse()
        return path

    def draw_contour(self, path: list[pc.Point], grey_img: ui.GreyImage) -> Image.Image:
        arr = grey_img.image
        # Convert to uint8 for RGB stacking
        arr_uint8 = arr.astype(np.uint8)
        rgb = np.stack([arr_uint8, arr_uint8, arr_uint8], axis=-1)  # shape (H, W, 3), uint8
        
        # red line drawing
        for p in path:
            for dx in [-2, -1, 0, 1, 2]:
                for dy in [-2, -1, 0, 1, 2]:
                    nx, ny = p.x + dx, p.y + dy
                    if 0 <= nx < grey_img.width and 0 <= ny < grey_img.height:
                        rgb[ny, nx] = [255, 0, 0]
        
        return Image.fromarray(rgb)
    
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


if __name__ == "__main__":
    application = widgets.QApplication(sys.argv)
    main_window = Window()
    main_window.showMaximized() # Full screen
    sys.exit(application.exec())