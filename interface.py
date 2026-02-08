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
import segmentation as seg
import pandas as pd

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

    def set_multiple(self, nb_steps : int) -> None:
        """Show the current step on the progress bar"""
        self.nb_steps = nb_steps
        self.step = 1
        self.setFormat(f"1/{nb_steps} : %p%  ")

    def change_step(self) -> None:
        """Change the current step on the progress bar"""
        self.step += 1
        self.setFormat(f"{self.step}/{self.nb_steps} : %p%  ")

    def set_single(self) -> None:
        """Set the progress bar to single mode"""
        self.setFormat("%p%")

    def update(self, value: int) -> None:
        """Updates the progress bar based on the current value."""
        if value < 0:
            self.change_step()
            self.reinitialise(-value)
        elif 100 - 100*value//self.maxi > self.value():
            self.setValue(int(100 - 100*value/self.maxi))
        super().update()

class Fenetre(widgets.QLabel):
    """A window class to display an image and handle mouse click events"""
    def __init__(self, parent: Vue) -> None:
        """Initializes the window with a parent view."""
        super().__init__(parent)
        self.ps = None
        self.pe = None
        self.points = []

    def mousePressEvent(self, event: gui.QMouseEvent) -> None:
        """Handles mouse press events to select starting and ending points on the image."""
        if self.underMouse():
            point = gui.QCursor.pos()
            point = self.mapFromGlobal(point)
            if point.x() < 0 or point.x() >= self.pixmap().width() or point.y() < 0 or point.y() >= self.pixmap().height():
                return
            point = pc.Point(int(self.parent().ratio*point.x()), int(self.parent().ratio*point.y()))
            
            if self.parent().menu._more_points_needed: #image segmentation mode
                p = self.mapFromGlobal(gui.QCursor.pos())
                self.points.append(p)
                self.parent()._menu._points_list.append(point)
                self.update()
                self.parent().texte.setText(f"<h1>{len(self.parent()._menu._points_list)} points chosen for segmentation. <br>When all points are chosen, click the 'All points chosen' button.</h1>")
                return
            
            elif self.parent()._menu.contour_mode and len(self.parent()._menu.contour_points)<2: # contour tracing mode
                self.parent()._menu.contour_points.append(point)
                if len(self.parent()._menu.contour_points) == 1:
                    self.ps = self.mapFromGlobal(gui.QCursor.pos())  # first point
                else:
                    self.pe = self.mapFromGlobal(gui.QCursor.pos())  # second point
                self.update()
                self.parent().texte.setText(f"<h1>{len(self.parent()._menu.contour_points)}/2 points chosen for the contour</h1>")
                return
            
            elif self.parent()._menu.shortest_mode and self.parent()._menu.starting_point is None: # first point
                self.ps = self.mapFromGlobal(gui.QCursor.pos())
                self.parent()._menu.starting_point = point
                self.parent().texte.setText("<h1>Select an ending point</h1>")

            elif self.parent()._menu.shortest_mode and self.parent()._menu.ending_point is None: # second point
                self.pe = self.mapFromGlobal(gui.QCursor.pos())
                self.parent()._menu.ending_point = point
                self.parent()._menu._starting_and_ending_points_set = True
                self.parent().texte.setText("<h1>Compute a distance map</h1>")
                self.parent()._menu.buttons["Distances map"].setVisible(True)
                self.parent()._menu.buttons["Sobel optimal path"].setVisible(True)

            elif self.parent()._menu.shortest_mode or self.parent()._menu.contour_mode or self.parent()._menu._more_points_needed: # both points are already set
                self.parent().texte.setText("<h1>Both starting and ending points are already set.</h1>")
            
            else:
                self.parent().texte.setText("<h1>Please select a mode first.</h1>")
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
        for p in self.points: # segmentation points
            painter.setBrush(gui.QBrush(gui.QColorConstants.Blue))
            painter.drawEllipse(p, 5, 5)
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
        self.texte = widgets.QLabel("<h1>Select a starting point</h1>", self)
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
        self.bar.setSizePolicy(widgets.QSizePolicy.Policy.Fixed, widgets.QSizePolicy.Policy.Fixed)
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
        buttons_info = pd.DataFrame.from_records([["Select an image", self.select_button_was_clicked, True],
                   ["Save the image", self.save_button_was_clicked,True],
                   ["Original image", self.original_image_button_was_clicked,True],
                   ["Shortest path", self.shortest_path_was_clicked,True],
                   ["Erase the points", self.erase_points_was_clicked,False],
                   ["Distances map", self.distances_map_button_was_clicked,False],
                   ["Gradients map", self.gradients_map_button_was_clicked,False],
                   ["Sobel optimal path", self.sobel_gradients_map_button_was_clicked,False],
                   ["Print the optimal path", self.path_button_was_clicked,False],
                   ["Edge detection", self.edge_detection_button_was_clicked,True],
                   ["Reset edge detection", self.reset_edge_detection_button_was_clicked,False],
                   ["Next image →", self.show_next_edge_image_button_was_clicked,False],
                   ["Draw the edge", self.contour_button_was_clicked_2,False],
                   ["Gradient Magnitude", self.gradient_magnitude_button_was_clicked,False],
                   ["Smoothed Gradient", self.smoothed_gradient_button_was_clicked,False],
                   ["Weight Map", self.weight_map_button_was_clicked,False],
                   ["Map with contour", self.print_contour_button_was_clicked,False],
                   ["New contour", self.new_contour_button_was_clicked,False],
                   ["Image segmentation", self.segmentation_button_was_clicked,True],
                   ["Reset segmentation", self.reset_segmentation_button_was_clicked,False],
                   ["All points chosen", self.all_points_chosen_button_was_clicked, False]], columns=["Label", "Function", "Enable"])
        
        self.buttons : dict[str, widgets.QPushButton] = dict()
        vertical = widgets.QVBoxLayout(self)
        for _, button in buttons_info.iterrows():
            self.buttons[button["Label"]] = widgets.QPushButton(button["Label"], self)
            self.buttons[button["Label"]].setMinimumHeight(30)
            vertical.addWidget(self.buttons[button["Label"]])
            self.buttons[button["Label"]].clicked.connect(button["Function"])
            self.buttons[button["Label"]].setVisible(button["Enable"])

        #Epsilon
        self.epsilon_label = widgets.QLabel("<h3>Epsilon :</h3>", self)
        vertical.addWidget(self.epsilon_label)
        self.epsilon_spin_box = widgets.QDoubleSpinBox(self)
        vertical.addWidget(self.epsilon_spin_box)
        self.epsilon_spin_box.setValue(2.)
        self.epsilon_spin_box.valueChanged.connect(slot=lambda d: (dijkstra.__setattr__("epsilon", d)))
        vertical.addStretch(0)

        # vue
        self._vue = vue
        self.shortest_mode = False

        # starting image
        self._original_image_name = 'images/Carte.png'
        self._original_image_grey_level = ui.GreyImage(self._original_image_name)
        self._distances_map_image_name = 'results/distances_map.png'
        # distances map
        self._list_visited = []
        self._distances_map_computed = False
        self._distances_costs = None
        # gradients map
        self._gradients_map_image_name = 'results/gradients_map.png'
        self._grad_image = None
        self._gradients_map_computed = False
        # Sobel gradients map
        self._sobel_gradients_map_image_name = 'results/sobel_gradients_map.png'
        self._sobel_grad_image = None
        self._sobel_gradients_map_computed = False
        # optimal path
        self._optimal_path_image_name = 'results/optimal_path.png'
        self._optimal_path_computed = False

        # edge detection images
        self._gradient_magnitude_name = 'results/gradient_magnitude.png'
        self._smoothed_gradient_name = 'results/smoothed_gradient.png'
        self._weight_map_name = 'results/weight_map.png'
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
        self._weight_map_float = None  # Map of floats for edge detection
        self._contour_result_name = "results/contour_result.png"

        # image segmentation
        self._segmentation_image_name = 'results/segmentation.png'
        self._more_points_needed = False
        self._segmentation_computed = False
        self._points_list = []

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
        print(file_name)
        if file_name != "":
            self._vue.change_image(file_name)
            self._original_image_name = file_name
            self._original_image_grey_level = ui.GreyImage(self._original_image_name)
            # Reset all computed maps and states
            self._distances_map_computed = False
            self._gradients_map_computed = False
            self._optimal_path_computed = False
            self._edge_images_computed = False
            self._sobel_gradients_map_computed = False
            self._segmentation_computed = False
            self.erase_points_was_clicked()
            # Update ratio for point selection
            self._vue.ratio = max(gui.QPixmap(file_name).width()/1000, gui.QPixmap(file_name).height()/700)
            self._vue.texte.setText("<h1>Image has been selected. Select a starting point</h1>")
            self.buttons["Distances map"].setVisible(False)
            self.buttons["Gradients map"].setVisible(False)
            self.buttons["Sobel optimal path"].setVisible(False)
            self.buttons["Print the optimal path"].setVisible(False)
            # Reset edge detection and segmentation states
            self.reset_edge_detection_button_was_clicked()
            self.reset_segmentation_button_was_clicked()

    # Save button functionality
    def save_button_was_clicked(self) -> None:
        """Handles the button click event to open a file dialog and save the current image."""
        file_name, _ = widgets.QFileDialog.getSaveFileName(self, filter="Images (*.png *.xpm *.jpg)")
        if file_name != "": # If a file name was provided
            self._vue.image.pixmap().save(file_name)

    # Shortest path button functionality
    def shortest_path_was_clicked(self) -> None:
        """Handles the button click event to enable shortest path mode."""
        self.shortest_mode = True
        self.buttons["Erase the points"].setVisible(True)
        self.buttons["Edge detection"].setVisible(False)
        self.buttons["Image segmentation"].setVisible(False)

    # Erase points button functionality
    def erase_points_was_clicked(self) -> None:
        """Handles the button click event to erase the selected starting and ending points."""
        # Reset points in the image view
        self._vue.image.ps = None
        self._vue.image.pe = None
        self._vue.image.points = []
        self._starting_point = None
        self._ending_point = None
        self._starting_and_ending_points_set = False
        # Reset computed maps
        self._distances_map_computed = False
        self._gradients_map_computed = False
        self._optimal_path_computed = False
        self._sobel_gradients_map_computed = False
        self._vue.image.update()
        self._vue.texte.setText("<h1>Select a starting point</h1>")
        self.buttons["Distances map"].setVisible(False)
        self.buttons["Gradients map"].setVisible(False)
        self.buttons["Sobel optimal path"].setVisible(False)
        self.buttons["Print the optimal path"].setVisible(False)
        self.contour_points = list()
        self._vue.print_stocked_image(self._original_image_name)

    # Original image button functionality
    def original_image_button_was_clicked(self) -> None:
        """Handles the button click event to display the original image."""
        self._vue.print_stocked_image(self._original_image_name)
        self._vue.texte.setText("<h1>Original image is displayed</h1>")
        self.buttons["Shortest path"].setVisible(True)
        self.buttons["Edge detection"].setVisible(True)
        self.buttons["Image segmentation"].setVisible(True)

    # Distances map button functionality
    def distances_map_creation(self, start: pc.Point, end: pc.Point) -> None:
        """Creates the distances map and stores it in the corresponding view."""
        print(dijkstra.epsilon)
        self._vue.bar.reinitialise(start.norm(end))
        self._vue.bar.show()
        self.obs.add_observer(self._vue.bar)
        im = self._original_image_grey_level
        # Compute distances costs using fast marching algorithm
        self._distances_costs = dijkstra.distances_costs(start, end, im, self._list_visited, obs=self.obs)
        distances_map_image = dijkstra.coloration_map(self._distances_costs, im)
        img = Image.fromarray(distances_map_image)
        img.save(self._distances_map_image_name)
        self._distances_map_computed = True
        self._vue.texte.setText("<h1>Compute a gradient map</h1>")
        self._vue.bar.hide()
        self.obs.del_observer(self._vue.bar)

    def distances_map_button_was_clicked(self) -> None:
        """Handles the button click event to display the distances map."""
        if self._distances_map_computed: # If already computed, just display it
            self._vue.print_stocked_image(self._distances_map_image_name)
            self.buttons["Gradients map"].setVisible(True)
        elif self._starting_and_ending_points_set: # If points are set, compute and display
            self.distances_map_creation(self._starting_point, self._ending_point)
            self._vue.print_stocked_image(self._distances_map_image_name)
            self.buttons["Gradients map"].setVisible(True)
        else: # Points not set
            print("Please select starting and ending points by clicking on the image.")
    
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
        # Compute gradient on image
        self._grad_image = dijkstra.gradient_on_image(self._distances_costs, im, self.obs)
        img = Image.fromarray(self._grad_image)
        img.save(self._gradients_map_image_name)
        self._gradients_map_computed = True
        self._vue.texte.setText("<h1>You can now print the optimal path</h1>")
        self._vue.bar.hide()
        self.obs.del_observer(self._vue.bar)
        
        
    def gradients_map_button_was_clicked(self) -> None:
        """Handles the button click event to display the gradients map."""
        if self._gradients_map_computed: # If already computed, just display it
            self._vue.print_stocked_image(self._gradients_map_image_name)
            self.buttons["Print the optimal path"].setVisible(True)
        elif self._starting_and_ending_points_set: # If points are set, compute and display
            self.gradients_map_creation(self._starting_point, self._ending_point)
            self._vue.print_stocked_image(self._gradients_map_image_name)
            self.buttons["Print the optimal path"].setVisible(True)
        else: # Points not set
            self._vue.texte.setText("<h1>Please select starting and ending points by clicking on the image.</h1>")
    
    def sobel_gradients_map_button_was_clicked(self) -> None:
        """Handles the button click event to display the Sobel gradients map."""
        if self._sobel_gradients_map_computed:
            self._vue.print_stocked_image(self._sobel_gradients_map_image_name)
            return
        self.obs.add_observer(self._vue.bar)
        im = self._original_image_grey_level
        self._vue.bar.reinitialise(self._starting_point.norm(self._ending_point))
        self._vue.bar.show()
        # Compute Sobel gradient descent
        sobel_grad = dijkstra.gradient_descent_Sobel(im, self._starting_point, self._ending_point, self.obs)
        img = dijkstra.affiche_descent_image(sobel_grad, im, Sobel=1, first_time=0)
        img_pil = Image.fromarray(img.astype(np.uint8))
        img_pil.save(self._sobel_gradients_map_image_name)
        self._sobel_gradients_map_computed = True
        self._vue.print_stocked_image(self._sobel_gradients_map_image_name)
        self._vue.texte.setText("<h1>Sobel gradients optimal path is displayed.</h1>")
        self._vue.bar.hide()
        self.obs.del_observer(self._vue.bar)

    # Path button functionality
    def path_button_was_clicked(self) -> None:
        """Handles the button click event to print the optimal path."""
        if self._optimal_path_computed: # If already computed, just display it
            self._vue.print_stocked_image(self._optimal_path_image_name)
        elif self._starting_and_ending_points_set: # If points are set, compute and display
            im = self._original_image_grey_level
            # Gradient descent from ending point to starting point
            descent = dijkstra.amelioration_descent(self._distances_costs, im, self._starting_point, self._ending_point, self._list_visited)
            final_img = dijkstra.affiche_descent_image(descent, im, Sobel=0, first_time=0)
            img = Image.fromarray(final_img)
            img.save(self._optimal_path_image_name)
            self._optimal_path_computed = True
            self._vue.print_stocked_image(self._optimal_path_image_name)
        else: # Points not set
            self._vue.texte.setText("<h1>Please select starting and ending points by clicking on the image.</h1>")
    
    # Edge detection button functionality
    def edge_detection_button_was_clicked(self) -> None:
        """Handles the button click event to perform edge detection 
        and display the three images sequentially."""
        self._edge_detection = True
        im = self._original_image_grey_level
        # Disable other buttons during edge detection
        self.buttons["Shortest path"].setVisible(False)
        self.buttons["Distances map"].setVisible(False)
        self.buttons["Gradients map"].setVisible(False)
        self.buttons["Print the optimal path"].setVisible(False)
        self.buttons["Reset edge detection"].setVisible(True)
        self.erase_points_was_clicked()

        if not self._edge_images_computed: # Compute edge detection images if not already done
            print("Computing edge detection maps...")
            self.obs.add_observer(self._vue.bar)
            self._vue.bar.reinitialise(im.width*im.height)
            self._vue.bar.show()
            # Compute gradient magnitude, smoothed gradient, and weight map
            magnitude = edge.compute_gradient_magnitude(im, self.obs)
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
            self._vue.bar.hide()
            self.obs.del_observer(self._vue.bar)

        # Prepare for displaying edge detection images
        self.current_edge_step = 0
        self.edge_steps = [
            (self._original_image_name, "<h1>Edge detection: 1. Original image</h1>"),
            (self._smoothed_gradient_name, "<h1>Edge detection: 2. Smoothed gradient (Gσ * |∇f|)</h1>"),
            (self._weight_map_name, "<h1>Edge detection: 3. Weight map W(x,y) = 1/(ε + smoothed)</h1>")
        ]

        # Show the next edge image button and start the sequence
        self.buttons["Next image →"].setVisible(True)
        self.show_next_edge_image_button_was_clicked()
    
    # Reset edge detection button functionality
    def reset_edge_detection_button_was_clicked(self) -> None:
        """Handles the button click event to reset edge detection state."""
        # Reset edge detection related states
        self._edge_detection = False
        self._edge_images_computed = False
        self._vue.texte.setText("<h1>Edge detection reset.</h1>")
        self._vue.print_stocked_image(self._original_image_name)
        self.contour_mode = False
        self.contour_points = []
        self.current_edge_step = 0
        self._weight_map_float = None
        self._starting_and_ending_points_set = False
        self.erase_points_was_clicked()
        self.buttons["Reset edge detection"].setVisible(False)
        # Hide edge detection related buttons
        self.buttons["Next image →"].setVisible(False)
        self.buttons["Draw the edge"].setVisible(False)
        self.buttons["Gradient Magnitude"].setVisible(False)
        self.buttons["Smoothed Gradient"].setVisible(False)
        self.buttons["Weight Map"].setVisible(False)
        self.buttons["Map with contour"].setVisible(False)
        self.buttons["New contour"].setVisible(False)

    # Next edge image button functionality
    def show_next_edge_image_button_was_clicked(self) -> None:
        """Prints the next image in the edge detection sequence."""
        if self.current_edge_step < len(self.edge_steps):
            filename, text = self.edge_steps[self.current_edge_step]
            self._vue.texte.setText(text)
            self._vue.print_stocked_image(filename)
            self.current_edge_step += 1
        else:
            # End of the sequence
            self._vue.texte.setText("<h1>Edge detection completed.<br>You can now select two points on a contour.</h1>")
            self._vue.print_stocked_image(self._original_image_name)
            self.buttons["Next image →"].setVisible(False)
            self.buttons["Draw the edge"].setVisible(True)
            self.contour_mode = True
            self.contour_points = []
            self.current_edge_step = 0
    
    # Contour tracing button functionality
    def contour_button_was_clicked(self) -> None:
        """Handles the button click event to trace an optimal path between two selected points."""
        if len(self.contour_points) != 2: # Need exactly two points
            self._vue.texte.setText("<h1>Error: select exactly two points.</h1>")
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
        self._vue.texte.setText(f"<h1>Detected edge ! Length : {len(path)} pixels</h1>")
        # Reset contour mode
        self.contour_mode = False
        self.buttons["Draw the edge"].setVisible(False)
        # Activate visualization of edge detection images again
        self.buttons["Gradient Magnitude"].setVisible(True)
        self.buttons["Smoothed Gradient"].setVisible(True)
        self.buttons["Weight Map"].setVisible(True)
        self.buttons["Map with contour"].setVisible(True)
    
    def contour_button_was_clicked_2(self) -> None:
        """ Handles the button click event to trace a whole
            contour between two selected points."""
        if len(self.contour_points) < 2: # Need exactly two points
            self._vue.texte.setText("<h1>Error: select exactly two points.</h1>")
            return
        
        # Get the two selected contour points
        start, goal = self.contour_points
        im = self._original_image_grey_level
        weight_map = self._weight_map_float
        
        list_visited_start = []
        list_visited_end = []
        
        self.obs.add_observer(self._vue.bar)
        im = self._original_image_grey_level
        self._vue.bar.reinitialise(im.width*im.height)
        self._vue.bar.set_multiple(2)
        self._vue.bar.show()
        
        # Compute distance dictionaries from both start and goal points
        dist_dict_start = dijkstra.distances_costs(
            start=start,
            end=None,
            grey_levels=im,
            list_visited=list_visited_start,
            edge_detection=True,
            weight_map=weight_map,
            obs=self.obs
        )
        self.obs.notify_observer(-im.width*im.height)
        dist_dict_end = dijkstra.distances_costs(
            start=goal,
            end=None,
            grey_levels=im,
            list_visited=list_visited_end,
            edge_detection=True,
            weight_map=weight_map,
            obs=self.obs
        )
        self._vue.bar.hide()
        self._vue.bar.set_single()
        self.obs.del_observer(self._vue.bar)
        
        # Find points that are equidistant from start and goal
        difference_dict = dist_dict_start - dist_dict_end
        tolerance = 1e-3
        equidistance_points = [p for p in difference_dict if abs(difference_dict[p]) < tolerance]
        
        point1 = min(equidistance_points, key=lambda p : dist_dict_start[p] + 10e6*np.sign((p - start).vect(goal - start)))

        point2 = min(equidistance_points, key=lambda p : dist_dict_start[p] - 10e6*np.sign((p - start).vect(goal - start)))
        
        # Chemin 1 : start → point1
        path_start_to_p1 = self.reconstruct_path(dist_dict_start, point1, start)
        
        # Path 2 : point1 → goal
        path_p1_to_goal = self.reconstruct_path(dist_dict_end, point1, goal)
        
        # Path 3 : goal → point2
        path_goal_to_p2 = self.reconstruct_path(dist_dict_end, point2, goal)
        
        # Path 4 : point2 → start
        path_p2_to_start = self.reconstruct_path(dist_dict_start, point2, start)
        
        complete_contour = (path_start_to_p1 +           # start → point1
                        path_p1_to_goal[::-1] +       # point1 → goal
                        path_goal_to_p2 +             # goal → point2
                        path_p2_to_start[::-1])       # point2 → start
        
        result_img = self.draw_contour(complete_contour, im)
        result_img.save(self._contour_result_name)
        self._vue.print_stocked_image(self._contour_result_name)
        self._vue.texte.setText(f"<h1>Detected edge ! Length : {len(complete_contour)} pixels</h1>")
        
        # Reset contour mode
        self.contour_mode = False
        # self.contour_button.hide()
        self.buttons["Gradient Magnitude"].setVisible(True)
        self.buttons["Smoothed Gradient"].setVisible(True)
        self.buttons["Weight Map"].setVisible(True)
        self.buttons["Map with contour"].setVisible(True)
        self.buttons["New contour"].setVisible(True)

    def reconstruct_path(self, dist: ui.NumpyDict, current: pc.Point, start: pc.Point) -> list[pc.Point]:
        """Reconstructs the path from the distance dictionary."""
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
        """Draws the contour path on the grey level image."""
        rgb = grey_img.image.astype(np.uint8)# shape (H, W, 3), uint8
        # red line drawing
        for p in path:
            for dx in [-2, -1, 0, 1, 2]:
                for dy in [-2, -1, 0, 1, 2]:
                    nx, ny = p.x + dx, p.y + dy
                    if 0 <= nx < grey_img.width and 0 <= ny < grey_img.height:
                        rgb[ny, nx] = [255, 0, 0]
        return Image.fromarray(rgb)
    
    # Gradient magnitude button functionality
    def gradient_magnitude_button_was_clicked(self) -> None:
        """Handles the button click event to display the gradient magnitude image."""
        self._vue.print_stocked_image(self._gradient_magnitude_name)
        self._vue.texte.setText("<h1>Gradient Magnitude image is displayed.</h1>")
    
    # Smoothed gradient button functionality
    def smoothed_gradient_button_was_clicked(self) -> None:
        """Handles the button click event to display the smoothed gradient image."""
        self._vue.print_stocked_image(self._smoothed_gradient_name)
        self._vue.texte.setText("<h1>Smoothed Gradient image is displayed.</h1>")
    
    # Weight map button functionality
    def weight_map_button_was_clicked(self) -> None:
        """Handles the button click event to display the weight map image."""
        self._vue.print_stocked_image(self._weight_map_name)
        self._vue.texte.setText("<h1>Weight Map image is displayed.</h1>")
    
    # Contour button functionality
    def print_contour_button_was_clicked(self) -> None:
        """Handles the button click event to display the contour image."""
        self._vue.print_stocked_image(self._contour_result_name)
        self._vue.texte.setText("<h1>Contour image is displayed.</h1>")
    
    # New contour button functionality
    def new_contour_button_was_clicked(self) -> None:
        """Handles the button click event to reset the contour tracing."""
        self.contour_mode = True
        self.contour_points = []
        self._vue.texte.setText("<h1>Select two points on a contour to trace it.</h1>")
        self._edge_images_computed = False
        self._vue.print_stocked_image(self._original_image_name)
        self._starting_and_ending_points_set = False
        self.erase_points_was_clicked()
    
    # Image segmentation button functionality
    def segmentation_button_was_clicked(self) -> None:
        """Handles the button click event to perform image segmentation."""
        self._more_points_needed = True
        self._vue.texte.setText("<h1>Please choose segmentation points by clicking on the image. <br> When all points are chosen, click the 'All points chosen' button.")
        self.buttons["All points chosen"].setVisible(True)
        self.buttons["Edge detection"].setVisible(False)
        self.buttons["Shortest path"].setVisible(False)
        self.buttons["Reset segmentation"].setVisible(True)
    
    # All points chosen button functionality
    def all_points_chosen_button_was_clicked(self) -> None:
        """Handles the button click event when all segmentation points are chosen."""
        if len(self._points_list) < 2: # Need at least two points
            print("Please choose at least two points for segmentation.")
            return
        
        # Observer for progress bar
        self.obs.add_observer(self._vue.bar)
        im = self._original_image_grey_level
        self._vue.bar.reinitialise(len(self._points_list)+1)
        self._vue.bar.set_multiple(2)
        self._vue.bar.show()
        # Compute segmentation
        self._more_points_needed = False
        print(f"Computing segmentation with {len(self._points_list)} points.")
        # Compute distance maps and segmentation
        list_distance_map, _ = seg.distances_map(self._points_list, im, self.obs)
        segmentation = seg.choice_segmentation_v1(self._points_list, list_distance_map, im, self.obs)
        img = ui.Image.fromarray(segmentation, 'RGB')
        img.save(self._segmentation_image_name)
        self._vue.print_stocked_image(self._segmentation_image_name)
        self._vue.texte.setText("<h1>Image segmentation completed.</h1>")
        self._points_list = []
        self._vue.bar.hide()
        self._vue.bar.set_single()
        self.obs.del_observer(self._vue.bar)

    # Reset segmentation button functionality
    def reset_segmentation_button_was_clicked(self) -> None:
        """Handles the button click event to reset the image segmentation."""
        self._segmentation_computed = False
        self._vue.print_stocked_image(self._original_image_name)
        self._vue.texte.setText("<h1>Segmentation reset. Select a starting point.</h1>")
        self.erase_points_was_clicked()
        self._more_points_needed = False
        self.buttons["All points chosen"].setVisible(False)
        self._points_list = []
        self.buttons["Reset segmentation"].setVisible(False)
            
    
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