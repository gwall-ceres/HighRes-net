import sys
import os
import json
import logging
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap  # Directly import QPixmap
import numpy as np
from scipy.ndimage import shift as ndi_shift
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress excessive matplotlib font manager logs
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

# Configuration loading function
DEFAULT_CONFIG = {
    "reference_image": "",
    "reference_mask": "",
    "template_image": "",
    "template_mask": "",
    "current_deltax": 0.0,
    "current_deltay": 0.0,
    "shift_step_x": 5.0,  # Example default shift steps
    "shift_step_y": 5.0
}

def load_config(config_path='config.json'):
    if not os.path.exists(config_path):
        logging.warning(f"Config file '{config_path}' not found. Using default settings.")
        return DEFAULT_CONFIG.copy()
    
    with open(config_path, 'r') as f:
        try:
            user_config = json.load(f)
            logging.info(f"Loaded configuration from {config_path}.")
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON: {e}. Using default settings.")
            return DEFAULT_CONFIG.copy()
    
    # Merge user_config with DEFAULT_CONFIG, preferring user_config values
    config = DEFAULT_CONFIG.copy()
    config.update({k: v for k, v in user_config.items() if k in DEFAULT_CONFIG})
    
    # Optionally, warn about any unknown fields
    unknown_fields = set(user_config.keys()) - set(DEFAULT_CONFIG.keys())
    if unknown_fields:
        logging.warning(f"Unknown config fields detected and ignored: {unknown_fields}")
    
    return config

def contrast_stretch(array):
    """
    Perform contrast stretching on a NumPy array to map its values to the 0-255 range.
    """
    logging.debug(f"Original array min: {array.min()}, max: {array.max()}")
    array = array.astype(float)
    min_val = np.min(array)
    max_val = np.max(array)

    # Avoid division by zero
    if max_val - min_val == 0:
        logging.warning("Max and min values are the same. Returning a zero array.")
        return np.zeros_like(array, dtype=np.uint8)

    # Perform contrast stretching
    stretched = (array - min_val) / (max_val - min_val) * 255.0
    logging.debug(f"Stretched array min: {stretched.min()}, max: {stretched.max()}")

    # Clip values to the 0-255 range and convert to uint8
    stretched = np.clip(stretched, 0, 255).astype(np.uint8)

    logging.debug(f"Final stretched array min: {stretched.min()}, max: {stretched.max()}")
    return stretched

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        
        # ----- Load Configuration -----
        self.config = load_config()
        logging.debug("Configuration loaded.")
        
        # ----- Menu Bar -----
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        
        load_ref_action = QtWidgets.QAction("Load Reference Image", self)
        load_ref_action.triggered.connect(lambda: self.load_image("reference_image"))
        file_menu.addAction(load_ref_action)
        
        load_ref_mask_action = QtWidgets.QAction("Load Reference Mask", self)
        load_ref_mask_action.triggered.connect(lambda: self.load_image("reference_mask"))
        file_menu.addAction(load_ref_mask_action)
        
        load_template_action = QtWidgets.QAction("Load Template Image", self)
        load_template_action.triggered.connect(lambda: self.load_image("template_image"))
        file_menu.addAction(load_template_action)
        
        load_template_mask_action = QtWidgets.QAction("Load Template Mask", self)
        load_template_mask_action.triggered.connect(lambda: self.load_image("template_mask"))
        file_menu.addAction(load_template_mask_action)
        
        # ----- Central Widget and Layout -----
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        
        # ----- Controls for current deltaX, deltaY -----
        current_shift_layout = QtWidgets.QHBoxLayout()
        self.deltaX_edit = QtWidgets.QLineEdit()
        self.deltaY_edit = QtWidgets.QLineEdit()
        
        self.deltaX_edit.setPlaceholderText("Current Delta X")
        self.deltaY_edit.setPlaceholderText("Current Delta Y")
        
        # Populate with config values
        self.deltaX_edit.setText(str(self.config["current_deltax"]))
        self.deltaY_edit.setText(str(self.config["current_deltay"]))
        
        # Connect signals for updating shifts
        self.deltaX_edit.editingFinished.connect(self.set_shift_x)
        self.deltaY_edit.editingFinished.connect(self.set_shift_y)
        
        current_shift_layout.addWidget(QtWidgets.QLabel("Current ΔX:"))
        current_shift_layout.addWidget(self.deltaX_edit)
        current_shift_layout.addWidget(QtWidgets.QLabel("Current ΔY:"))
        current_shift_layout.addWidget(self.deltaY_edit)
        
        main_layout.addLayout(current_shift_layout)
        
        # ----- Controls for shift steps -----
        shift_step_layout = QtWidgets.QHBoxLayout()
        self.shift_step_x_edit = QtWidgets.QLineEdit()
        self.shift_step_y_edit = QtWidgets.QLineEdit()
        
        self.shift_step_x_edit.setPlaceholderText("Shift Step X")
        self.shift_step_y_edit.setPlaceholderText("Shift Step Y")
        
        # Populate with config values
        self.shift_step_x_edit.setText(str(self.config["shift_step_x"]))
        self.shift_step_y_edit.setText(str(self.config["shift_step_y"]))
        
        shift_step_layout.addWidget(QtWidgets.QLabel("Shift Step X:"))
        shift_step_layout.addWidget(self.shift_step_x_edit)
        shift_step_layout.addWidget(QtWidgets.QLabel("Shift Step Y:"))
        shift_step_layout.addWidget(self.shift_step_y_edit)
        
        main_layout.addLayout(shift_step_layout)
        
        # ----- Image Display Layout -----
        images_layout = QtWidgets.QHBoxLayout()
        
        # Placeholder for reference+template overlay image
        self.overlay_image_label = QtWidgets.QLabel("Overlay Image Here")
        self.overlay_image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.overlay_image_label.setStyleSheet("border: 1px solid gray;")
        
        # Placeholder for masked difference image
        self.diff_image_label = QtWidgets.QLabel("Masked Difference Image Here")
        self.diff_image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.diff_image_label.setStyleSheet("border: 1px solid gray;")
        
        images_layout.addWidget(self.overlay_image_label)
        images_layout.addWidget(self.diff_image_label)
        
        main_layout.addLayout(images_layout)
        
        # ----- Graphs Layout -----
        graphs_layout = QtWidgets.QHBoxLayout()
        
        # Initialize MSE Plot
        self.mse_fig = Figure(figsize=(4, 2))
        self.mse_canvas = FigureCanvas(self.mse_fig)
        self.mse_ax = self.mse_fig.add_subplot(111)
        self.mse_ax.set_title("MSE over Shifts")
        self.mse_ax.set_xlabel("Shift Steps")
        self.mse_ax.set_ylabel("MSE")
        graphs_layout.addWidget(self.mse_canvas)
        
        # Initialize Perceptual Loss Plot
        self.pl_fig = Figure(figsize=(4, 2))
        self.pl_canvas = FigureCanvas(self.pl_fig)
        self.pl_ax = self.pl_fig.add_subplot(111)
        self.pl_ax.set_title("Perceptual Loss over Shifts")
        self.pl_ax.set_xlabel("Shift Steps")
        self.pl_ax.set_ylabel("Perceptual Loss")
        graphs_layout.addWidget(self.pl_canvas)
        
        main_layout.addLayout(graphs_layout)
        
        # ----- Set Window Properties -----
        self.setWindowTitle("Interactive Image Alignment Tool")
        self.resize(850, 1000)  # Adjust as needed to accommodate image and graph sizes
        
        # ----- Focus Policy for Keyboard Events -----
        self.setFocusPolicy(Qt.StrongFocus)  # Now works because Qt is imported
        
        # ----- Initialize Image Arrays and Pixmaps -----
        self.ref_image_array = None        # Original Reference Image (Grayscale)
        self.ref_display_pixmap = None     # Display Reference Image (Contrast Stretched)
        
        self.template_image_array = None   # Original Template Image (Grayscale)
        self.template_display_pixmap = None  # Display Template Image (Contrast Stretched)
        
        self.shifted_template_image_array = None  # Shifted Template Image
        
        self.ref_mask_array = None         # Original Reference Mask (Grayscale)
        self.ref_mask_pixmap = None
        
        self.template_mask_array = None    # Original Template Mask (Grayscale)
        self.template_mask_pixmap = None
        
        # ----- Initialize Loss Histories -----
        self.mse_history = []
        self.pl_history = []
        
        # ----- Initialize Plots -----
        self.initialize_plots()
        
        # ----- Automatically Load Images from Config -----
        if self.config["reference_image"]:
            self.load_image_from_path("reference_image", self.config["reference_image"])
        if self.config["reference_mask"]:
            self.load_image_from_path("reference_mask", self.config["reference_mask"])
        if self.config["template_image"]:
            self.load_image_from_path("template_image", self.config["template_image"])
        if self.config["template_mask"]:
            self.load_image_from_path("template_mask", self.config["template_mask"])

    def initialize_plots(self):
        """
        Initialize the plots with empty data.
        """
        self.mse_ax.clear()
        self.mse_ax.set_title("MSE over Shifts")
        self.mse_ax.set_xlabel("Shift Steps")
        self.mse_ax.set_ylabel("MSE")
        self.mse_ax.plot([], [], 'r-')
        self.mse_canvas.draw()

        self.pl_ax.clear()
        self.pl_ax.set_title("Perceptual Loss over Shifts")
        self.pl_ax.set_xlabel("Shift Steps")
        self.pl_ax.set_ylabel("Perceptual Loss")
        self.pl_ax.plot([], [], 'b-')
        self.pl_canvas.draw()
        logging.debug("Initialized MSE and Perceptual Loss plots.")
    
    def load_image(self, image_type):
        """
        Open a file dialog to load an image or mask.
        """
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 
            f"Select {image_type.replace('_', ' ').title()}", 
            "", 
            "Image Files (*.png *.jpg *.bmp)"
        )
        if fname:
            self.load_image_from_path(image_type, fname)
    
    def load_image_from_path(self, image_type, filepath):
        """
        Load an image or mask from the given filepath, store the original NumPy array,
        and update the display pixmap with contrast stretching if it's an image (not a mask).
        """
        # Load image using QImage
        qimage = QtGui.QImage(filepath)
        if qimage.isNull():
            QtWidgets.QMessageBox.warning(self, "Load Image", f"Failed to load {image_type} from {filepath}.")
            logging.error(f"Failed to load {image_type} from {filepath}.")
            return

        logging.debug(f"Loaded {image_type} with format: {qimage.format()} and depth: {qimage.depth()}")

        # Ensure the image is grayscale
        if not qimage.isGrayscale():
            QtWidgets.QMessageBox.warning(self, "Load Image", f"The {image_type} must be a grayscale image.")
            logging.error(f"The {image_type} is not a grayscale image.")
            return

        # Convert QImage to Grayscale8 format
        qimage = qimage.convertToFormat(QtGui.QImage.Format_Grayscale8)
        logging.debug(f"Converted {image_type} to Grayscale8 format.")

        width = qimage.width()
        height = qimage.height()
        ptr = qimage.bits()
        ptr.setsize(qimage.byteCount())
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((height, width))
        logging.debug(f"{image_type} NumPy array shape: {arr.shape}, dtype: {arr.dtype}")

        # Assign to the correct attribute based on image type
        if image_type == "reference_image":
            self.ref_image_array = arr.copy()  # Store original
            display_arr = contrast_stretch(self.ref_image_array)  # Apply contrast stretching for display
            self.ref_display_pixmap = self.array_to_qpixmap(display_arr, is_grayscale=True)
        elif image_type == "template_image":
            self.template_image_array = arr.copy()  # Store original
            self.shifted_template_image_array = self.template_image_array.copy()  # Initialize shifted image
            display_arr = contrast_stretch(self.template_image_array)  # Apply contrast stretching for display
            self.template_display_pixmap = self.array_to_qpixmap(display_arr, is_grayscale=True)
        elif image_type == "reference_mask":
            self.ref_mask_array = arr.copy()  # Store original mask without contrast stretching
            self.ref_mask_pixmap = self.array_to_qpixmap(self.ref_mask_array, is_grayscale=True)
        elif image_type == "template_mask":
            self.template_mask_array = arr.copy()  # Store original mask without contrast stretching
            self.template_mask_pixmap = self.array_to_qpixmap(self.template_mask_array, is_grayscale=True)
        else:
            logging.warning(f"Unknown image type: {image_type}")
            return

        logging.debug(f"Stored {image_type} data.")

        # Check for dimension consistency
        if image_type in ["reference_image", "template_image"]:
            # After loading the reference or template image
            if image_type == "template_image" and self.ref_image_array is not None:
                if self.template_image_array.shape != self.ref_image_array.shape:
                    QtWidgets.QMessageBox.critical(self, "Dimension Mismatch",
                                                   "Template image dimensions do not match Reference image dimensions.")
                    logging.error("Template image dimensions do not match Reference image dimensions.")
                    return
        elif image_type in ["reference_mask", "template_mask"]:
            # After loading the masks
            if self.ref_mask_array is not None and self.template_mask_array is not None:
                if self.ref_mask_array.shape != self.template_mask_array.shape:
                    QtWidgets.QMessageBox.critical(self, "Dimension Mismatch",
                                                   "Reference mask dimensions do not match Template mask dimensions.")
                    logging.error("Reference mask dimensions do not match Template mask dimensions.")
                    return

        self.update_overlay()
    
    def array_to_qpixmap(self, array, is_grayscale):
        """
        Convert a NumPy array to QPixmap for display.
        
        Parameters:
            array (np.ndarray): The image array.
            is_grayscale (bool): Flag indicating if the image is grayscale.
        
        Returns:
            QPixmap: The resulting pixmap.
        """
        try:
            if is_grayscale:
                height, width = array.shape
                bytes_per_line = width
                qimage = QtGui.QImage(array.tobytes(), width, height, bytes_per_line, QtGui.QImage.Format_Grayscale8)
            else:
                # This block can be omitted if all images are guaranteed to be grayscale
                height, width, channels = array.shape
                bytes_per_line = 3 * width
                qimage = QtGui.QImage(array.tobytes(), width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
            
            if qimage.isNull():
                raise ValueError("QImage conversion resulted in a null image.")
            
            pixmap = QPixmap.fromImage(qimage)
            logging.debug("Converted NumPy array to QPixmap.")
            
            # Optionally, scale the pixmap to fit the display label
            scaled_pixmap = pixmap.scaled(self.overlay_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logging.debug("Scaled QPixmap to fit QLabel size.")
            
            return scaled_pixmap
        except Exception as e:
            logging.error(f"Failed to convert array to QPixmap: {e}")
            QtWidgets.QMessageBox.critical(self, "Image Conversion Error", f"Failed to convert image for display: {e}")
            return QPixmap()
    
    def update_overlay(self):
        """
        Update the overlay image based on the current reference and shifted template image arrays,
        applying contrast stretching to brighten the display.
        """
        if self.ref_image_array is None or self.shifted_template_image_array is None:
            self.overlay_image_label.setText("Overlay Image Here")
            logging.debug("Overlay not updated: Reference or Shifted Template image array is None.")
            return

        # Apply contrast stretching to both arrays
        ref_enhanced = contrast_stretch(self.ref_image_array)
        template_enhanced = contrast_stretch(self.shifted_template_image_array)

        # Create a 3-channel RGB overlay using the enhanced arrays
        overlay_array = np.zeros((self.ref_image_array.shape[0], self.ref_image_array.shape[1], 3), dtype=np.uint8)
        overlay_array[:, :, 0] = ref_enhanced        # Red channel (Reference Image)
        overlay_array[:, :, 1] = template_enhanced    # Green channel (Shifted Template Image)
        # Blue channel remains zero for clarity

        logging.debug("Created RGB overlay from enhanced reference and shifted template images.")

        # Convert the overlay array to QImage
        bytes_per_line = 3 * overlay_array.shape[1]
        overlay_qimage = QtGui.QImage(overlay_array.tobytes(), overlay_array.shape[1], overlay_array.shape[0], bytes_per_line, QtGui.QImage.Format_RGB888)

        if overlay_qimage.isNull():
            logging.error("Failed to create QImage from overlay array.")
            self.overlay_image_label.setText("Overlay Creation Failed")
            return

        # Convert QImage to QPixmap
        overlay_pixmap = QPixmap.fromImage(overlay_qimage)
        logging.debug("Converted overlay NumPy array to QPixmap.")

        # Scale the pixmap to fit the QLabel's size
        scaled_pixmap = overlay_pixmap.scaled(self.overlay_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.overlay_image_label.setPixmap(scaled_pixmap)
        logging.debug("Set scaled overlay pixmap to QLabel.")

    
    def resizeEvent(self, event):
        """
        Override the resizeEvent to rescale the overlay pixmap when the window is resized.
        """
        #self.update_overlay()
        super(MainWindow, self).resizeEvent(event)
        logging.debug("Handled window resize event and updated overlay.")
    
    def keyPressEvent(self, event):
        """
        Handle key press events for shifting the template image.
        """
        key = event.key()
        try:
            shift_x = float(self.shift_step_x_edit.text()) if self.shift_step_x_edit.text() else 0.0
            shift_y = float(self.shift_step_y_edit.text()) if self.shift_step_y_edit.text() else 0.0
            logging.debug(f"Shift steps - X: {shift_x}, Y: {shift_y}")
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Invalid Input", "Shift steps must be numeric.")
            logging.error("Non-numeric shift step entered.")
            return

        if key == Qt.Key_Up:
            self.config["current_deltay"] -= shift_y
            logging.debug(f"Pressed Up key. New Delta Y: {self.config['current_deltay']}")
        elif key == Qt.Key_Down:
            self.config["current_deltay"] += shift_y
            logging.debug(f"Pressed Down key. New Delta Y: {self.config['current_deltay']}")
        elif key == Qt.Key_Left:
            self.config["current_deltax"] -= shift_x
            logging.debug(f"Pressed Left key. New Delta X: {self.config['current_deltax']}")
        elif key == Qt.Key_Right:
            self.config["current_deltax"] += shift_x
            logging.debug(f"Pressed Right key. New Delta X: {self.config['current_deltax']}")
        else:
            super(MainWindow, self).keyPressEvent(event)
            return

        # Update the shift fields
        self.deltaX_edit.setText(str(self.config["current_deltax"]))
        self.deltaY_edit.setText(str(self.config["current_deltay"]))
        logging.debug("Updated Delta X and Delta Y QLineEdits.")

        # Apply the shift to the template image and update the overlay
        self.apply_shift_and_update_overlay()
    
    def set_shift_x(self):
        """
        Handle updating the shift based on user input for Delta X.
        """
        text = self.deltaX_edit.text()
        try:
            new_shift_x = float(text)
            self.config["current_deltax"] = new_shift_x
            logging.debug(f"Set current_deltax to {new_shift_x} from user input.")
            self.apply_shift_and_update_overlay()
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Invalid Input", "Current Delta X must be a numeric value.")
            logging.error("Invalid input for Current Delta X.")
            # Reset to previous valid value
            self.deltaX_edit.setText(str(self.config["current_deltax"]))
    
    def set_shift_y(self):
        """
        Handle updating the shift based on user input for Delta Y.
        """
        text = self.deltaY_edit.text()
        try:
            new_shift_y = float(text)
            self.config["current_deltay"] = new_shift_y
            logging.debug(f"Set current_deltay to {new_shift_y} from user input.")
            self.apply_shift_and_update_overlay()
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Invalid Input", "Current Delta Y must be a numeric value.")
            logging.error("Invalid input for Current Delta Y.")
            # Reset to previous valid value
            self.deltaY_edit.setText(str(self.config["current_deltay"]))
    
    def apply_shift_and_update_overlay(self):
        """
        Shift the template image and its mask based on current_deltax and current_deltay,
        update the overlay image, and recompute MSE and perceptual loss.
        """
        if self.template_image_array is None or self.ref_image_array is None:
            logging.warning("Cannot apply shift: Original Template or Reference image is None.")
            return

        if self.template_mask_array is None or self.ref_mask_array is None:
            logging.warning("Cannot apply shift: Template or Reference mask is None.")
            return

        # Ensure that image and mask dimensions match
        if self.template_image_array.shape != self.ref_image_array.shape:
            QtWidgets.QMessageBox.critical(self, "Dimension Mismatch",
                                           "Template image dimensions do not match Reference image dimensions.")
            logging.error("Template image dimensions do not match Reference image dimensions.")
            return

        if self.template_mask_array.shape != self.ref_mask_array.shape:
            QtWidgets.QMessageBox.critical(self, "Dimension Mismatch",
                                           "Template mask dimensions do not match Reference mask dimensions.")
            logging.error("Template mask dimensions do not match Reference mask dimensions.")
            return

        # Total shifts
        total_shift_x = self.config["current_deltax"]
        total_shift_y = self.config["current_deltay"]

        logging.debug(f"Applying total shift: Delta X={total_shift_x}, Delta Y={total_shift_y}")

        # Apply shift to the template image
        shifted_image = ndi_shift(
            self.template_image_array,
            shift=(total_shift_y, total_shift_x),
            mode='reflect',
            order=3  # Cubic interpolation for images
        )
        logging.debug("Applied shift to template image using scipy.ndimage.shift.")

        # Update the shifted template image array
        self.shifted_template_image_array = shifted_image.copy()

        # Apply shift to the template mask
        shifted_mask = ndi_shift(
            self.template_mask_array.astype(float),
            shift=(total_shift_y, total_shift_x),
            mode='reflect',
            order=0  # Nearest-neighbor for masks
        )
        shifted_mask = shifted_mask > 0.5  # Re-binarize the mask
        logging.debug("Applied shift to template mask and re-binarized.")

        # Update the display pixmap with the shifted image
        shifted_display_arr = contrast_stretch(shifted_image)  # Apply contrast stretching for display
        self.template_display_pixmap = self.array_to_qpixmap(shifted_display_arr, is_grayscale=True)
        logging.debug("Updated template_display_pixmap with shifted and contrast-stretched image.")

        # Update the overlay
        self.update_overlay()
        logging.debug("Updated overlay after shifting.")

        # Compute Losses
        mse = self.compute_mse(self.ref_image_array, shifted_image)
        pl = self.compute_perceptual_loss(self.ref_image_array, shifted_image)
        logging.debug(f"Computed MSE: {mse}, Perceptual Loss: {pl}")

        # Append to loss history
        self.mse_history.append(mse)
        self.pl_history.append(pl)
        logging.debug("Appended MSE and Perceptual Loss to histories.")

        # Update plots
        self.update_plots(self.mse_history, self.pl_history)
        logging.debug("Updated MSE and Perceptual Loss plots.")

        # Update difference heatmap
        self.update_difference_heatmap()
        logging.debug("Updated difference heatmap.")

        # Update the mask display
        self.template_mask_array = shifted_mask.astype(np.uint8) * 255  # Convert boolean to binary mask
        self.template_mask_pixmap = self.array_to_qpixmap(self.template_mask_array, is_grayscale=True)
        logging.debug("Updated template_mask_pixmap with shifted mask.")
    
    def compute_mse(self, ref, shifted):
        """
        Compute Mean Squared Error between reference and shifted template images.
        """
        mse = np.mean((ref.astype(float) - shifted.astype(float)) ** 2)
        return mse
    
    def compute_perceptual_loss(self, ref, shifted):
        """
        Compute Perceptual Loss between reference and shifted template images.
        Placeholder function. Replace with actual implementation.
        """
        # Placeholder: using MSE as a stand-in
        pl = self.compute_mse(ref, shifted)
        return pl
    
    def initialize_plots(self):
        """
        Initialize the plots with empty data.
        """
        self.mse_ax.clear()
        self.mse_ax.set_title("MSE over Shifts")
        self.mse_ax.set_xlabel("Shift Steps")
        self.mse_ax.set_ylabel("MSE")
        self.mse_ax.plot([], [], 'r-')
        self.mse_canvas.draw()

        self.pl_ax.clear()
        self.pl_ax.set_title("Perceptual Loss over Shifts")
        self.pl_ax.set_xlabel("Shift Steps")
        self.pl_ax.set_ylabel("Perceptual Loss")
        self.pl_ax.plot([], [], 'b-')
        self.pl_canvas.draw()
        logging.debug("Initialized MSE and Perceptual Loss plots.")
    
    def update_plots(self, mse_values, pl_values):
        """
        Update the plots with new MSE and perceptual loss values.
        """
        shift_steps = range(len(mse_values))
        logging.debug(f"Updating plots with {len(mse_values)} data points.")

        # Update MSE Plot
        self.mse_ax.clear()
        self.mse_ax.set_title("MSE over Shifts")
        self.mse_ax.set_xlabel("Shift Steps")
        self.mse_ax.set_ylabel("MSE")
        self.mse_ax.plot(shift_steps, mse_values, 'r-')
        self.mse_canvas.draw()
        logging.debug("Updated MSE plot.")

        # Update Perceptual Loss Plot
        self.pl_ax.clear()
        self.pl_ax.set_title("Perceptual Loss over Shifts")
        self.pl_ax.set_xlabel("Shift Steps")
        self.pl_ax.set_ylabel("Perceptual Loss")
        self.pl_ax.plot(shift_steps, pl_values, 'b-')
        self.pl_canvas.draw()
        logging.debug("Updated Perceptual Loss plot.")
    
    def compute_difference_heatmap(self):
        """
        Compute the masked difference heatmap between reference and shifted template images.
        """
        if self.ref_image_array is None or self.shifted_template_image_array is None:
            logging.warning("Cannot compute difference heatmap: Reference or Shifted Template image array is None.")
            return QPixmap()

        # Compute absolute difference
        diff_array = np.abs(self.ref_image_array.astype(float) - self.shifted_template_image_array.astype(float))
        logging.debug("Computed absolute difference between Reference and Shifted Template images.")

        # Apply masks if available
        if self.ref_mask_array is not None and self.template_mask_array is not None:
            combined_mask = (self.ref_mask_array > 0) & (self.template_mask_array > 0)
            diff_array[~combined_mask] = 0
            logging.debug("Applied combined masks to difference image.")

        # Normalize the difference for heatmap visualization
        diff_normalized = contrast_stretch(diff_array)
        logging.debug("Applied contrast stretching to difference image for heatmap.")

        # Apply color map (e.g., Jet) using matplotlib
        colored_diff = plt.get_cmap('jet')(diff_normalized / 255.0)[:, :, :3]  # Ignore alpha
        logging.debug("Applied Jet color map to difference image.")

        # Convert to 8-bit
        colored_diff = (colored_diff * 255).astype(np.uint8)
        logging.debug("Converted color-mapped difference image to 8-bit.")

        # Convert NumPy array to QImage
        height, width = colored_diff.shape[:2]
        bytes_per_line = 3 * width
        heatmap_qimage = QtGui.QImage(colored_diff.tobytes(), width, height, bytes_per_line, QtGui.QImage.Format_RGB888).copy()
        if heatmap_qimage.isNull():
            logging.error("Failed to create QImage from difference heatmap array.")
            return QPixmap()
        
        # Convert QImage to QPixmap
        heatmap_pixmap = QPixmap.fromImage(heatmap_qimage)
        logging.debug("Converted difference QImage to QPixmap.")

        # Scale to fit the diff_image_label
        scaled_pixmap = heatmap_pixmap.scaled(self.diff_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        logging.debug("Scaled difference pixmap to fit QLabel size.")

        return scaled_pixmap

    def update_difference_heatmap(self):
        """
        Update the difference heatmap display.
        """
        heatmap_pixmap = self.compute_difference_heatmap()
        if not heatmap_pixmap.isNull():
            self.diff_image_label.setPixmap(heatmap_pixmap)
            logging.debug("Set updated difference pixmap to QLabel.")
        else:
            self.diff_image_label.setText("Difference Heatmap Failed")
            logging.error("Failed to update difference heatmap.")

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    logging.debug("Application window displayed.")
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()