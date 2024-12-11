import sys
import os
import json
import logging
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtGui import QPixmap, QPainter, QColor
import numpy as np
from PyQt5.QtCore import Qt
from scipy.ndimage import shift as ndimage_shift

# Suppress excessive matplotlib font manager logs
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
import registration_helpers as rh



class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        
        # ----- Load Configuration -----
        self.config = rh.load_config()
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
        self.overlay_image_label.setFixedSize(400, 400)  # Adjust as needed
        
        # Placeholder for masked difference image
        self.diff_image_label = QtWidgets.QLabel("Masked Difference Image Here")
        self.diff_image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.diff_image_label.setStyleSheet("border: 1px solid gray;")
        self.diff_image_label.setFixedSize(400, 400)  # Adjust as needed
        
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
        self.setFocusPolicy(Qt.StrongFocus)
        
        # ----- Initialize Pixmaps and Masks -----
        self.ref_pixmap = None
        self.template_pixmap = None
        self.original_template_pixmap = None  # Store original template
        self.ref_mask = None
        self.template_mask = None
        
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
        Load an image or mask from the given filepath, apply contrast stretching, and update the pixmap.
        """
        pixmap = QPixmap(filepath)
        if pixmap.isNull():
            QtWidgets.QMessageBox.warning(self, "Load Image", f"Failed to load {image_type} from {filepath}.")
            logging.error(f"Failed to load {image_type} from {filepath}.")
            return

        # Convert QPixmap to QImage
        qimage = pixmap.toImage()
        logging.debug(f"Loaded {image_type} with format: {qimage.format()} and depth: {qimage.depth()}")

        # Determine the appropriate format based on bit depth
        if qimage.depth() == 16:
            # Check if the image is grayscale or RGB
            if qimage.format() == QtGui.QImage.Format_Grayscale16:
                qimage = qimage.convertToFormat(QtGui.QImage.Format_Grayscale16)
                logging.debug(f"Converted {image_type} to Grayscale16 format.")
            elif qimage.format() == QtGui.QImage.Format_RGB16:
                qimage = qimage.convertToFormat(QtGui.QImage.Format_RGB16)
                logging.debug(f"Converted {image_type} to RGB16 format.")
            else:
                logging.warning(f"{image_type} has an unexpected 16-bit format: {qimage.format()}.")
        else:
            qimage = qimage.convertToFormat(QtGui.QImage.Format_RGB32)
            logging.debug(f"Converted {image_type} to RGB32 format.")

        # Convert QImage to NumPy array
        width = qimage.width()
        height = qimage.height()
        ptr = qimage.bits()
        ptr.setsize(qimage.byteCount())

        if qimage.depth() == 16:
            if qimage.format() == QtGui.QImage.Format_Grayscale16:
                # Grayscale 16-bit
                arr = np.frombuffer(ptr, dtype=np.uint16).reshape((height, width))
                logging.debug(f"{image_type} NumPy array shape (grayscale 16-bit): {arr.shape}")
                # Apply contrast stretching
                arr_stretched = rh.contrast_stretch(arr)
                logging.debug(f"Applied contrast stretching to {image_type} (grayscale 16-bit).")
                # Convert to 8-bit
                arr_stretched = arr_stretched.astype(np.uint8)
                # Convert back to QImage
                qimage = QtGui.QImage(arr_stretched.tobytes(), width, height, width, QtGui.QImage.Format_Grayscale8)
                logging.debug(f"Converted {image_type} to Grayscale8 format after stretching.")
            elif qimage.format() == QtGui.QImage.Format_RGB16:
                # RGB 16-bit
                arr = np.frombuffer(ptr, dtype=np.uint16).reshape((height, width, 3))
                logging.debug(f"{image_type} NumPy array shape (RGB 16-bit): {arr.shape}")
                # Apply contrast stretching to each channel
                for channel in range(3):
                    original_min, original_max = arr[:, :, channel].min(), arr[:, :, channel].max()
                    arr[:, :, channel] = rh.contrast_stretch(arr[:, :, channel])
                    logging.debug(f"Applied contrast stretching to {image_type} channel {channel}: min {original_min}, max {original_max}")
                # Convert to 8-bit
                arr = arr.astype(np.uint8)
                # Convert back to QImage
                qimage = QtGui.QImage(arr.tobytes(), width, height, 3 * width, QtGui.QImage.Format_RGB888)
                logging.debug(f"Converted {image_type} to RGB888 format after stretching.")
        else:
            # Assuming RGB 8-bit
            arr = np.frombuffer(ptr, dtype=np.uint8).reshape((height, width, 4))[:, :, :3]
            logging.debug(f"{image_type} NumPy array shape (RGB 8-bit): {arr.shape}")
            # Apply contrast stretching to each channel
            for channel in range(3):
                original_min, original_max = arr[:, :, channel].min(), arr[:, :, channel].max()
                arr[:, :, channel] = rh.contrast_stretch(arr[:, :, channel])
                logging.debug(f"Applied contrast stretching to {image_type} channel {channel}: min {original_min}, max {original_max}")
            # Convert back to QImage
            qimage = QtGui.QImage(arr.tobytes(), width, height, 3 * width, QtGui.QImage.Format_RGB888)
            logging.debug(f"Converted {image_type} to RGB888 format after stretching.")

        # Convert QImage back to QPixmap
        stretched_pixmap = QPixmap.fromImage(qimage)
        logging.debug(f"Created stretched QPixmap for {image_type}.")

        # Assign to the correct attribute
        if image_type == "reference_image":
            self.ref_pixmap = stretched_pixmap
        elif image_type == "template_image":
            self.template_pixmap = stretched_pixmap
            self.original_template_pixmap = stretched_pixmap  # Store original
        elif image_type == "reference_mask":
            self.ref_mask = stretched_pixmap
        elif image_type == "template_mask":
            self.template_mask = stretched_pixmap
        else:
            logging.warning(f"Unknown image type: {image_type}")

        logging.debug(f"Updated {image_type} pixmap attribute.")
        self.update_overlay()

    
    def create_color_coded_overlay(self, ref_pixmap, template_pixmap):
        """
        Create a color-coded overlay where the reference image is mapped to the red channel
        and the template image is mapped to the green channel.
        """
        if ref_pixmap is None or template_pixmap is None:
            logging.debug("Cannot create overlay: one or both pixmaps are None.")
            return QPixmap()

        # Ensure both pixmaps are the same size
        if ref_pixmap.size() != template_pixmap.size():
            # Scale both pixmaps to the minimum size while keeping aspect ratio
            min_width = min(ref_pixmap.width(), template_pixmap.width())
            min_height = min(ref_pixmap.height(), template_pixmap.height())
            ref_pixmap = ref_pixmap.scaled(min_width, min_height, Qt.KeepAspectRatio)
            template_pixmap = template_pixmap.scaled(min_width, min_height, Qt.KeepAspectRatio)
            logging.debug("Scaled reference and template pixmaps to the minimum common size.")

        # Convert QPixmap to QImage
        ref_qimage = ref_pixmap.toImage().convertToFormat(QtGui.QImage.Format_RGB32)
        template_qimage = template_pixmap.toImage().convertToFormat(QtGui.QImage.Format_RGB32)
        logging.debug("Converted pixmaps to QImage Format_RGB32.")

        # Convert QImage to NumPy arrays
        ref_buffer = ref_qimage.bits().asstring(ref_qimage.byteCount())
        ref_array = np.frombuffer(ref_buffer, dtype=np.uint8).reshape((ref_qimage.height(), ref_qimage.width(), 4))[:, :, :3]

        template_buffer = template_qimage.bits().asstring(template_qimage.byteCount())
        template_array = np.frombuffer(template_buffer, dtype=np.uint8).reshape((template_qimage.height(), template_qimage.width(), 4))[:, :, :3]

        logging.debug("Converted QImages to NumPy arrays.")

        # Initialize overlay array with zeros
        overlay_array = np.zeros_like(ref_array)

        # Assign reference image to red channel
        overlay_array[:, :, 0] = ref_array[:, :, 0]
        logging.debug("Assigned reference image to red channel of overlay.")

        # Assign template image to green channel
        overlay_array[:, :, 1] = template_array[:, :, 1]
        logging.debug("Assigned template image to green channel of overlay.")

        # Blue channel remains zero
        logging.debug("Blue channel remains zero in overlay.")

        # Convert NumPy array back to QImage
        height, width, channel = overlay_array.shape
        bytes_per_line = 3 * width
        overlay_qimage = QtGui.QImage(overlay_array.tobytes(), width, height, bytes_per_line, QtGui.QImage.Format_RGB888).copy()
        logging.debug("Converted overlay NumPy array back to QImage.")

        # Convert QImage back to QPixmap
        overlay_pixmap = QPixmap.fromImage(overlay_qimage)
        logging.debug("Converted overlay QImage to QPixmap.")

        # Scale the overlay pixmap to fit the QLabel's size
        label_size = self.overlay_image_label.size()
        scaled_pixmap = overlay_pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        logging.debug("Scaled overlay pixmap to fit QLabel size.")

        return scaled_pixmap

    def update_overlay(self):
        """
        Update the overlay image based on the current reference and template pixmaps.
        """
        if self.ref_pixmap is None or self.template_pixmap is None:
            self.overlay_image_label.setText("Overlay Image Here")
            logging.debug("Overlay pixmap not updated because one of the images is None.")
            return

        overlay_pixmap = self.create_color_coded_overlay(self.ref_pixmap, self.template_pixmap)
        self.overlay_image_label.setPixmap(overlay_pixmap)
        logging.debug("Overlay pixmap updated and set to QLabel.")
    
    def resizeEvent(self, event):
        """
        Override the resizeEvent to rescale the overlay pixmap when the window is resized.
        """
        self.update_overlay()
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
    
    def apply_shift_and_update_overlay(self):
        """
        Shift the template image based on current_deltax and current_deltay,
        update the overlay image, and recompute MSE and perceptual loss.
        """
        if self.original_template_pixmap is None or self.ref_pixmap is None:
            logging.warning("Cannot apply shift: original_template_pixmap or ref_pixmap is None.")
            return

        # Convert original_template_pixmap to QImage
        template_qimage = self.original_template_pixmap.toImage().convertToFormat(QtGui.QImage.Format_RGB32)
        ref_qimage = self.ref_pixmap.toImage().convertToFormat(QtGui.QImage.Format_RGB32)
        logging.debug("Converted original_template_pixmap and ref_pixmap to QImage Format_RGB32 for shifting.")

        # Convert QImage to NumPy arrays
        template_buffer = template_qimage.bits().asstring(template_qimage.byteCount())
        template_array = np.frombuffer(template_buffer, dtype=np.uint8).reshape((template_qimage.height(), template_qimage.width(), 4))[:, :, :3]
        logging.debug("Converted original_template_pixmap QImage to NumPy array.")

        ref_buffer = ref_qimage.bits().asstring(ref_qimage.byteCount())
        ref_array = np.frombuffer(ref_buffer, dtype=np.uint8).reshape((ref_qimage.height(), ref_qimage.width(), 4))[:, :, :3]
        logging.debug("Converted ref_pixmap QImage to NumPy array.")

        # Apply fractional shift using scipy.ndimage.shift
        shifted_template = ndimage_shift(template_array, shift=(self.config["current_deltay"], self.config["current_deltax"], 0), order=1, mode='constant', cval=0)
        logging.debug(f"Applied shift: Delta X={self.config['current_deltax']}, Delta Y={self.config['current_deltay']}")

        # Convert shifted array back to QImage
        height, width, channel = shifted_template.shape
        bytes_per_line = 3 * width
        shifted_qimage = QtGui.QImage(shifted_template.tobytes(), width, height, bytes_per_line, QtGui.QImage.Format_RGB888).copy()
        logging.debug("Converted shifted NumPy array back to QImage.")

        # Convert QImage back to QPixmap
        shifted_pixmap = QPixmap.fromImage(shifted_qimage)
        if shifted_pixmap.isNull():
            logging.error("Shifted pixmap is null. QImage to QPixmap conversion failed.")
        else:
            logging.debug("Converted shifted QImage to QPixmap.")

        # Update the template_pixmap with shifted image
        self.template_pixmap = shifted_pixmap
        logging.debug("Updated template_pixmap with shifted image.")

        # Update the overlay
        self.update_overlay()
        logging.debug("Updated overlay after shifting.")

        # Compute Losses
        mse = self.compute_mse(ref_array, shifted_template)
        pl = self.compute_perceptual_loss(ref_array, shifted_template)
        logging.debug(f"Computed MSE: {mse}, Perceptual Loss: {pl}")

        # Append to loss history
        self.mse_history.append(mse)
        self.pl_history.append(pl)
        logging.debug(f"Appended MSE and Perceptual Loss to histories.")

        # Update plots
        self.update_plots(self.mse_history, self.pl_history)
        logging.debug("Updated MSE and Perceptual Loss plots.")

        # Update difference heatmap
        self.update_difference_heatmap()
        logging.debug("Updated difference heatmap.")

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
        if self.ref_pixmap is None or self.template_pixmap is None:
            logging.warning("Cannot compute difference heatmap: one or both pixmaps are None.")
            return QPixmap()

        # Convert QPixmap to QImage
        ref_qimage = self.ref_pixmap.toImage().convertToFormat(QtGui.QImage.Format_RGB32)
        template_qimage = self.template_pixmap.toImage().convertToFormat(QtGui.QImage.Format_RGB32)
        logging.debug("Converted reference and template pixmaps to QImage for difference computation.")

        # Convert QImage to NumPy arrays
        ref_buffer = ref_qimage.bits().asstring(ref_qimage.byteCount())
        ref_array = np.frombuffer(ref_buffer, dtype=np.uint8).reshape((ref_qimage.height(), ref_qimage.width(), 4))[:, :, :3]

        template_buffer = template_qimage.bits().asstring(template_qimage.byteCount())
        template_array = np.frombuffer(template_buffer, dtype=np.uint8).reshape((template_qimage.height(), template_qimage.width(), 4))[:, :, :3]

        logging.debug("Converted QImages to NumPy arrays for difference computation.")

        # Compute absolute difference
        diff_array = np.abs(ref_array.astype(float) - template_array.astype(float))
        logging.debug("Computed absolute difference between reference and template images.")

        # Compute grayscale difference
        diff_gray = np.mean(diff_array, axis=2)
        logging.debug("Computed grayscale difference.")

        # Normalize to 0-255
        diff_normalized = rh.contrast_stretch(diff_gray)
        logging.debug("Applied contrast stretching to difference image.")

        # Apply masks if available
        if self.ref_mask is not None and self.template_mask is not None:
            # Convert masks to QImage
            ref_mask_qimage = self.ref_mask.toImage().convertToFormat(QtGui.QImage.Format_Grayscale8)
            template_mask_qimage = self.template_mask.toImage().convertToFormat(QtGui.QImage.Format_Grayscale8)
            # Convert QImage to NumPy arrays
            ref_mask_buffer = ref_mask_qimage.bits().asstring(ref_mask_qimage.byteCount())
            ref_mask = np.frombuffer(ref_mask_buffer, dtype=np.uint8).reshape((ref_mask_qimage.height(), ref_mask_qimage.width()))
            template_mask_buffer = template_mask_qimage.bits().asstring(template_mask_qimage.byteCount())
            template_mask = np.frombuffer(template_mask_buffer, dtype=np.uint8).reshape((template_mask_qimage.height(), template_mask_qimage.width()))
            # Combine masks
            combined_mask = (ref_mask > 0) & (template_mask > 0)
            # Apply mask to difference
            diff_normalized[~combined_mask] = 0
            logging.debug("Applied combined masks to difference image.")

        # Apply color map (e.g., Jet) using matplotlib
        colored_diff = plt.get_cmap('jet')(diff_normalized / 255.0)[:, :, :3]  # Ignore alpha
        logging.debug("Applied Jet color map to difference image.")

        # Convert to 8-bit
        colored_diff = (colored_diff * 255).astype(np.uint8)
        logging.debug("Converted color-mapped difference image to 8-bit.")

        # Convert NumPy array to QImage
        height, width, channel = colored_diff.shape
        bytes_per_line = 3 * width
        heatmap_qimage = QtGui.QImage(colored_diff.tobytes(), width, height, bytes_per_line, QtGui.QImage.Format_RGB888).copy()
        logging.debug("Converted colored difference NumPy array to QImage.")

        # Convert QImage back to QPixmap
        heatmap_pixmap = QPixmap.fromImage(heatmap_qimage)
        logging.debug("Converted difference QImage to QPixmap.")

        # Scale to fit the diff_image_label
        label_size = self.diff_image_label.size()
        scaled_pixmap = heatmap_pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        logging.debug("Scaled difference pixmap to fit QLabel size.")

        return scaled_pixmap

    def update_difference_heatmap(self):
        """
        Update the difference heatmap display.
        """
        heatmap_pixmap = self.compute_difference_heatmap()
        self.diff_image_label.setPixmap(heatmap_pixmap)
        logging.debug("Set updated difference pixmap to QLabel.")

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    logging.debug("Application window displayed.")
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
