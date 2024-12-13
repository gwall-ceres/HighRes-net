# registration_app.py

import sys
import logging
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap  # Directly import QPixmap
import numpy as np
from scipy.ndimage import shift as ndi_shift
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import registration_helpers as rh
import preprocess_images as ppi
from heatmap_canvas import HeatmapCanvas

# Corrected logging configuration to suppress DEBUG messages
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress excessive matplotlib font manager logs
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        # Initialize the perceptual loss model
        self.perceptual_loss_model = ppi.init_VGG_for_perceptual_loss()

        # ----- Load Configuration -----
        self.config = rh.load_config()

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

        # ----- Central Widget and Grid Layout -----
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        grid_layout = QtWidgets.QGridLayout(central_widget)

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

        # Add current shift layout to grid
        grid_layout.addLayout(current_shift_layout, 0, 0, 1, 2)  # Span across 2 columns

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

        # Add shift step layout to grid
        grid_layout.addLayout(shift_step_layout, 1, 0, 1, 2)  # Span across 2 columns

        # ----- Image Display Layout -----
        images_layout = QtWidgets.QHBoxLayout()

        # Placeholder for reference+template overlay image
        self.overlay_image_label = QtWidgets.QLabel("Overlay Image Here")
        self.overlay_image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.overlay_image_label.setStyleSheet("border: 1px solid gray;")
        self.overlay_image_label.setScaledContents(True)  # Ensure the image scales within the label
        images_layout.addWidget(self.overlay_image_label)

        # HeatmapCanvas for the difference heatmap
        self.heatmap_canvas = HeatmapCanvas(self, width=5, height=4, dpi=100)
        self.heatmap_canvas.setStyleSheet("border: 1px solid gray;")
        images_layout.addWidget(self.heatmap_canvas)

        # Add image display layout to grid
        grid_layout.addLayout(images_layout, 2, 0, 1, 2)  # Span across 2 columns

        # ----- Graphs Layout -----
        graphs_layout = QtWidgets.QHBoxLayout()

        # Initialize MSE Plot
        self.mse_fig = Figure(figsize=(4, 3))
        self.mse_canvas = FigureCanvas(self.mse_fig)
        self.mse_ax = self.mse_fig.add_subplot(111)
        self.mse_ax.set_title("MSE over Shifts")
        self.mse_ax.set_xlabel("Shift Steps")
        self.mse_ax.set_ylabel("MSE")
        graphs_layout.addWidget(self.mse_canvas)

        # Initialize Perceptual Loss Plot
        self.pl_fig = Figure(figsize=(4, 3))
        self.pl_canvas = FigureCanvas(self.pl_fig)
        self.pl_ax = self.pl_fig.add_subplot(111)
        self.pl_ax.set_title("Perceptual Loss over Shifts")
        self.pl_ax.set_xlabel("Shift Steps")
        self.pl_ax.set_ylabel("Perceptual Loss")
        graphs_layout.addWidget(self.pl_canvas)

        # Add graphs layout to grid
        grid_layout.addLayout(graphs_layout, 3, 0, 1, 2)  # Span across 2 columns

        # Set stretch factors for rows and columns
        grid_layout.setRowStretch(0, 1)  # First row (shift controls)
        grid_layout.setRowStretch(1, 1)  # Second row (shift steps)
        grid_layout.setRowStretch(2, 3)  # Third row (image display)
        grid_layout.setRowStretch(3, 2)  # Fourth row (graphs)

        grid_layout.setColumnStretch(0, 1)  # Left column (overlay image + graphs)
        grid_layout.setColumnStretch(1, 1)  # Right column (heatmap + graphs)

        # ----- Set Window Properties -----
        self.setWindowTitle("Interactive Image Alignment Tool")
        self.resize(1200, 800)  # Adjust as needed to accommodate image and graph sizes

        # ----- Focus Policy for Keyboard Events -----
        self.setFocusPolicy(Qt.StrongFocus)  # Now works because Qt is imported

        # ----- Initialize Image Arrays and Pixmaps -----
        self.ref_image_array = None  # Original Reference Image (Grayscale)
        self.ref_display_pixmap = None  # Display Reference Image (Contrast Stretched)

        self.template_image_array = None  # Original Template Image (Grayscale)
        self.template_display_pixmap = None  # Display Template Image (Contrast Stretched)

        self.ref_mask_array = None  # Original Reference Mask (Grayscale)
        self.ref_mask_pixmap = None

        self.template_mask_array = None  # Original Template Mask (Grayscale)
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
        # logging.debug("Initialized MSE and Perceptual Loss plots.")

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
        # Use scikit-image to load images
        arr = ppi.read_image(filepath)
        # logging.debug(f"{image_type} NumPy array shape: {arr.shape}, dtype: {arr.dtype}")

        # Assign to the correct attribute based on image type
        if image_type == "reference_image":
            self.ref_image_array = arr.astype(np.float32)  # Store original
            self.ref_image_array.flags.writeable = False
            display_arr = ppi.contrast_stretch_8bit(self.ref_image_array)  # Apply contrast stretching for display
            self.ref_display_pixmap = self.array_to_qpixmap(display_arr, is_grayscale=True)
        elif image_type == "template_image":
            self.template_image_array = arr.astype(np.float32)  # Store original
            self.template_image_array.flags.writeable = False
            display_arr = ppi.contrast_stretch_8bit(self.template_image_array)  # Apply contrast stretching for display
            self.template_display_pixmap = self.array_to_qpixmap(display_arr, is_grayscale=True)
        elif image_type == "reference_mask":
            self.ref_mask_array = arr.astype(bool)  # Store original mask without contrast stretching
            self.ref_mask_array.flags.writeable = False
            self.ref_mask_pixmap = self.array_to_qpixmap(self.ref_mask_array, is_grayscale=True)
        elif image_type == "template_mask":
            self.template_mask_array = arr.astype(bool)  # Store original mask without contrast stretching
            self.template_mask_array.flags.writeable = False
            self.template_mask_pixmap = self.array_to_qpixmap(self.template_mask_array, is_grayscale=True)
        else:
            logging.warning(f"Unknown image type: {image_type}")
            return

        # logging.debug(f"Stored {image_type} data.")

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

        # Update the overlay only if both reference and template images are loaded
        if image_type in ["reference_image", "template_image"] and self.ref_image_array is not None and self.template_image_array is not None:
            self.update_overlay(self.template_image_array)

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
            # Remove manual scaling
            return pixmap
        except Exception as e:
            logging.error(f"Failed to convert array to QPixmap: {e}")
            QtWidgets.QMessageBox.critical(self, "Image Conversion Error", f"Failed to convert image for display: {e}")
            return QPixmap()

    def update_overlay(self, shifted_template_image):
        """
        Update the overlay image based on the current reference and shifted template image arrays,
        applying contrast stretching to brighten the display.

        Args:
            shifted_template_image (np.ndarray): Shifted template image array.
        """
        if self.ref_image_array is None or shifted_template_image is None:
            self.overlay_image_label.setText("Overlay Image Here")
            # logging.debug("Overlay not updated: Reference or Shifted Template image array is None.")
            return

        # Apply contrast stretching to both arrays
        ref_enhanced = ppi.contrast_stretch_8bit(self.ref_image_array)
        template_enhanced = ppi.contrast_stretch_8bit(shifted_template_image)

        # Create a 3-channel RGB overlay using the enhanced arrays
        overlay_array = np.zeros((self.ref_image_array.shape[0], self.ref_image_array.shape[1], 3), dtype=np.uint8)
        overlay_array[:, :, 0] = template_enhanced       # Red channel (Template Image)
        overlay_array[:, :, 1] = ref_enhanced           # Green channel (Reference Image)
        overlay_array[:, :, 2] = ref_enhanced           # Blue channel (Reference Image), creating cyan

        # Debugging: Verify the range of overlay channels
        # print(f"Overlay Array - R: {overlay_array[:, :, 0].min()} to {overlay_array[:, :, 0].max()}")
        # print(f"Overlay Array - G: {overlay_array[:, :, 1].min()} to {overlay_array[:, :, 1].max()}")
        # print(f"Overlay Array - B: {overlay_array[:, :, 2].min()} to {overlay_array[:, :, 2].max()}")

        # Convert the overlay array to QImage
        bytes_per_line = 3 * overlay_array.shape[1]
        overlay_qimage = QtGui.QImage(overlay_array.tobytes(), overlay_array.shape[1], overlay_array.shape[0],
                                      bytes_per_line, QtGui.QImage.Format_RGB888)

        if overlay_qimage.isNull():
            logging.error("Failed to create QImage from overlay array.")
            self.overlay_image_label.setText("Overlay Creation Failed")
            return

        # Convert QImage to QPixmap
        overlay_pixmap = QPixmap.fromImage(overlay_qimage)
        # Set the pixmap directly without manual scaling
        self.overlay_image_label.setPixmap(overlay_pixmap)
        # logging.debug("Set overlay pixmap to QLabel.")

    def resizeEvent(self, event):
        """
        Override the resizeEvent to rescale the overlay pixmap when the window is resized.
        """
        super(MainWindow, self).resizeEvent(event)
        if self.template_image_array is not None:
            self.update_overlay(self.template_image_array)
        if self.heatmap_canvas:
            self.heatmap_canvas.draw()
        # logging.debug("Handled window resize event and updated overlay.")

    def keyPressEvent(self, event):
        """
        Handle key press events for shifting the template image.
        """
        key = event.key()
        try:
            shift_x = float(self.shift_step_x_edit.text()) if self.shift_step_x_edit.text() else 0.0
            shift_y = float(self.shift_step_y_edit.text()) if self.shift_step_y_edit.text() else 0.0
            # logging.debug(f"Shift steps - X: {shift_x}, Y: {shift_y}")
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Invalid Input", "Shift steps must be numeric.")
            logging.error("Non-numeric shift step entered.")
            return

        if key == Qt.Key_Up:
            self.config["current_deltay"] -= shift_y
            # logging.debug(f"Pressed Up key. New Delta Y: {self.config['current_deltay']}")
        elif key == Qt.Key_Down:
            self.config["current_deltay"] += shift_y
            # logging.debug(f"Pressed Down key. New Delta Y: {self.config['current_deltay']}")
        elif key == Qt.Key_Left:
            self.config["current_deltax"] -= shift_x
            # logging.debug(f"Pressed Left key. New Delta X: {self.config['current_deltax']}")
        elif key == Qt.Key_Right:
            self.config["current_deltax"] += shift_x
            # logging.debug(f"Pressed Right key. New Delta X: {self.config['current_deltax']}")
        else:
            super(MainWindow, self).keyPressEvent(event)
            return

        # Update the shift fields
        self.deltaX_edit.setText(str(self.config["current_deltax"]))
        self.deltaY_edit.setText(str(self.config["current_deltay"]))
        # logging.debug("Updated Delta X and Delta Y QLineEdits.")

        # Apply the shift to the template image and update the overlay
        self.apply_shift_and_update_overlay()

    def closeEvent(self, event):
        """
        Handle the close event to ensure proper cleanup.
        """
        try:
            # logging.debug("Closing MainWindow and performing cleanup.")
            pass  # Perform any additional cleanup here if necessary
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")
        finally:
            super(MainWindow, self).closeEvent(event)

    def set_shift_x(self):
        """
        Handle updating the shift based on user input for Delta X.
        """
        text = self.deltaX_edit.text()
        try:
            new_shift_x = float(text)
            self.config["current_deltax"] = new_shift_x
            # logging.debug(f"Set current_deltax to {new_shift_x} from user input.")
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
            # logging.debug(f"Set current_deltay to {new_shift_y} from user input.")
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

        # logging.debug(f"Applying total shift: Delta X={total_shift_x}, Delta Y={total_shift_y}")

        # Apply shift to the template image
        shifted_image = ndi_shift(
            self.template_image_array,
            shift=(total_shift_y, total_shift_x),
            mode='reflect',
            order=3  # Cubic interpolation for images
        )
        shifted_image.flags.writeable = False
        # logging.debug("Applied shift to template image using scipy.ndimage.shift.")

        # Apply shift to the template mask
        shifted_mask = ndi_shift(
            self.template_mask_array.astype(float),
            shift=(total_shift_y, total_shift_x),
            mode='constant',
            order=0  # Nearest-neighbor for masks
        )
        shifted_mask = shifted_mask > 0.5  # Re-binarize the mask
        shifted_mask.flags.writeable = False
        # logging.debug("Applied shift to template mask and re-binarized.")

        # Update the display pixmap with the shifted image
        shifted_display_arr = ppi.contrast_stretch_8bit(shifted_image)  # Apply contrast stretching for display
        self.template_display_pixmap = self.array_to_qpixmap(shifted_display_arr, is_grayscale=True)
        # logging.debug("Updated template_display_pixmap with shifted and contrast-stretched image.")

        # Update the overlay
        self.update_overlay(shifted_image)
        # logging.debug("Updated overlay after shifting.")

        # Compute Losses
        mse = self.compute_mse(shifted_image, shifted_mask)
        pl = self.compute_perceptual_loss(shifted_image, shifted_mask)
        # logging.debug(f"Computed MSE: {mse}, Perceptual Loss: {pl}")

        # Append to loss history
        self.mse_history.append(mse)
        self.pl_history.append(pl)
        # logging.debug("Appended MSE and Perceptual Loss to histories.")

        # Update plots
        self.update_plots(self.mse_history, self.pl_history)
        # logging.debug("Updated MSE and Perceptual Loss plots.")

        # Update difference heatmap
        self.compute_and_display_heatmap(shifted_image, shifted_mask)
        # logging.debug("Computed and displayed difference heatmap.")

    def compute_mse(self, shifted_template, shifted_template_mask):
        """
        Compute Mean Squared Error between reference and shifted template images.
        """
        mse = ppi.compute_mse(self.ref_image_array, self.ref_mask_array, shifted_template, shifted_template_mask)
        return mse

    def compute_perceptual_loss(self, shifted_template, shifted_template_mask):
        """
        Compute Perceptual Loss between reference and shifted template images.
        Placeholder function. Replace with actual implementation.
        """
        # Placeholder: using MSE as a stand-in
        #pl = self.compute_mse(shifted_template, shifted_template_mask)
        pl = ppi.compute_perceptual_loss(self.perceptual_loss_model,
                                         self.ref_image_array,
                                         shifted_template,
                                         self.ref_mask_array,
                                         shifted_template_mask,
                                         self.perceptual_loss_model.hardware,
                                         False)
        return pl

    def update_plots(self, mse_values, pl_values):
        """
        Update the plots with new MSE and perceptual loss values.
        """
        shift_steps = range(len(mse_values))
        # logging.debug(f"Updating plots with {len(mse_values)} data points.")

        # Update MSE Plot
        self.mse_ax.clear()
        self.mse_ax.set_title("MSE over Shifts")
        self.mse_ax.set_xlabel("Shift Steps")
        self.mse_ax.set_ylabel("MSE")
        self.mse_ax.plot(shift_steps, mse_values, 'r-')
        self.mse_canvas.draw()
        # logging.debug("Updated MSE plot.")

        # Update Perceptual Loss Plot
        self.pl_ax.clear()
        self.pl_ax.set_title("Perceptual Loss over Shifts")
        self.pl_ax.set_xlabel("Shift Steps")
        self.pl_ax.set_ylabel("Perceptual Loss")
        self.pl_ax.plot(shift_steps, pl_values, 'b-')
        self.pl_canvas.draw()
        # logging.debug("Updated Perceptual Loss plot.")

    def compute_and_display_heatmap(self, shifted_template_image, shifted_template_mask):
        """
        Compute and display the masked difference heatmap between reference and shifted template images.

        Args:
            shifted_template_image (np.ndarray): Shifted template image array.
            shifted_template_mask (np.ndarray): Shifted template mask array.
        """
        try:
            self.compute_difference_heatmap(shifted_template_image, shifted_template_mask)
        except Exception as e:
            logging.error(f"Error computing and displaying heatmap: {e}")
            QtWidgets.QMessageBox.critical(self, "Heatmap Error", f"Failed to compute heatmap: {e}")

    def compute_difference_heatmap(self, shifted_template_image, shifted_template_mask):
        """
        Compute and display the masked difference heatmap between reference and shifted template images.

        Args:
            shifted_template_image (np.ndarray): Shifted template image array.
            shifted_template_mask (np.ndarray): Shifted template mask array.
        """
        if self.ref_image_array is None or shifted_template_image is None:
            logging.warning("Cannot compute difference heatmap: Reference or Shifted Template image array is None.")
            return

        if self.ref_mask_array is None or shifted_template_mask is None:
            logging.warning("Cannot compute difference heatmap: Reference or Shifted Template mask is None.")
            return

        # Combine masks
        combined_mask = np.logical_and(self.ref_mask_array, shifted_template_mask)
        if not np.any(combined_mask):
            logging.warning("No overlapping valid pixels found between the two masks.")
            return

        # logging.debug(f"Combined mask has {np.sum(combined_mask)} valid pixels.")

        # Normalize the reference image
        mean_ref = np.mean(self.ref_image_array[combined_mask])
        std_ref = np.std(self.ref_image_array[combined_mask])
        normed_ref = np.zeros_like(self.ref_image_array, dtype=float)
        normed_ref[combined_mask] = (self.ref_image_array[combined_mask] - mean_ref) / std_ref

        # Normalize the template image
        mean_template = np.mean(shifted_template_image[combined_mask])
        std_template = np.std(shifted_template_image[combined_mask])
        normed_template = np.zeros_like(shifted_template_image, dtype=float)
        normed_template[combined_mask] = (shifted_template_image[combined_mask] - mean_template) / std_template
        # Compute absolute difference
        diff_array = np.abs(normed_ref - normed_template)
        # Plot the heatmap using the updated HeatmapCanvas
        self.heatmap_canvas.plot_heatmap(diff_array, mask=combined_mask, cmap='jet')

        '''
        ref_masked = self.ref_image_array * (self.ref_mask_array.astype(np.float32) * 0.5 + 0.5)
        mov_masked = shifted_template_image * (shifted_template_mask.astype(np.float32) * 0.5 + 0.5)
        diff_array = np.abs(ref_masked - mov_masked)
        # Plot the heatmap using the updated HeatmapCanvas
        self.heatmap_canvas.plot_heatmap(diff_array, mask=None, cmap='jet')
        '''


    def save_heatmap(self):
        """
        Save the current heatmap as an image file.
        """
        # Since the user doesn't need this feature, this method can be removed.
        pass


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    # logging.debug("Application window displayed.")
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
