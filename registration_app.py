# registration_app.py
'''
Next steps
then implement several more methods for estimating the shift
skimage.registration.optical_flow_tvl1
skimage.registration.optical_flow_ilk

Look into other methods of estimating subpixel shifts.
See if SIFT works better than ORB.

'''
import sys
import logging
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap  # Directly import QPixmap
import numpy as np
from skimage import io, transform, color
from scipy.ndimage import shift as ndi_shift
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import registration_helpers as rh
import preprocess_images as ppi
import registration_metrics as rm
import registration_search as rs
from heatmap_canvas import HeatmapCanvas
from VGGFeatureExtractor import VGGFeatureExtractor

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
        self.perceptual_loss_model = VGGFeatureExtractor.init_VGG_for_perceptual_loss()

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

        # New action for computing and applying the shift
        #compute_shift_action = QtWidgets.QAction("Compute and Apply Shift", self)
        #compute_shift_action.triggered.connect(self.compute_and_apply_shift)
        #file_menu.addAction(compute_shift_action)

        toolbar = self.addToolBar("Main Toolbar")
    

        # ----- Dropdown Menu for Visualization -----
        # Add label for registration method dropdown
        reg_method_label = QtWidgets.QLabel("Registration Method:", self)
        reg_method_label.setContentsMargins(10, 0, 5, 0)  # Left, Top, Right, Bottom margins
        toolbar.addWidget(reg_method_label)
        
        self.coreg_dropdown = QtWidgets.QComboBox(self)
        self.coreg_dropdown.addItem("Fourier")
        self.coreg_dropdown.addItem("Point Matching")
        self.coreg_dropdown.addItem("NCC")
        self.coreg_dropdown.addItem("Perceptual Loss")
        self.coreg_dropdown.addItem("SSIM")
        self.coreg_dropdown.addItem("Mutual Information")
        self.coreg_dropdown.addItem("Mean L1 Error")
        #self.coreg_dropdown.currentIndexChanged.connect(self.update_visualization_choice)
        toolbar.addWidget(self.coreg_dropdown)

        compute_shift_action = QtWidgets.QAction("Compute and Apply Shift", self)
        compute_shift_action.triggered.connect(self.compute_and_apply_shift)
        toolbar.addAction(compute_shift_action)

         # Add after other toolbar items
        apply_best_shift_action = QtWidgets.QAction("Apply Best Shift", self)
        apply_best_shift_action.triggered.connect(self.apply_best_shift)
        toolbar.addAction(apply_best_shift_action)



        # ----- Dropdown Menu for Visualization -----
        self.layer_dropdown = QtWidgets.QComboBox(self)
        self.layer_dropdown.addItem("Heatmap")
        self.layer_dropdown.addItem("Layer 0 (Conv1)")
        self.layer_dropdown.addItem("Layer 5 (Conv2)")
        self.layer_dropdown.addItem("Layer 10 (Conv3)")
        self.layer_dropdown.addItem("Layer 19 (Conv4)")
        self.layer_dropdown.addItem("Layer 28 (Conv5)")
        self.layer_dropdown.addItem("Sum of Layers")

        self.layer_dropdown.currentIndexChanged.connect(self.update_visualization_choice)
        toolbar.addWidget(self.layer_dropdown)

       
        # Add after other toolbar items
        clear_history_action = QtWidgets.QAction("Clear History", self)
        clear_history_action.triggered.connect(self.reset_history)
        toolbar.addAction(clear_history_action)


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

        # Create a vertical layout for overlay controls and image
        overlay_layout = QtWidgets.QVBoxLayout()
        
        # Add dropdown for overlay type
        overlay_type_layout = QtWidgets.QHBoxLayout()
        overlay_type_label = QtWidgets.QLabel("Overlay Type:")
        self.overlay_type_dropdown = QtWidgets.QComboBox()
        self.overlay_type_dropdown.addItems([
            "Red-Cyan",
            "Template Only",
            "Template Mask",
            "Reference Only",
            "Reference Mask",
            "Side by Side",
            "Checkerboard",
            "Difference Blend",
            "HSV Difference"
        ])
        self.overlay_type_dropdown.currentIndexChanged.connect(self.handle_overlay_change)
        overlay_type_layout.addWidget(overlay_type_label)
        overlay_type_layout.addWidget(self.overlay_type_dropdown)
        overlay_layout.addLayout(overlay_type_layout)

        # Placeholder for reference+template overlay image
        self.overlay_image_label = QtWidgets.QLabel("Overlay Image Here")
        self.overlay_image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.overlay_image_label.setStyleSheet("border: 1px solid gray;")
        self.overlay_image_label.setScaledContents(True)
        overlay_layout.addWidget(self.overlay_image_label)

        images_layout.addLayout(overlay_layout)

        # HeatmapCanvas for the difference heatmap
        self.heatmap_canvas = HeatmapCanvas(self, width=5, height=4, dpi=100)
        self.heatmap_canvas.setStyleSheet("border: 1px solid gray;")
        images_layout.addWidget(self.heatmap_canvas)

        # Add image display layout to grid
        grid_layout.addLayout(images_layout, 2, 0, 1, 2)  # Span across 2 columns

        # ----- Graphs Layout -----
        graphs_layout = QtWidgets.QHBoxLayout()

        # Initialize ml1e Plot
        self.ml1e_fig = Figure(figsize=(4, 3))
        self.ml1e_canvas = FigureCanvas(self.ml1e_fig)
        self.ml1e_ax = self.ml1e_fig.add_subplot(111)
        self.ml1e_ax.set_title("ml1e over Shifts")
        #self.ml1e_ax.set_xlabel("Shift Steps")
        self.ml1e_ax.set_ylabel("ml1e")
        graphs_layout.addWidget(self.ml1e_canvas)

        #Add new figure/canvas for combined metrics
        self.metrics_fig = Figure(figsize=(4, 3))
        self.metrics_canvas = FigureCanvas(self.metrics_fig)
        self.metrics_ax = self.metrics_fig.add_subplot(111)
        self.metrics_ax.set_title("Registration Metrics")
        graphs_layout.addWidget(self.metrics_canvas)
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
        #self.template_display_pixmap = None  # Display Template Image (Contrast Stretched)

        self.ref_mask_array = None  # Original Reference Mask (Grayscale)
        self.ref_mask_pixmap = None

        self.template_mask_array = None  # Original Template Mask (Grayscale)
        self.template_mask_pixmap = None

        self.diff_features = None

        # ----- Initialize Loss Histories -----
        self.ml1e_history = []
        self.pl_history = []
        self.ssim_history = []
        self.nmi_history = []
        self.ncc_history = []
        self.shift_x_history = []
        self.shift_y_history = []

        self.best_shift_x = 0.0
        self.best_shift_y = 0.0
        self.best_perceptual_loss = float('inf')
        

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

    def reset_history(self):
        # Reset all history lists
        self.ml1e_history = []
        self.pl_history = []
        self.ssim_history = []
        self.nmi_history = []
        self.ncc_history = []
        self.shift_x_history = []
        self.shift_y_history = []

        # Clear the plots
        self.ml1e_ax.clear()
        self.metrics_ax.clear()
        
        # Reset the axes titles and labels
        self.ml1e_ax.set_title("ml1e and Perceptual Loss over Shifts")
        self.ml1e_ax.set_xlabel("Shifts (x, y)")
        self.ml1e_ax.set_ylabel("ml1e")
        
        self.metrics_ax.set_title("Registration Metrics")
        self.metrics_ax.set_xlabel("Shifts (x, y)")
        self.metrics_ax.set_ylabel("Metric Value")
        
        # Redraw the empty canvases
        self.ml1e_canvas.draw()
        self.metrics_canvas.draw()

    def initialize_plots(self):
        """
        Initialize the plots with empty data.
        """
        self.ml1e_ax.clear()
        self.ml1e_ax.set_title("ml1e over Shifts")
        self.ml1e_ax.set_xlabel("Shift Steps")
        self.ml1e_ax.set_ylabel("ml1e")
        self.ml1e_ax.plot([], [], 'r-')
        self.ml1e_canvas.draw()


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

    #scale the reference image to match the template image
    def align_image_sizes(self):
        if self.ref_image_array is None or self.template_image_array is None:
            return

        #save the original reference image
        self.ref_image_orig = self.ref_image_array
        if self.ref_image_array.shape != self.template_image_array.shape:
            # Get template dimensions
            template_height, template_width = self.template_image_array.shape
        
            # Resize reference image using Lanczos interpolation
            resized_ref = transform.resize(
                self.ref_image_array,
                (template_height, template_width),
                order=4,  # Lanczos interpolation
                mode='reflect',
                anti_aliasing=True,
                preserve_range=True
            ).astype(np.float32)
            
            
            self.ref_image_array = resized_ref
            self.ref_image_array.flags.writeable = False


    #scale the reference mask to match the template mask
    def align_mask_sizes(self):
        if self.ref_mask_array is None or self.template_mask_array is None:
            return
        
        self.ref_mask_orig = self.ref_mask_array
        if self.ref_mask_array.shape != self.template_mask_array.shape:
            # Get template dimensions
            template_height, template_width = self.template_mask_array.shape
        
            
            resized_mask = transform.resize(
                self.ref_mask_array.astype(float),
                (template_height, template_width),
                order=0,  # Nearest-neighbor interpolation
                mode='constant',
                preserve_range=True
            )
            resized_mask = resized_mask > 0.5  
            
            self.ref_mask_array = resized_mask
            self.ref_mask_array.flags.writeable = False


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
            #display_arr = ppi.contrast_stretch_8bit(self.ref_image_array, self.ref_mask_array)  # Apply contrast stretching for display
            #self.ref_display_pixmap = self.array_to_qpixmap(display_arr, is_grayscale=True)
            self.align_image_sizes()
        elif image_type == "template_image":
            self.template_image_array = arr.astype(np.float32)  # Store original
            self.template_image_array.flags.writeable = False
            #display_arr = ppi.contrast_stretch_8bit(self.template_image_array)  # Apply contrast stretching for display
            #self.template_display_pixmap = self.array_to_qpixmap(display_arr, is_grayscale=True)
            self.align_image_sizes()
        elif image_type == "reference_mask":
            self.ref_mask_array = arr.astype(bool)  # Store original mask without contrast stretching
            self.ref_mask_array.flags.writeable = False
            #self.ref_mask_pixmap = self.array_to_qpixmap(self.ref_mask_array, is_grayscale=True)
            self.align_mask_sizes()
        elif image_type == "template_mask":
            self.template_mask_array = arr.astype(bool)  # Store original mask without contrast stretching
            self.template_mask_array.flags.writeable = False
            #self.template_mask_pixmap = self.array_to_qpixmap(self.template_mask_array, is_grayscale=True)
            self.align_mask_sizes()
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
                    
        self.reset_history()
        # Update the overlay only if both reference and template images are loaded
        if image_type in ["reference_image", "template_image"] and self.ref_image_array is not None and self.template_image_array is not None:
            self.update_overlay(self.template_image_array, self.template_mask_array)


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


    def update_overlay(self, shifted_template_image, shifted_template_mask):
        """
        Update the overlay image based on the current reference and shifted template image arrays,
        applying contrast stretching to brighten the display.

        Args:
            shifted_template_image (np.ndarray): Shifted template image array.
        """
        if self.ref_image_array is None or shifted_template_image is None or shifted_template_mask is None:
            self.overlay_image_label.setText("Overlay Image Here")
            return

        # Apply contrast stretching to both arrays
        ref_enhanced = ppi.contrast_stretch_8bit(self.ref_image_array)
        template_enhanced = ppi.contrast_stretch_8bit(shifted_template_image, shifted_template_mask)
        '''
        template_enhanced = transform.resize(
            template_enhanced,
            ref_enhanced.shape,
            order=4,
            mode='constant',
            cval=0,
            anti_aliasing=True,
            preserve_range=True
        ).astype(np.float32)
        '''

        overlay_type = self.overlay_type_dropdown.currentText()
        
        if overlay_type == "Red-Cyan":
            # Create RGB overlay (Red for template, Cyan for reference)
            overlay_array = np.zeros((ref_enhanced.shape[0], ref_enhanced.shape[1], 3), dtype=np.uint8)
            overlay_array[:, :, 0] = template_enhanced  # Red channel (Template)
            overlay_array[:, :, 1] = ref_enhanced      # Green channel (Reference)
            overlay_array[:, :, 2] = ref_enhanced      # Blue channel (Reference)
        
        elif overlay_type == "Template Only":
            # Show only the template image (properly handled as grayscale)
            template_uint8 = template_enhanced.astype(np.uint8)
            overlay_array = np.stack([template_uint8] * 3, axis=-1)

        elif overlay_type == "Template Mask":
            # Show only the template mask (properly scaled to 0-255)
            mask_uint8 = (shifted_template_mask * 255).astype(np.uint8)
            overlay_array = np.stack([mask_uint8] * 3, axis=-1)
        
        elif overlay_type == "Reference Only":
            # Show only the reference image
            overlay_array = np.stack([ref_enhanced] * 3, axis=-1)

        elif overlay_type == "Reference Mask":
            # Show only the reference mask
            mask_uint8 = (self.ref_mask_array * 255).astype(np.uint8)
            overlay_array = np.stack([mask_uint8] * 3, axis=-1)
        
        elif overlay_type == "Side by Side":
            # Create side-by-side comparison
            half_width = ref_enhanced.shape[1] // 2
            overlay_array = np.zeros((ref_enhanced.shape[0], ref_enhanced.shape[1], 3), dtype=np.uint8)
            overlay_array[:, :half_width] = np.stack([ref_enhanced[:, :half_width]] * 3, axis=-1)
            overlay_array[:, half_width:] = np.stack([template_enhanced[:, half_width:]] * 3, axis=-1)
        
        elif overlay_type == "Checkerboard":
            # Create checkerboard pattern
            checker_size = 50  # Size of each checker square
            x, y = np.indices(ref_enhanced.shape)
            checker_pattern = ((x // checker_size) + (y // checker_size)) % 2
            overlay_array = np.zeros((ref_enhanced.shape[0], ref_enhanced.shape[1], 3), dtype=np.uint8)
            overlay_array[checker_pattern == 0] = np.stack([ref_enhanced[checker_pattern == 0]] * 3, axis=-1)
            overlay_array[checker_pattern == 1] = np.stack([template_enhanced[checker_pattern == 1]] * 3, axis=-1)

        elif overlay_type == "Difference Blend":
            # Create RGB array for overlay
            overlay_array = np.zeros((ref_enhanced.shape[0], ref_enhanced.shape[1], 3), dtype=np.uint8)
            combined_mask = self.ref_mask_array.astype(float) * shifted_template_mask
            # Calculate absolute difference between enhanced images
            diff = np.abs(ref_enhanced.astype(float) - template_enhanced.astype(float)) * combined_mask
            diff = (diff / diff.max() * 255).astype(np.uint8)  # Normalize to 0-255 range
            
            # Create base grayscale image (average of enhanced images)
            base = ((ref_enhanced.astype(float) + template_enhanced.astype(float)) / 2).astype(np.uint8)
            
            # Apply base grayscale to all channels
            overlay_array[..., 0] = base  # R
            overlay_array[..., 1] = base  # G
            overlay_array[..., 2] = base  # B
            
            # Calculate red boost and green/blue reduction with bounds checking
            red_channel = overlay_array[..., 0].astype(np.int16) + diff  # Use int16 to prevent overflow
            green_blue_reduction = diff // 2
            green_channel = overlay_array[..., 1].astype(np.int16) - green_blue_reduction
            blue_channel = overlay_array[..., 2].astype(np.int16) - green_blue_reduction
            
            # Clip values to valid range and assign back to overlay array
            overlay_array[..., 0] = np.clip(red_channel, 0, 255).astype(np.uint8)
            overlay_array[..., 1] = np.clip(green_channel, 0, 255).astype(np.uint8)
            overlay_array[..., 2] = np.clip(blue_channel, 0, 255).astype(np.uint8)

        elif overlay_type == "HSV Difference":
            # Create HSV array (using float for calculations)
            hsv_array = np.zeros((ref_enhanced.shape[0], ref_enhanced.shape[1], 3), dtype=float)
            combined_mask = self.ref_mask_array.astype(float) * shifted_template_mask

            # Calculate absolute difference between enhanced images
            diff = np.abs(ref_enhanced.astype(float) - template_enhanced.astype(float)) * combined_mask
            #diff_normalized = diff / diff.max()  # Normalize to 0-1 range
            bin_mask = combined_mask > 0.5
            p1, p99 = np.percentile(diff[bin_mask], (0.5, 99.5))
            diff_normalized = np.clip((diff - p1) / (p99 - p1), 0, 1)
            
            # Set Value/Luminance to reference image (normalized to 0-1)
            hsv_array[..., 2] = ref_enhanced.astype(float) / 255.0
            
            # Set Hue based on difference (0.66 for blue to 0 for red)
            # Small differences -> blue (0.66)
            # Large differences -> red (0)
            hsv_array[..., 0] = 0.66 * (1 - diff_normalized)
            
            # Set Saturation based on difference
            # No difference -> no saturation (grayscale)
            # Large difference -> full saturation (colorful)
            hsv_array[..., 1] = diff_normalized
            
            # Convert HSV to RGB
            rgb_array = np.clip(color.hsv2rgb(hsv_array) * 255, 0, 255).astype(np.uint8)
            overlay_array = rgb_array

        # Convert the overlay array to QImage
        bytes_per_line = 3 * overlay_array.shape[1]
        overlay_qimage = QtGui.QImage(overlay_array.tobytes(), overlay_array.shape[1], overlay_array.shape[0],
                                    bytes_per_line, QtGui.QImage.Format_RGB888)

        if overlay_qimage.isNull():
            logging.error("Failed to create QImage from overlay array.")
            self.overlay_image_label.setText("Overlay Creation Failed")
            return

        # Convert QImage to QPixmap and display
        overlay_pixmap = QPixmap.fromImage(overlay_qimage)
        self.overlay_image_label.setPixmap(overlay_pixmap)

    def resizeEvent(self, event):
        """
        Override the resizeEvent to rescale the overlay pixmap when the window is resized.
        """
        super(MainWindow, self).resizeEvent(event)
        if self.template_image_array is not None:
            self.update_overlay(self.template_image_array, self.template_mask_array)
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
        self.deltaX_edit.setText(f"{self.config['current_deltax']:.3f}")
        self.deltaY_edit.setText(f"{self.config['current_deltay']:.3f}")
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


    def apply_shift_to_template(self, total_shift_x, total_shift_y):
         # Apply shift to the template image
    
        return rs.apply_shift_to_template(total_shift_x, total_shift_y, 
                                            self.template_image_array, 
                                        self.template_mask_array)
           

    # Add new method to compute metrics
    def compute_registration_metrics(self, shifted_template, shifted_mask):
        """
        Compute various registration metrics between reference and shifted template images.
        """
        if self.ref_image_array is None or shifted_template is None:
            return None, None, None

        # Compute SSIM using original intensity values
        ssim_val = rm.compute_ssim(self.ref_image_array, shifted_template, 
                                   self.ref_mask_array, shifted_mask)
        
        # Compute NMI using original intensity values
        nmi_val = rm.compute_mi(self.ref_image_array, shifted_template, 
                                self.ref_mask_array, shifted_mask)
        
        # NCC typically benefits from normalization, so keep as is
        ncc_val = rm.compute_masked_ncc(self.ref_image_array, shifted_template, 
                                        self.ref_mask_array, shifted_mask)

        return ssim_val, nmi_val, ncc_val
    
    def apply_shift_and_update_overlay(self):
        """
        Shift the template image and its mask based on current_deltax and current_deltay,
        update the overlay image, and recompute ml1e and perceptual loss.
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
        print(f"Total shift: X={total_shift_x}, Y={total_shift_y}")
        self.shift_x_history.append(total_shift_x)
        self.shift_y_history.append(total_shift_y)

        # logging.debug(f"Applying total shift: Delta X={total_shift_x}, Delta Y={total_shift_y}")

        shifted_image, shifted_mask = self.apply_shift_to_template(total_shift_x, total_shift_y)
        # logging.debug("Applied shift to template mask and re-binarized.")

        # Update the display pixmap with the shifted image
        #shifted_display_arr = ppi.contrast_stretch_8bit(shifted_image)  # Apply contrast stretching for display
        #self.template_display_pixmap = self.array_to_qpixmap(shifted_display_arr, is_grayscale=True)
        # logging.debug("Updated template_display_pixmap with shifted and contrast-stretched image.")

        # Update the overlay
        self.update_overlay(shifted_image, shifted_mask)
        # logging.debug("Updated overlay after shifting.")

        # Compute Losses
        ml1e = self.compute_ml1e(shifted_image, shifted_mask)
         # Append to loss history
        self.ml1e_history.append(ml1e)
        pl, self.diff_features = self.compute_perceptual_loss(shifted_image, shifted_mask)
        # logging.debug(f"Computed ml1e: {ml1e}, Perceptual Loss: {pl}")

       
        selected_choice = self.layer_dropdown.currentText()
        '''
        if selected_choice == "Heatmap" or selected_choice == "Sum of Layers":
            pass
        else:
            selected_layer = selected_choice.split(' ')[1]  # Extract layer number, e.g., "Layer 5"
            #print(f"selected_layer = {selected_layer}")
            if self.diff_features is not None:
                activations = self.diff_features[selected_layer]
                pl = np.sum(activations)
        '''

        self.pl_history.append(pl)

        # After computing shifted_image and shifted_mask
        ssim_val, nmi_val, ncc_val = self.compute_registration_metrics(shifted_image, shifted_mask)
        #print(f"ssim_val = {ssim_val}, nmi_val = {nmi_val}, ncc_val = {ncc_val}")
        # Append to histories
        self.ssim_history.append(ssim_val)
        self.nmi_history.append(nmi_val)
        self.ncc_history.append(ncc_val)
    
        # Update plots
        self.update_plots()
        # logging.debug("Updated ml1e and Perceptual Loss plots.")

        # Update difference heatmap
        self.compute_and_display_heatmap(shifted_image, shifted_mask)
        # logging.debug("Computed and displayed difference heatmap.")

    def apply_best_shift(self):
        """
        Apply the shift that was voted best across multiple metrics.
        SSIM, NMI, NCC: higher is better
        ML1E, PL: lower is better
        """
        if len(self.pl_history) == 0:
            QtWidgets.QMessageBox.warning(self, "No History", 
                                        "No shifts have been applied yet.")
            return

        # Get indices of best values for each metric
        best_indices = {
            # For metrics where lower is better (minimize)
            'ML1E': np.argmin(self.ml1e_history),
            'PL': np.argmin(self.pl_history),
            # For metrics where higher is better (maximize)
            'SSIM': np.argmax(self.ssim_history),
            'NMI': np.argmax(self.nmi_history),
            'NCC': np.argmax(self.ncc_history)
        }

        # Count votes for each index
        vote_counts = {}
        for metric, index in best_indices.items():
            vote_counts[index] = vote_counts.get(index, 0) + 1
            print(f"{metric} votes for shift index {index}")

        # Find the index with the most votes
        winning_index = max(vote_counts.items(), key=lambda x: x[1])[0]
        winning_votes = vote_counts[winning_index]
        
        # Get the corresponding shifts
        best_shift_x = self.shift_x_history[winning_index]
        best_shift_y = self.shift_y_history[winning_index]

        # Print detailed results
        print(f"\nVoting Results:")
        print(f"Winning shift index {winning_index} with {winning_votes} votes")
        print(f"Applying shift: X={best_shift_x:.3f}, Y={best_shift_y:.3f}")
        print(f"Metric values at winning shift:")
        print(f"ML1E: {self.ml1e_history[winning_index]:.3f}")
        print(f"PL: {self.pl_history[winning_index]:.3f}")
        print(f"SSIM: {self.ssim_history[winning_index]:.3f}")
        print(f"NMI: {self.nmi_history[winning_index]:.3f}")
        print(f"NCC: {self.ncc_history[winning_index]:.3f}")

        # Update the current shifts
        self.config["current_deltax"] = best_shift_x
        self.config["current_deltay"] = best_shift_y

        # Update the shift fields
        self.deltaX_edit.setText(f"{best_shift_x:.3f}")
        self.deltaY_edit.setText(f"{best_shift_y:.3f}")

        # Apply the shift and update the display
        self.apply_shift_and_update_overlay()


    def compute_ml1e(self, shifted_template, shifted_template_mask):
        """
        Compute Mean Squared Error between reference and shifted template images.
        """
        ml1e = rm.compute_ml1e(self.ref_image_array, shifted_template, 
                               self.ref_mask_array, shifted_template_mask)
        return ml1e

    def compute_perceptual_loss(self, shifted_template, shifted_template_mask):
        """
        Compute Perceptual Loss between reference and shifted template images.
        Placeholder function. Replace with actual implementation.
        """
        pl, diff_features = rm.compute_perceptual_loss(
                                         self.ref_image_array, shifted_template,
                                         self.ref_mask_array, shifted_template_mask,
                                         self.perceptual_loss_model)
        return pl, diff_features

    def compute_and_apply_shift(self):
        """
        Apply the current shift if non-zero, then compute a new shift using phase_cross_correlation,
        and update all displays with the new shift.
        """
        if self.template_image_array is None or self.ref_image_array is None:
            QtWidgets.QMessageBox.warning(self, "Missing Images", "Both reference and template images are required.")
            return

        # Step 1: Apply current shift (if non-zero) to the template image
        shift_x = self.config["current_deltax"]
        shift_y = self.config["current_deltay"]
        #print(f"Current shift: X={shift_x}, Y={shift_y}")
        if shift_x != 0.0 or shift_y != 0.0:
            shifted_image, shifted_mask = self.apply_shift_to_template(shift_x, shift_y)
        else:
            shifted_image = self.template_image_array
            shifted_mask = self.template_mask_array

        
        selected_choice = self.coreg_dropdown.currentText()
        if selected_choice == "Fourier":
            shift_yx = rs.compute_shift_pcc(self.ref_image_array, shifted_image, self.ref_mask_array, shifted_mask)
        elif selected_choice == "Point Matching":
            shift_yx = rs.compute_shift_point_matching(self.ref_image_array, shifted_image)
        elif selected_choice == "NCC":
            shift_yx = rs.compute_shift_ncc(self.ref_image_array, shifted_image, self.ref_mask_array, shifted_mask)
        elif selected_choice == "Perceptual Loss":
            shift_yx = rs.compute_shift_pl(self.perceptual_loss_model, self.ref_image_array, shifted_image, self.ref_mask_array, shifted_mask)
        elif selected_choice == "SSIM":
            shift_yx = rs.compute_shift_with_metric(rm.compute_ssim, minimize=False, 
                                                    ref_image=self.ref_image_array, template_image=shifted_image, 
                                                    ref_mask=self.ref_mask_array, template_mask=shifted_mask)
        elif selected_choice == "Mutual Information":
            shift_yx = rs.compute_shift_with_metric(rm.compute_mi, minimize=False, 
                                                    ref_image=self.ref_image_array, template_image=shifted_image, 
                                                    ref_mask=self.ref_mask_array, template_mask=shifted_mask)
        elif selected_choice == "Mean L1 Error":
            shift_yx = rs.compute_shift_with_metric(rm.compute_ml1e, minimize=True, 
                                                    ref_image=self.ref_image_array, template_image=shifted_image, 
                                                    ref_mask=self.ref_mask_array, template_mask=shifted_mask)
            '''
        elif selected_choice == "ILK Optical Flow":
            shift_yx = rs.compute_shift_ilk_optical_flow(self.ref_image_array, shifted_image, self.ref_mask_array, shifted_mask)
        elif selected_choice == "TVL1 Optical Flow":
            shift_yx = rs.compute_shift_tvl1_optical_flow(self.ref_image_array, shifted_image, self.ref_mask_array, shifted_mask)
        '''    
        else:
            shift_yx = [0, 0]

        # Step 2: Compute the new shift using phase_cross_correlation
        #shift_yx = ppi.compute_shift(self.ref_image_array, shifted_image, self.ref_mask_array, shifted_mask)
      
        # Step 3: Add the computed shift to the current shift values
        new_shift_x = self.config["current_deltax"] + shift_yx[1]
        new_shift_y = self.config["current_deltay"] + shift_yx[0]

        # Update the total shift in the configuration
        self.config["current_deltax"] = new_shift_x
        self.config["current_deltay"] = new_shift_y
        print(f"New shift: X={new_shift_x}, Y={new_shift_y}")

        # Update the shift fields at the top of the interface
        self.deltaX_edit.setText(f"{self.config['current_deltax']:.3f}")
        self.deltaY_edit.setText(f"{self.config['current_deltay']:.3f}")

        # Step 4: Apply the new shift and update all displays
        self.apply_shift_and_update_overlay()

        # Step 5: Recompute ml1e, Perceptual Loss, Heatmap, and Graphs
        #self.compute_and_display_heatmap(shifted_image, shifted_mask)
        self.update_plots()


    def update_plots(self):
        """
        Update the plots with new ml1e and perceptual loss values, and show shifts as x-axis labels.
        """
        shift_steps = range(len(self.ml1e_history))  # X-axis positions

        # Clear the plots before updating
        self.ml1e_ax.clear()
        self.metrics_ax.clear()
        # Clear the twin axis if it exists
        for ax in self.ml1e_fig.axes:
            if ax != self.ml1e_ax:
                self.ml1e_fig.delaxes(ax)
        
        # Only plot last N points to keep plotting fast
        max_history = 20
        if len(shift_steps) > max_history:
            start_idx = -max_history
            shift_steps = range(max_history)
        else:
            start_idx = None

        # Create new twin axis for perceptual loss
        pl_ax = self.ml1e_ax.twinx()
        
        # Plot ml1e over shifts (red, left axis)
        ml1e_line = self.ml1e_ax.plot(shift_steps, self.ml1e_history[start_idx:], 'r-', label='ml1e')
        self.ml1e_ax.set_ylabel("ml1e", color='r')
        self.ml1e_ax.tick_params(axis='y', labelcolor='r')

        # Plot Perceptual Loss over shifts (blue, right axis)
        pl_line = pl_ax.plot(shift_steps, self.pl_history[start_idx:], 'b-', label='Perceptual Loss')
        pl_ax.set_ylabel("Perceptual Loss", color='b')
        pl_ax.tick_params(axis='y', labelcolor='b')

        # Add combined title
        self.ml1e_ax.set_title("ml1e and Perceptual Loss over Shifts")

        # Add combined legend
        lines = ml1e_line + pl_line
        labels = [line.get_label() for line in lines]
        self.ml1e_ax.legend(lines, labels, loc='upper right')

        # Create x-axis labels showing the shifts
        x_labels = [f"{x:.3f},{y:.3f}" for x, y in zip(self.shift_x_history[start_idx:], self.shift_y_history[start_idx:])]
        
        # Reduce number of x-tick labels
        step = max(len(x_labels) // 5, 1)  # Show ~5 labels
        self.ml1e_ax.set_xticks(shift_steps[::step])
        self.ml1e_ax.set_xticklabels([x_labels[i] for i in range(0, len(x_labels), step)], rotation=45, ha='right')
        
        # Add x-axis label
        self.ml1e_ax.set_xlabel("Shifts (x, y)")

        # Plot normalized metrics in the second graph with reduced points
        if self.ssim_history:
            self.metrics_ax.plot(shift_steps, self.ssim_history[start_idx:], 'g-', label='SSIM')
        if self.nmi_history:
            self.metrics_ax.plot(shift_steps, self.nmi_history[start_idx:], 'y-', label='NMI')
        if self.ncc_history:
            self.metrics_ax.plot(shift_steps, self.ncc_history[start_idx:], 'm-', label='NCC')

        # Use same x-axis labeling scheme for metrics plot
        self.metrics_ax.set_xticks(shift_steps[::step])
        self.metrics_ax.set_xticklabels([x_labels[i] for i in range(0, len(x_labels), step)], rotation=45, ha='right')
        
        self.metrics_ax.set_xlabel("Shifts (x, y)")
        self.metrics_ax.set_ylabel("Metric Value")
        self.metrics_ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        # Use tight_layout only when needed (e.g., on window resize)
        # self.ml1e_fig.tight_layout()
        # self.metrics_fig.tight_layout()

        # Use draw_idle() instead of draw() for better performance
        self.ml1e_canvas.draw_idle()
        self.metrics_canvas.draw_idle()


    def compute_sum_of_layers(self):
        """
        Compute the sum of all layers in diff_features dictionary.
        This is used to visualize the sum of all layers in VGG feature space.
        """
        summed_activations = ppi.compute_sum_of_layers(self.diff_features, normalize=False)

        return summed_activations

    def compute_and_display_heatmap(self, shifted_template_image, shifted_template_mask):
        #print(f"compute_and_display_heatmap")
        """
        Compute and display the masked difference heatmap between reference and shifted template images.

        Args:
            shifted_template_image (np.ndarray): Shifted template image array.
            shifted_template_mask (np.ndarray): Shifted template mask array.
        """
        selected_choice = self.layer_dropdown.currentText()
        if selected_choice == "Heatmap":
            self.compute_difference_heatmap(shifted_template_image, shifted_template_mask)
        elif selected_choice == "Sum of Layers":
            summed_activations = self.compute_sum_of_layers()
            self.heatmap_canvas.plot_heatmap(summed_activations, mask=None, cmap='jet')
        else:
            selected_layer = selected_choice.split(' ')[1]  # Extract layer number, e.g., "Layer 5"
            #print(f"selected_layer = {selected_layer}")
            if self.diff_features is not None:
                #print(f"diff_features keys = {self.diff_features.keys()}")
                activations = self.diff_features[selected_layer]
                self.heatmap_canvas.plot_heatmap(activations, mask=None, cmap='jet')
                #self.heatmap_canvas.plot_heatmap(activations)
        

    def compute_difference_heatmap(self, shifted_template_image, shifted_template_mask):
        #print(f"compute_difference_heatmap")
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
        #print(f"compute_difference_heatmap sum: {np.sum(diff_array)}")
        # Plot the heatmap using the updated HeatmapCanvas
        self.heatmap_canvas.plot_heatmap(diff_array, mask=combined_mask, cmap='jet')


    
    def update_visualization_choice(self):
        """
        Update the visualization choice based on the selected item in the dropdown menu.
        """
        selected_choice = self.layer_dropdown.currentText()
        
        if selected_choice == "Heatmap":
            shift_x = self.config["current_deltax"]
            shift_y = self.config["current_deltay"]
            shifted_image, shifted_mask = self.apply_shift_to_template(shift_x, shift_y)
            self.compute_and_display_heatmap(shifted_image, shifted_mask)
        elif selected_choice == "Sum of Layers":
            summed_activations = self.compute_sum_of_layers()
            self.heatmap_canvas.plot_heatmap(summed_activations, mask=None, cmap='jet') 
        else:
            selected_layer = selected_choice.split(' ')[1]  # Extract layer number, e.g., "Layer 5"
            #print(f"selected_layer = {selected_layer}")
            if self.diff_features is not None:
                #print(f"diff_features keys = {self.diff_features.keys()}")
                activations = self.diff_features[selected_layer]
                self.heatmap_canvas.plot_heatmap(activations, mask=None, cmap='jet')
        
          
    
    def handle_overlay_change(self):
        """
        Wrapper function to handle overlay type changes and call update_overlay with correct arguments.
        """
        if hasattr(self, 'template_image_array') and self.template_image_array is not None:
            # Get the current shifts
            total_shift_x = self.config["current_deltax"]
            total_shift_y = self.config["current_deltay"]
            
            # Apply current shift to get shifted template
            shifted_image, shifted_mask = self.apply_shift_to_template(total_shift_x, total_shift_y)
            
            # Update the overlay with shifted template
            self.update_overlay(shifted_image, shifted_mask)

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    # logging.debug("Application window displayed.")
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
