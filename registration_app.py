import sys
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QPainter, QColor
import json
import os
import numpy as np


DEFAULT_CONFIG = {
    "reference_image": "",
    "reference_mask": "",
    "template_image": "",
    "template_mask": "",
    "current_deltax": 0.0,
    "current_deltay": 0.0,
    "shift_step_x": 0.1,
    "shift_step_y": 0.1
}



class MainWindow(QtWidgets.QMainWindow):

    def load_config(self, config_path='config.json'):
        if not os.path.exists(config_path):
            print(f"Config file '{config_path}' not found. Using default settings.")
            return DEFAULT_CONFIG.copy()
    
        with open(config_path, 'r') as f:
            try:
                user_config = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}. Using default settings.")
                return DEFAULT_CONFIG.copy()
        
        # Merge user_config with DEFAULT_CONFIG, preferring user_config values
        config = DEFAULT_CONFIG.copy()
        config.update({k: v for k, v in user_config.items() if k in DEFAULT_CONFIG})
        
        # Optionally, warn about any unknown fields
        unknown_fields = set(user_config.keys()) - set(DEFAULT_CONFIG.keys())
        if unknown_fields:
            print(f"Warning: Unknown config fields detected and ignored: {unknown_fields}")
    
        #save config
        self.config = config
        # Populate shift fields with config values
        self.deltaX_edit.setText(str(self.config["current_deltax"]))
        self.deltaY_edit.setText(str(self.config["current_deltay"]))
        self.shift_step_x_edit.setText(str(self.config["shift_step_x"]))
        self.shift_step_y_edit.setText(str(self.config["shift_step_y"]))
        
        # Automatically load images if paths are provided
        if self.config["reference_image"]:
            self.load_reference_image(self.config["reference_image"])
        if self.config["reference_mask"]:
            self.load_reference_mask(self.config["reference_mask"])
        if self.config["template_image"]:
            self.load_template_image(self.config["template_image"])
        if self.config["template_mask"]:
            self.load_template_mask(self.config["template_mask"])


    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        # ----- Menu Bar -----
        menubar = self.menuBar()
        
        file_menu = menubar.addMenu("File")
        
        load_ref_action = QtWidgets.QAction("Load Reference Image", self)
        load_ref_action.triggered.connect(self.load_reference_image)
        file_menu.addAction(load_ref_action)
        
        load_ref_mask_action = QtWidgets.QAction("Load Reference Mask", self)
        load_ref_mask_action.triggered.connect(self.load_reference_mask)
        file_menu.addAction(load_ref_mask_action)
        
        load_template_action = QtWidgets.QAction("Load Template Image", self)
        load_template_action.triggered.connect(self.load_template_image)
        file_menu.addAction(load_template_action)
        
        load_template_mask_action = QtWidgets.QAction("Load Template Mask", self)
        load_template_mask_action.triggered.connect(self.load_template_mask)
        file_menu.addAction(load_template_mask_action)
        
        # Create central widget and main layout
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        
        # ----- Controls for current deltaX, deltaY -----
        current_shift_layout = QtWidgets.QHBoxLayout()
        self.deltaX_edit = QtWidgets.QLineEdit()
        self.deltaY_edit = QtWidgets.QLineEdit()
        
        self.deltaX_edit.setPlaceholderText("Current Delta X")
        self.deltaY_edit.setPlaceholderText("Current Delta Y")
        
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
        
        shift_step_layout.addWidget(QtWidgets.QLabel("Shift Step X:"))
        shift_step_layout.addWidget(self.shift_step_x_edit)
        shift_step_layout.addWidget(QtWidgets.QLabel("Shift Step Y:"))
        shift_step_layout.addWidget(self.shift_step_y_edit)
        
        main_layout.addLayout(shift_step_layout)
        
        # Image display layout
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
        
        # Graphs layout
        graphs_layout = QtWidgets.QHBoxLayout()
        
        # Placeholder for MSE graph
        self.mse_graph_label = QtWidgets.QLabel("MSE Graph Here")
        self.mse_graph_label.setAlignment(QtCore.Qt.AlignCenter)
        self.mse_graph_label.setStyleSheet("border: 1px solid gray;")
        
        # Placeholder for Perceptual Loss graph
        self.pl_graph_label = QtWidgets.QLabel("Perceptual Loss Graph Here")
        self.pl_graph_label.setAlignment(QtCore.Qt.AlignCenter)
        self.pl_graph_label.setStyleSheet("border: 1px solid gray;")
        
        graphs_layout.addWidget(self.mse_graph_label)
        graphs_layout.addWidget(self.pl_graph_label)
        
        main_layout.addLayout(graphs_layout)
        
        # Set a window title and a reasonable initial size
        self.setWindowTitle("Interactive Image Alignment Tool")
        self.resize(900, 700)
        
        # For now, just focus policy to capture keyboard events (we'll handle events later)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        
        # Initialize pixmaps and masks
        self.ref_pixmap = None
        self.template_pixmap = None
        self.ref_mask = None
        self.template_mask = None

        self.load_config()
        
    # ----- Dummy methods to be implemented later -----
    def load_reference_image(self, filepath=None):
        if not filepath:
            fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Reference Image", "", "Image Files (*.png *.jpg *.bmp)")
            if not fname:
                return
            filepath = fname
        self.ref_pixmap = QPixmap(filepath)
        self.update_overlay()
    
    def load_reference_mask(self, filepath=None):
        if not filepath:
            fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Reference Mask", "", "Image Files (*.png *.jpg *.bmp)")
            if not fname:
                return
            filepath = fname
        self.ref_mask = QPixmap(filepath)
        # TODO: Convert mask to binary format if necessary
        self.update_overlay()
    
    def load_template_image(self, filepath=None):
        if not filepath:
            fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Template Image", "", "Image Files (*.png *.jpg *.bmp)")
            if not fname:
                return
            filepath = fname
        self.template_pixmap = QPixmap(filepath)
        self.update_overlay()
    
    def load_template_mask(self, filepath=None):
        if not filepath:
            fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Template Mask", "", "Image Files (*.png *.jpg *.bmp)")
            if not fname:
                return
            filepath = fname
        self.template_mask = QPixmap(filepath)
        # TODO: Convert mask to binary format if necessary
        self.update_overlay()
    
    def create_transparent_overlay(self, ref_pixmap, template_pixmap, alpha=0.5):
        if ref_pixmap is None or template_pixmap is None:
            return QPixmap()
        
        # Ensure both pixmaps are the same size
        if ref_pixmap.size() != template_pixmap.size():
            # Optionally, scale one to match the other or handle resizing
            min_width = min(ref_pixmap.width(), template_pixmap.width())
            min_height = min(ref_pixmap.height(), template_pixmap.height())
            ref_pixmap = ref_pixmap.scaled(min_width, min_height, QtCore.Qt.KeepAspectRatio)
            template_pixmap = template_pixmap.scaled(min_width, min_height, QtCore.Qt.KeepAspectRatio)
        
        # Create a new pixmap with the same size
        overlay = QPixmap(ref_pixmap.size())
        overlay.fill(QtCore.Qt.transparent)  # Start with a transparent pixmap
        
        painter = QPainter(overlay)
        
        # Draw the reference image
        painter.drawPixmap(0, 0, ref_pixmap)
        
        # Set the transparency for the template image
        painter.setOpacity(alpha)
        
        # Draw the template image on top
        painter.drawPixmap(0, 0, template_pixmap)
        
        painter.end()
        
        return overlay

    def create_color_coded_overlay(self, ref_pixmap, template_pixmap):
        """
        Create a color-coded overlay where the reference image is mapped to the red channel
        and the template image is mapped to the green channel.

        Parameters:
            ref_pixmap (QPixmap): QPixmap of the reference image.
            template_pixmap (QPixmap): QPixmap of the template image.

        Returns:
            QPixmap: Color-coded overlay pixmap.
        """
        if ref_pixmap is None or template_pixmap is None:
            return QPixmap()

        # Ensure both pixmaps are the same size
        if ref_pixmap.size() != template_pixmap.size():
            # Optionally, scale one to match the other or handle resizing
            min_width = min(ref_pixmap.width(), template_pixmap.width())
            min_height = min(ref_pixmap.height(), template_pixmap.height())
            ref_pixmap = ref_pixmap.scaled(min_width, min_height, Qt.KeepAspectRatio)
            template_pixmap = template_pixmap.scaled(min_width, min_height, Qt.KeepAspectRatio)

        # Convert QPixmap to QImage
        ref_qimage = ref_pixmap.toImage().convertToFormat(QtGui.QImage.Format_RGB32)
        template_qimage = template_pixmap.toImage().convertToFormat(QtGui.QImage.Format_RGB32)

        # Convert QImage to NumPy arrays
        ref_buffer = ref_qimage.bits().asstring(ref_qimage.byteCount())
        ref_array = np.frombuffer(ref_buffer, dtype=np.uint8).reshape((ref_qimage.height(), ref_qimage.width(), 4))[:, :, :3]

        template_buffer = template_qimage.bits().asstring(template_qimage.byteCount())
        template_array = np.frombuffer(template_buffer, dtype=np.uint8).reshape((template_qimage.height(), template_qimage.width(), 4))[:, :, :3]

        # Initialize overlay array with zeros
        overlay_array = np.zeros_like(ref_array)

        # Assign reference image to red channel
        overlay_array[:, :, 0] = ref_array[:, :, 0]

        # Assign template image to green channel
        overlay_array[:, :, 1] = template_array[:, :, 1]

        # Assign template image to blue channel
        overlay_array[:, :, 2] = template_array[:, :, 1]

        # Convert NumPy array back to QImage
        height, width, channel = overlay_array.shape
        bytes_per_line = 3 * width
        overlay_qimage = QtGui.QImage(overlay_array.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888).copy()

        # Convert QImage back to QPixmap
        overlay_pixmap = QPixmap.fromImage(overlay_qimage)

        return overlay_pixmap


    def update_overlay(self):
        if self.ref_pixmap is None or self.template_pixmap is None:
            self.overlay_image_label.setText("Overlay Image Here")
            return
        
        overlay_pixmap = self.create_color_coded_overlay(self.ref_pixmap, self.template_pixmap)
        #self.create_transparent_overlay(self.ref_pixmap, self.template_pixmap, alpha=0.5)
        self.overlay_image_label.setPixmap(overlay_pixmap)
    
    def keyPressEvent(self, event):
        key = event.key()
        shift_x = float(self.shift_step_x_edit.text()) if self.shift_step_x_edit.text() else 0.0
        shift_y = float(self.shift_step_y_edit.text()) if self.shift_step_y_edit.text() else 0.0

        if key == Qt.Key_Up:
            self.config["current_deltay"] -= shift_y
        elif key == Qt.Key_Down:
            self.config["current_deltay"] += shift_y
        elif key == Qt.Key_Left:
            self.config["current_deltax"] -= shift_x
        elif key == Qt.Key_Right:
            self.config["current_deltax"] += shift_x
        else:
            super(MainWindow, self).keyPressEvent(event)
            return

        # Update the shift fields
        self.deltaX_edit.setText(str(self.config["current_deltax"]))
        self.deltaY_edit.setText(str(self.config["current_deltay"]))
        
        # Apply the shift to the template image and update the overlay
        self.apply_shift_and_update_overlay()



def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
