import sys
from PyQt5 import QtWidgets, QtGui, QtCore

class MainWindow(QtWidgets.QMainWindow):
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
    
    # ----- Dummy methods to be implemented later -----
    def load_reference_image(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Reference Image")
        if fname:
            # TODO: Load reference image logic will go here
            pass

    def load_reference_mask(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Reference Mask")
        if fname:
            # TODO: Load reference mask logic will go here
            pass

    def load_template_image(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Template Image")
        if fname:
            # TODO: Load template image logic will go here
            pass

    def load_template_mask(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Template Mask")
        if fname:
            # TODO: Load template mask logic will go here
            pass

    def keyPressEvent(self, event):
        # We'll handle arrow keys and updating the shift here later
        super(MainWindow, self).keyPressEvent(event)


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
