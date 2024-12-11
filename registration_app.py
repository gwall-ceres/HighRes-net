import sys
from PyQt5 import QtWidgets, QtGui, QtCore

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        
        # Create central widget and main layout
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        
        # Top controls layout for deltaX and deltaY
        controls_layout = QtWidgets.QHBoxLayout()
        self.deltaX_edit = QtWidgets.QLineEdit()
        self.deltaY_edit = QtWidgets.QLineEdit()
        
        self.deltaX_edit.setPlaceholderText("Enter delta X")
        self.deltaY_edit.setPlaceholderText("Enter delta Y")
        
        controls_layout.addWidget(QtWidgets.QLabel("Delta X:"))
        controls_layout.addWidget(self.deltaX_edit)
        controls_layout.addWidget(QtWidgets.QLabel("Delta Y:"))
        controls_layout.addWidget(self.deltaY_edit)
        
        main_layout.addLayout(controls_layout)
        
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
        self.resize(800, 600)

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
