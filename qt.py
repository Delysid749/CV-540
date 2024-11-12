import sys
import cv2
import numpy as np
import subprocess
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal

class Worker(QThread):
    finished = pyqtSignal(str, str)

    def __init__(self, image_path):
        super().__init__()
        self.image_path = image_path

    def run(self):
        depth_map_path = f"./outputs/{self.image_path.split('/')[-1].split('.')[0]}.png"
        ply_file = f"./outputs/{self.image_path.split('/')[-1].split('.')[0]}.ply"

        subprocess.run(["python", "run.py", "--encoder", "vitl", "--load-from",
                        "checkpoints/depth_anything_v2_metric_vkitti_vitl.pth", "--max-depth", "80", "--pred-only", "--img-path",
                        self.image_path, "--outdir", "./outputs"])
        subprocess.run(["python", "depth_to_pointcloud.py", "--encoder", "vitl", "--load-from",
                        "checkpoints/depth_anything_v2_metric_vkitti_vitl.pth", "--max-depth", "20", "--img-path",
                        self.image_path, "--outdir", "./outputs"])

        self.finished.emit(depth_map_path, ply_file)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Depth Map Estimation")
        self.setGeometry(100, 100, 1000, 600)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.image_label = QLabel(self)
        self.depth_map_label = QLabel(self)
        self.push_button_select_image = QPushButton("Select Image", self)
        self.push_button_fill_plane = QPushButton("Fill With Plane", self)
        self.push_button_fill_mesh = QPushButton("Fill With Mesh", self)
        self.push_button_fill_curvature = QPushButton("Fill With Curvature", self)

        top_layout = QHBoxLayout()
        top_layout.addWidget(self.push_button_select_image)
        top_layout.addWidget(self.push_button_fill_plane)
        top_layout.addWidget(self.push_button_fill_mesh)
        top_layout.addWidget(self.push_button_fill_curvature)

        image_layout = QHBoxLayout()
        image_layout.addWidget(self.image_label)
        image_layout.addWidget(self.depth_map_label)

        main_layout = QVBoxLayout()
        main_layout.addLayout(top_layout)
        main_layout.addLayout(image_layout)

        self.central_widget.setLayout(main_layout)

        self.push_button_select_image.clicked.connect(self.on_select_image_clicked)
        self.push_button_fill_plane.clicked.connect(self.on_fill_plane_clicked)
        self.push_button_fill_mesh.clicked.connect(self.on_fill_mesh_clicked)
        self.push_button_fill_curvature.clicked.connect(self.on_fill_curvature_clicked)

        self.ply_file = None

    def on_select_image_clicked(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.bmp *.jpeg)")
        if file_name:
            self.display_image(file_name)
            self.worker = Worker(file_name)
            self.worker.finished.connect(self.on_processing_finished)
            self.worker.start()

    def display_image(self, image_path):
        image = QImage(image_path)
        self.image_label.setPixmap(QPixmap.fromImage(image).scaled(self.image_label.size(), Qt.KeepAspectRatio))

    def on_processing_finished(self, depth_map_path, ply_file):
        depth_map = cv2.imread(depth_map_path, cv2.IMREAD_COLOR)
        depth_map = cv2.cvtColor(depth_map, cv2.COLOR_BGR2RGB)
        if depth_map is None:
            print("Error: Depth map not found")
        else:
            print("Depth map found")
        if ply_file is None:
            print("Error: Point cloud not found")
        else:
            print("Point cloud found")

        self.display_depth_map(depth_map)
        self.ply_file = ply_file

    def display_depth_map(self, depth_map):
        depth_map_image = QImage(depth_map.data, depth_map.shape[1], depth_map.shape[0], depth_map.strides[0],
                                 QImage.Format_RGB888)
        self.depth_map_label.setPixmap(
            QPixmap.fromImage(depth_map_image).scaled(self.depth_map_label.size(), Qt.KeepAspectRatio))

    def on_fill_plane_clicked(self):
        if self.ply_file:
            subprocess.run(["python", "fill_plane.py", "--file", self.ply_file])
        else:
            print("Please import an image and wait for depth estimation and point cloud conversion.")

    def on_fill_mesh_clicked(self):
        if self.ply_file:
            subprocess.run(["python", "fill_mesh.py", "--file", self.ply_file])
        else:
            print("Please import an image and wait for depth estimation and point cloud conversion.")

    def on_fill_curvature_clicked(self):
        if self.ply_file:
            subprocess.run(["python", "fill_curvature.py", "--file", self.ply_file])
        else:
            print("Please import an image and wait for depth estimation and point cloud conversion.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())