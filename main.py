import sys
import os
import glob
import cv2
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import  QWidget, QTextEdit, QAction, QApplication, QFileDialog, QGridLayout, QLabel,QMenuBar, QToolBar, QVBoxLayout
from PyQt5.QtGui import QIcon, QPixmap
from numrec import num_rec




class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.width = 600
        self.height = 800
    def initUI(self):

        self.setWindowIcon(QIcon('icon/container.png'))
        self.textEdit = QTextEdit(self)
        self.label = QLabel()
        self.label.resize(600, 800)
        grid = QGridLayout()
        grid.setSpacing(1)


        openSingle = QAction(QIcon('icon/image.png'), '打开一张照片', self)
        openSingle.setStatusTip('打开一张图片')
        openSingle.triggered.connect(self.openFileNameDialog)

        openMulti = QAction(QIcon('icon/open.png'), '打开照片文件夹', self)
        openMulti.setStatusTip('打开照片文件夹')
        openMulti.triggered.connect(self.openFolderDialog)

        clearText = QAction(QIcon('icon/clear.png'), '清空', self)
        clearText.setStatusTip('清空文字')
        clearText.triggered.connect(self.clearText)
        #self.statusBar()

        self.menubar = QMenuBar(self)
        fileMenu = self.menubar.addMenu('&文件')
        fileMenu.addAction(openSingle)
        fileMenu.addAction(openMulti)

        self.toolBar = QToolBar()
        self.toolBar.addAction(openSingle)
        self.toolBar.addAction(openMulti)
        self.toolBar.addAction(clearText)
        '''
        toolbar = self.toolBar.addToolBar('打开一张照片')
        toolbar.addAction(openSingle)
        toolbar = self.toolBar.addToolBar('打开照片文件夹')
        toolbar.addAction(openMulti)
        '''
        grid.addWidget(self.menubar, 1, 0)
        grid.addWidget(self.toolBar, 2, 0)

        grid.addWidget(self.textEdit, 3, 0, 1, 1)
        grid.addWidget(self.label,    3, 1, 1, 1)


        self.setLayout(grid)
        self.setGeometry(300, 300, 800,600)
        self.setWindowTitle('集装箱号码识别 v1')
        self.show()

    def clearText(self):
        self.textEdit.clear()

    def openFileNameDialog(self):
        self.textEdit.append("识别中...")
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                  "jpg Files (*.jpg)", options=options)

        if fileName:
            pixmap1 = QPixmap(fileName)
            pixmap1 = pixmap1.scaled(self.width*2/3, self.height*2/3)
            self.label.setPixmap(pixmap1)
            rec_results = num_rec(fileName)
            self.textEdit.append(os.path.basename(os.path.normpath(fileName)) + "柜号： "+rec_results)

    def openFolderDialog(self):
        self.textEdit.append("识别中...")
        folder = str(QFileDialog.getExistingDirectory(self, "选择图片文件夹"))
        animation_time = QTimer()
        animation_time.setSingleShot(False)
        #animation_time.timeout.connect(self.)
        for file in glob.glob(folder+"/*.jpg"):
            #pixmap1 = QPixmap(file)
            #pixmap1.scaled(self.width, self.height)
            #self.label.setPixmap(pixmap1)
            #self.label.setMinimumSize(1, 1)
            rec_results = num_rec(file)
            self.textEdit.append(os.path.basename(os.path.normpath(file)) + "柜号： "+ rec_results)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())
