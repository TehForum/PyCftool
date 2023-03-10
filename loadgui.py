from PySide6 import QtCore, QtUiTools
from mplwidget import MplWidget

def loadUiWidget(uifilename, parent=None):
    loader = QtUiTools.QUiLoader()
    uifile = QtCore.QFile(uifilename)
    uifile.open(QtCore.QFile.ReadOnly)
    loader.registerCustomWidget(MplWidget)
    ui = loader.load(uifile, parent)
    uifile.close()
    return ui