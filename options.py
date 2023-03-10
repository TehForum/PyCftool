from PySide6 import QtWidgets
from PySide6.QtWidgets import QApplication, QTableWidgetItem, QMainWindow, QPushButton, QDialog
from PySide6 import QtGui
import sys
from loadgui import loadUiWidget
import numpy as np


class Options_(QtWidgets.QDialog):
    def __init__(self,parameters,start,lower,upper):
        super().__init__()
        self.ui = loadUiWidget('cftooloptions.ui')
        self.r = QtWidgets.QVBoxLayout(self)
        self.r.addWidget(self.ui)

        self.parameters = parameters
        self.start = start
        self.lower = lower
        self.upper = upper
        self.create_Table()
        self.return_start = []
        self.return_lower = []
        self.return_upper = []
        self.return_bounds = ()

    def create_Table(self):
        for i in range(len(self.parameters)):
            self.ui.tableWidget.insertRow(i)
            self.ui.tableWidget.setItem(i, 0, QTableWidgetItem(self.parameters[i]))
            self.ui.tableWidget.setItem(i, 1, QTableWidgetItem(str(self.start[i])))
            self.ui.tableWidget.setItem(i, 2, QTableWidgetItem(str(self.lower[i])))
            self.ui.tableWidget.setItem(i, 3, QTableWidgetItem(str(self.upper[i])))

    def ignore(self):
        self.return_start = self.start
        self.return_bounds = (np.array(self.lower), np.array(self.upper))

    def closeEvent(self, event):
        try:
            for i in range(len(self.parameters)):
                start = self.ui.tableWidget.item(i,1).text()
                lower = self.ui.tableWidget.item(i,2).text()
                upper = self.ui.tableWidget.item(i,3).text()

                start = float(start)

                if '-inf' == lower:
                    lower = -np.inf
                else:
                    lower = float(lower)

                if 'inf' == upper:
                    upper = np.inf
                else:
                    upper = float(upper)
                self.return_start.append(start)
                self.return_lower.append(lower)
                self.return_upper.append(upper)
            self.return_bounds = (np.array(self.return_lower), np.array(self.return_upper))

        except:
            self.ignore()