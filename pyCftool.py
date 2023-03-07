import sys

import PySide6.QtGui
import numpy as np
import scipy.optimize
from PySide6.QtWidgets import QStackedWidget, QApplication, QFileDialog, QWidget, QMainWindow
from PySide6 import QtCore, QtGui, QtUiTools
from mplwidget import MplWidget
from scipy import optimize as optim
import re
from export import export_fit

class function():
    def __init__(self, eq):
        self.eq = eq
        self.eq_refine(eq)

    def eq_refine(self, s):
        key_words = ['+','-','*','/','^','(',')',',','',
                     'sqrt','sin','cos','exp','pi',
                     '0','1','2','3','4','5','6','7','8','9']
        numpy_words = key_words[9:14]
        list_eq = re.split(r'(\W+)', s)
        vars = []
        for i, char in enumerate(list_eq):
            if char not in key_words and re.match(r'[(+\-*^x)]', char) == None:
                vars.append(char)
                list_eq[i] = f'dictin[\'{char}\']'
            elif char in numpy_words:
                list_eq[i] = f'np.{char}'
            elif char == '^':
                list_eq[i] = '**'
        vars = list(set(vars))
        self.eq = ''.join(list_eq)
        self.vars = vars

    def func(self, x, *args):
        dictin = {}
        eq = self.eq
        for i in range(len(args)):
            dictin[self.vars[i]] = args[i]
        return eval(eq)


def loadUiWidget(uifilename, parent=None):
    loader = QtUiTools.QUiLoader()
    uifile = QtCore.QFile(uifilename)
    uifile.open(QtCore.QFile.ReadOnly)
    loader.registerCustomWidget(MplWidget)
    ui = loader.load(uifile, parent)
    uifile.close()
    return ui

class MainWindow(QMainWindow):
    def __init__(self,x,y,weights,local_variables):
        QMainWindow.__init__(self)
        self.ui = loadUiWidget('pycftoolGUI.ui')
        self.x = x
        self.y = y
        self.weights = weights

        self.local_vars = local_variables
        if len(local_variables) != 0:
            if isinstance(x,type(None)):
                self.ui.xDataComboBox.addItems([''])
            if isinstance(y,type(None)):
                self.ui.yDataComboBox.addItems([''])
            self.ui.weightsComboBox.addItems([''])
            self.localvarsInitialize()

        if not isinstance(x, type(None)):
            self.ComboBoxX_initialize()
            self.x_span = np.max(x) - np.min(x)
            self.x_fit = np.linspace(self.x[0] - self.x_span * 0.6, self.x[-1] + self.x_span * 0.6, self.x.shape[0])
            self.x_interpolate = np.linspace(x[0] - self.x_span * 0.6, x[-1] + self.x_span * 0.6, x.shape[0] * 100)
        if not isinstance(y, type(None)):
            self.ComboBoxY_initialize()
            self.y_fit_interpolate = np.zeros(y.shape[0] * 10)
            self.y_fit = np.zeros(y.shape[0])
            self.y_GOF = np.zeros(y.shape[0])
        if not isinstance(weights, type(None)):
            self.ComboBoxWeights_initialize()
        else:
            self.sigma = None

        self.ui.label_2.setText('')
        self.ui.textEdit.setVisible(False)
        self.setCentralWidget(self.ui)
        self.MplWidget = self.ui.mplwidget.canvas
        self.ui.pushButton_2.clicked.connect(self.Button_2)
        self.ui.pushButton.clicked.connect(self.manual_fit)

        self.ui.textEdit.textChanged.connect(self.customChanged)

        self.ui.checkBox.stateChanged.connect(self.autoFit)
        self.ui.checkBox_2.stateChanged.connect(self.interpolate)

        self.ui.comboBox.currentTextChanged.connect(self.equation_select)

        self.ui.degreeComboBox.currentTextChanged.connect(self.degreeBox)

        self.ui.xDataComboBox.currentTextChanged.connect(self.comboXData)
        self.ui.yDataComboBox.currentTextChanged.connect(self.comboYData)
        self.ui.weightsComboBox.currentTextChanged.connect(self.comboWeights)

        #Menubar
        self.ui.actionExport.triggered.connect(self.export_fit)

        #Variables
        self.order = 0
        self.eq_str = ''
        self.covars = 0
        self.params = ''
        self.paramvals = 0
        self.n_vars = 0
        self.p0 = []
        self.eq_setting = 'polynomial'

        #GOF Variables
        self.SSE= 0
        self.RSQR = 0
        self.adjRSQR = 0
        self.RMSE = 0
        self.chi_squared = 0
        #Intitalize
        self.initiate_fit()

########## INITIALIZATION
    def ComboBoxX_initialize(self):
        self.local_vars['X pycftool'] = self.x
        self.ui.xDataComboBox.addItems(['X pycftool'])
        self.ui.xDataComboBox.setCurrentText('X pycftool')

    def ComboBoxY_initialize(self):
        self.local_vars['Y pycftool'] = self.y
        self.ui.yDataComboBox.addItems(['Y pycftool'])
        self.ui.yDataComboBox.setCurrentText('Y pycftool')

    def ComboBoxWeights_initialize(self):
        self.sigma = 1 / weights
        self.local_vars['Weights pycftool'] = self.weights
        self.ui.weightsComboBox.addItems(['Weights pycftool'])
        self.ui.weightsComboBox.setCurrentText('Weights pycftool')


    def localvarsInitialize(self):
        self.ui.xDataComboBox.addItems(self.local_vars)
        self.ui.yDataComboBox.addItems(self.local_vars)
        self.ui.weightsComboBox.addItems(self.local_vars)


    ##########  MECHANICS
    def initiate_fit(self,bypass=False):
        if isinstance(self.x,np.ndarray) and isinstance(self.y,np.ndarray):
            self.ui.label_3.setText('')
            if self.eq_setting == 'polynomial':
                self.polynomial_fit()

            elif self.eq_setting == 'exponential':
                self.exponential_fit()
            elif self.eq_setting == 'resonance 1':
                self.resonance_fit_1()
            elif self.eq_setting == 'resonance 2':
                self.resonance_fit_2()
            elif self.eq_setting == 'custom equation':
                txtbox = self.ui.textEdit.toPlainText()
                if txtbox and self.customFilter(txtbox):
                    try:
                        self.custom_fit(txtbox)
                    except:
                        self.ui.label_3.setText('Could not understand input text')

            self.update(bypass=bypass)

    def update(self,bypass=False):
        if self.ui.checkBox.isChecked():
            self.GOD()
            self.update_graph()
            self.update_results()
        elif bypass:
            self.GOD()
            self.update_graph()
            self.update_results()

    def customFilter(self,s):
        #Count parenthasis:
        count = 0
        for l in s:
            if l == '(':
                count +=1
            elif l == ')':
                count -=1
        count_check = count == 0
        if not count_check:
            count_err = 'Missing Parenthesis!   '
        else:
            count_err = ''

        if 'np.' in s:
            np_check = False
            np_err = 'Do not use numpy functions in input text!'
        else:
            np_check = True
            np_err = ''
        self.ui.label_3.setText(count_err+np_err)
        return np_check*count_check

    def customChanged(self):
        if self.ui.checkBox.isChecked():
            self.initiate_fit()

    def export_fit(self):
        export_dir = QFileDialog.getSaveFileName(self, 'Save','','Python file (*.py)')[0]
        if export_dir != '':
            export_fit(self.eq_str,self.params, self.paramvals,self.weights, export_dir)
#####################s####

#########CHECKBOX######
    def autoFit(self,val):
        if self.ui.checkBox.isChecked():
            self.initiate_fit()
    def interpolate(self):
        self.initiate_fit()

#########BUTTONS########
    def manual_fit(self):
        self.initiate_fit(bypass=True)

    def Button_2(self):
        pass

############################

    ### COMBO BOXES
    def comboXData(self,value):
        self.x = self.local_vars[value]
        self.x_span = np.max(self.x)-np.min(self.x)
        self.x_fit = np.linspace(self.x[0]-self.x_span*0.1,self.x[-1]+self.x_span*0.1,self.x.shape[0])
        self.x_interpolate = np.linspace(x[0] - self.x_span * 0.6, x[-1] + self.x_span * 0.6, x.shape[0] * 100)
        self.initiate_fit()

    def comboYData(self,value):
        self.y = self.local_vars[value]
        self.y_fit = np.zeros(self.y.shape[0])
        self.y_fit_interpolate = np.zeros(self.y.shape[0] * 10)
        self.y_GOF = np.zeros(self.y.shape[0])
        self.initiate_fit()

    def comboWeights(self,value):
        if value != '':
            self.weights = self.local_vars[value]
            self.sigma = 1/self.weights
        else:
            self.weights = None
            self.sigma = None
        self.initiate_fit()

    def degreeBox(self,value):
        self.order = int(value)
        self.initiate_fit()

    def equation_select(self,value):
        self.eq_setting = value.lower()
        if value.lower() == 'polynomial':
            self.ui.textEdit.setVisible(False)
            self.ui.label_2.setText('')
            self.ui.degreeLabel.setVisible(True)
            self.ui.degreeComboBox.setVisible(True)
            self.ui.degreeComboBox.setDisabled(False)

        elif value.lower() == 'custom equation':
            self.ui.label_2.setText('y(x)=')
            self.ui.textEdit.setVisible(True)
            self.ui.degreeLabel.setVisible(False)
            self.ui.degreeComboBox.setVisible(False)

        else:
            self.ui.textEdit.setVisible(False)
            self.ui.label_2.setText('')
            self.ui.degreeLabel.setVisible(True)
            self.ui.degreeComboBox.setVisible(True)
            self.ui.degreeComboBox.setDisabled(True)
        self.initiate_fit()
#############################################


    ####DISPLAY
    def update_graph(self):
        if (self.x is not None) and (self.y is not None):
            x_min = np.min(self.x)
            x_max = np.max(self.x)
            y_min = np.min(self.y)
            y_max = np.max(self.y)
            y_span = y_max-y_min

            self.MplWidget.axes1.clear()
            self.MplWidget.axes1.set_xlabel('x')
            self.MplWidget.axes1.set_ylabel('y')
            self.MplWidget.axes1.set_xlim([x_min-self.x_span*0.05,x_max+self.x_span*0.05])
            self.MplWidget.axes1.set_ylim([y_min-y_span*0.05,y_max+y_span*0.05])
            self.MplWidget.axes1.plot(self.x, self.y, '.', c='black')
            if self.ui.checkBox_2.isChecked():
                self.MplWidget.axes1.plot(self.x_interpolate, self.y_fit_interpolate, c='blue')
            else:
                self.MplWidget.axes1.plot(self.x_fit, self.y_fit, c='blue')
            self.MplWidget.axes1.grid()
            self.MplWidget.draw()

    def update_results(self):
        self.ui.listWidget.clear()
        self.ui.listWidget.addItem('\nModel: '+self.ui.comboBox.currentText())
        self.ui.listWidget.addItem('    y(x)='+self.eq_str)
        self.ui.listWidget.addItem('')

        self.ui.listWidget.addItem('Coefficients (with standard deviation bounds):')
        for i, parameter in enumerate(self.params):
            self.ui.listWidget.addItem('    '+parameter+' =     '+str(np.round(self.paramvals[i],5))
                                       +'    ('+str(np.round(self.paramvals[i]+self.covars[i],5))+'  '+str(np.round(self.paramvals[i]-self.covars[i],5))+')')

        self.ui.listWidget.addItem('')
        self.ui.listWidget.addItem('Goodness of fit')
        self.ui.listWidget.addItem('  SSE: ' + str(np.round(self.SSE,5)))
        self.ui.listWidget.addItem('  R-Squared: ' + str(np.round(self.RSQR,5)))
        if isinstance(self.adjRSQR,float):
            self.ui.listWidget.addItem('  Adjusted R-Squared: ' + str(np.round(self.adjRSQR,5)))
        elif isinstance(self.adjRSQR,str):
            self.ui.listWidget.addItem('  Adjusted R-Squared: ' + self.adjRSQR)
        self.ui.listWidget.addItem('  RMSE: ' + str(np.round(self.RMSE,5)))
        if isinstance(self.chi_squared,np.float64):
            self.ui.listWidget.addItem('  Ⲭ-squared: ' + str(np.round(self.chi_squared,5)))
        elif isinstance(self.chi_squared,str):
            self.ui.listWidget.addItem('  Ⲭ-squared: ' + self.chi_squared)

    ##############################

    ###FITTING##
    def polynomial_fit(self):
        parameters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k']
        select = parameters[0:self.order + 1]
        self.n_vars = len(select)
        popt, pcov = np.polyfit(self.x, self.y, self.order,w=self.weights, cov=True)
        eq_str = (''.join([select[i] + '*x^' + str((self.order - i)) + '+' for i in range(len(select)-1)])+select[-1]).replace('^1','')
        self.covars = np.sqrt(np.diag(pcov))
        self.paramvals = popt
        self.eq_str = eq_str
        self.params = select
        self.y_fit = np.sum(np.array([popt[i] * self.x_fit ** (self.order - i) for i in range(len(select))]), axis=0)
        self.y_GOF = np.sum(np.array([popt[i] * self.x ** (self.order - i) for i in range(len(select))]), axis=0)
        if self.ui.checkBox_2.isChecked():
            self.y_fit_interpolate = np.sum(np.array([popt[i] * self.x_interpolate ** (self.order - i) for i in range(len(select))]), axis=0)

    def exponential(self,x,a,b):
        return b*np.exp(x*a)
    def exponential_fit(self,p0=[0,0]):
        parameters = ['a', 'b']
        self.n_vars = 2
        popt, pcov = scipy.optimize.curve_fit(self.exponential,self.x, self.y, p0, sigma=self.sigma)
        eq_str = "b*exp(a*x)"
        self.covars = np.sqrt(np.diag(pcov))
        self.paramvals = popt
        self.eq_str = eq_str
        self.params = parameters
        self.y_fit = popt[1]*np.exp(self.x_fit*popt[0])
        self.y_GOF = popt[1]*np.exp(self.x*popt[0])
        if self.ui.checkBox_2.isChecked():
            self.y_fit_interpolate = popt[1]*np.exp(self.x_interpolate*popt[0])

    def resonance1(self,x,a,b,c):
        return a*x/np.sqrt((x**2-b)**2+c*x**2)

    def resonance_fit_1(self,p0=[1,1,1]):
        parameters = ['a', 'b','c']
        self.n_vars = 3
        popt, pcov = scipy.optimize.curve_fit(self.resonance1,self.x,self.y,p0,sigma = self.sigma)

        eq_str = "a*x/sqrt((x^2-b)^2+c*x^2)"
        self.covars = np.sqrt(np.diag(pcov))
        self.paramvals = popt
        self.eq_str = eq_str
        self.params = parameters
        self.y_fit = popt[0]*self.x_fit/np.sqrt((self.x_fit**2-popt[1])**2+popt[2]*self.x_fit**2)
        self.y_GOF = popt[0]*self.x/np.sqrt((self.x**2-popt[1])**2+popt[2]*self.x**2)
        if self.ui.checkBox_2.isChecked():
            self.y_fit_interpolate = popt[0]*self.x_interpolate/np.sqrt((self.x_interpolate**2-popt[1])**2+popt[2]*self.x_interpolate**2)

    def resonance2(self,x,a,b,c):
        return a/np.sqrt((x**2-b)**2+c*x**2)

    def resonance_fit_2(self,p0=[1,1,1]):
        parameters = ['a', 'b','c']
        self.n_vars = 3
        popt, pcov = scipy.optimize.curve_fit(self.resonance2,self.x,self.y,p0, sigma = self.sigma)

        eq_str = "a/sqrt((x^2-b)^2+c*x^2)"
        self.covars = np.sqrt(np.diag(pcov))
        self.paramvals = popt
        self.eq_str = eq_str
        self.params = parameters
        self.y_fit = popt[0] / np.sqrt((self.x_fit ** 2 - popt[1]) ** 2 + popt[2] * self.x_fit ** 2)
        self.y_GOF = popt[0] / np.sqrt((self.x ** 2 - popt[1]) ** 2 + popt[2] * self.x ** 2)

        if self.ui.checkBox_2.isChecked():
            self.y_fit_interpolate = popt[0]/np.sqrt((self.x_interpolate**2-popt[1])**2+popt[2]*self.x_interpolate**2)

    def custom_fit(self,eq):
        customFit = function(eq)
        parameters = customFit.vars
        self.n_vars = len(parameters)
        popt, pcov = optim.curve_fit(customFit.func, self.x, self.y,p0=np.ones(len(parameters)),sigma = self.sigma)
        self.covars = np.sqrt(np.diag(pcov))

        self.eq_str = eq
        self.params = parameters
        self.paramvals = popt

        self.y_fit = customFit.func(self.x_fit, *popt)
        self.y_GOF = customFit.func(self.x, *popt)

        if self.ui.checkBox_2.isChecked():
            self.y_fit_interpolate = customFit.func(self.x_interpolate,*popt)


###################

#####GOF
    def sse(self):
        self.SSE = np.sum((self.y-self.y_GOF)**2)
    def rsqr(self):
        SSTOT = np.sum((self.y-np.mean(self.y))**2)
        self.RSQR = 1-self.SSE/SSTOT
    def adjustRsqrt(self):
        n = self.y.shape[0]
        if self.n_vars>1:
            self.adjRSQR = 1-(1-self.RSQR)*((n-1)/(self.n_vars-1))
        else:
            self.adjRSQR = 'Not Available with less than 2 variables'
    def rmse(self):
        n = self.y.shape[0]
        self.RMSE = np.sqrt(self.SSE/n)
    def chiSquared(self):
        if not isinstance(self.weights,type(None)):
            n = self.y.shape[0]
            self.chi_squared = 1/(n-self.n_vars)*np.sum(((self.y-self.y_GOF)*self.weights)**2)
        else:
            self.chi_squared = 'Not available'
    def GOD(self):
        self.sse()
        self.rsqr()
        self.adjustRsqrt()
        self.rmse()
        self.chiSquared()

def pyCftool(x=None, y=None, weights=None,local_vars = {}):
    if len(local_vars) != 0:
        for item in list(local_vars):
            if not isinstance(local_vars[item],np.ndarray):
                del local_vars[item]
    app = QApplication(sys.argv)
    mainwindow = MainWindow(x,y,weights=weights,local_variables=local_vars)
    mainwindow.setWindowTitle('CfTool')
    mainwindow.resize(1400, 900)
    mainwindow.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    x = np.linspace(-5,5,100)+np.random.normal(0,scale=0.1,size=100)
    noise = np.random.normal(0,scale=0.1,size=100)
    weights = 1/noise
    #y = 3/np.sqrt((x**2-4)**2+1*x**2)+noise  #Ressonance 2
    #y = 6.6*x**2-3*x+0.3+noise #Polynomial
    y = 2*x+0.2+2.2*np.sin(1.1*x)+noise
    lv = locals().copy()
    pyCftool(x,y,weights)
