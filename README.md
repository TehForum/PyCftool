# PyCftool
Heavily inspired by matlabs interactive curvefitter (Cftool), this program attempts to copy its fantastic features for use in python.
<p align="center">
  <img src="https://user-images.githubusercontent.com/126679979/224439240-eed3dccd-48ca-4e11-8626-7d01737bfc71.png" width="850">
</p>



## Features

#### Easy access to data via dropdown menu
Using the argument `local_vars` you can pass multiple arrays to the GUI. From two dropdown menus (**X Data** and **Y Data**) you can choose to select, which x and y data the program should fit to.

#### Automatic fitting
Everytime the program detects a change in any option, it will automatically fit a new line using these new options. This can be disabled by unchecking **Auto Fit** in the top right corner. By doing so, a fit will only be computed once the **Fit** button, located under **Auto Fit**, is pressed.

#### Multiple equation support
From a dropdown menu it's possible to use a wide variety of predefined functions. So far these include
| Name | Equation |
| ------------- | ------------- |
| Polynomial of N degree  | $$y= \sum_{k=0}^N  a_kx^{(N-k)}$$ |
| Exponential  | $$y= be^{ax}$$ |
| Resonance 1  | $$y=\frac{ax}{\sqrt{(x^2-b)^2+cx^2}}$$  |
| Resonance 2  | $$y=\frac{a}{\sqrt{(x^2-b)^2+cx^2}}$$  |
| Custom Equation  | Text input  |

#### Custom Equation support
Choosing **Custom Equation** from the dropdown menu, it's possible to fit to any equation you desire!  

#### Interpolation
The fit will by default contain as many datapoints as the input data. When using a small amount of datapoints or fitting "pointy" functions, checking the **Interpolation** box will give the fit line a smoother appearence. Note that this ONLY influences how the fit appear on the graph and not any of the actual fit parameters.

#### Display Window
The left hand side contains a window that displays the current fit model, best fit parameters and GOF analysis.<br>
The best fit parameters come with a one $\sigma$ (68%) confidence interval. The program calculates the **SSE, R-Squard, Adjusted R-Squared, RMSE**, and 
if weights have been given, the $\chi$**-squared** of the model.<br>

#### Export to .py file
Under **"File"** you can choose to **"Export"**, which will create a .py file. This file will contain code, that generates the fit parameters that are used in plotting the graph.


#### Robust
The dropdown menu **Robust** gives access to different fitting algorithms. Default is the [Levenberg–Marquardt algorithm](https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm) (least squares). For bounded problems, the algorithm is automatically set to **TRF**. For more information on the different fitting algorithms check the [scipy.optimize.curve_fit](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html) documentation.


#### Fit Options
Fit options opens a window in which it's possible to adjust the initial variable guess as well as their boundaries.
<p align="center">
  <img src="https://user-images.githubusercontent.com/126679979/224429671-37055759-0699-423b-ac3a-c03bb2786744.png">
</p>




#### To be implemented

- Residual options
- Save
- Multiple plots
- 3D curve fit
## Prerequisite
Is currently supported for Python >=3.7.<br>
Requires the following packages to be installed: [PySide6](https://pypi.org/project/PySide6), Matplotlib, Scipy and Numpy.

## Installation
Will soon be on pip! For now the only way to install it to copy all the files into your working directory.

## Usage
Import the function `from pyCftool import pyCftool`.

### pyCftool
#### pyCftool(x=None ,y=None ,weights=None ,local_vars={})
&nbsp; Opens an interactive fitting program.<br>

&nbsp;&nbsp;  **Parameters:&nbsp;&nbsp; x :&nbsp;&nbsp; one-dimensional numpy array of length N, optional**<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The x component of the data that will immediately be displayed and fitted for once the GUI opens. <br><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**y :&nbsp;&nbsp; one-dimensional numpy array of length N, optional**<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The y component of the data that will immediately be displayed and fitted for once the GUI opens. <br><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**weights :&nbsp;&nbsp; float or one-dimensional numpy arrayof length N, optional**<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The weights of each datapoint. Defined as being $w=1/\sigma$, where $\sigma$ is the uncertainty of &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;the datapoints.<br><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**local_vars :&nbsp;&nbsp; dictionary, optional**<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; A dictionary can be passed, but only values that are of type `np.ndarray` will be accessable by the &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;GUI.  A dropdown menu in the GUI allows the user to freely choose between the arrays contained in &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;the dictionary. Passing the built-in function `locals()` will give the GUI access to all previously &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;defined numpy arrays.<br><br>

## Examples
### Polynomial fit
```Python
import numpy as np
from pyCftool import pyCftool

x = np.linspace(-5,5,100)+np.random.normal(0,scale=0.01,size=100)
noise = np.random.normal(0,scale=0.1,size=100)
y = 6.6*x**2-3*x+0.3+noise # Polynomial
lv = locals().copy()
pyCftool(x,y,local_vars=lv)
```
Will produce a window like this:
<p align="center">
  <img src="https://user-images.githubusercontent.com/126679979/224442985-5fab6f97-f30e-434c-a589-4333bb88d86b.png" width="750">
</p>

The 'best-fit' parameters are listed in the window on the right hand side. The bounds are within one σ (68%) standard deviation. Since this is a second order polynomial, we can select Degree = 2 from the dropdown menu. The fitter will automatically fit unless **"Auto Fit"** has been unchecked. The resultning image will become
<p align="center">
  <img src="https://user-images.githubusercontent.com/126679979/224443057-5b3694d3-3b18-4388-8119-e2d7481b9e04.png" width="750">
</p>

### Custom Fit
We can also make the program fit any function we desire.

```Python
import numpy as np
from pyCftool import pyCftool

x = np.linspace(-5,5,100)+np.random.normal(0,scale=0.01,size=100)
noise = np.random.normal(0,scale=0.1,size=100)
y = 2*x+0.2+2.2*np.sin(1.1*x)+noise # Linear with sinus wave
lv = locals().copy()
pyCftool(x,y,local_vars=lv)
```
By selecting `Custom Equation` from the dropdown menu at the top, we can input our own equations. In this case we want to use the formula $y=ax+b+A\sin(\omega x)$.
<p align="center">
  <img src="https://user-images.githubusercontent.com/126679979/224443276-b7090965-3f31-4869-9dba-ec738aa47ac0.png" width="800">
</p>
