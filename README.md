# PyCftool
Heavily inspired by matlabs interactive curvefitter (Cftool), this program attempts to copy its fantastic features for use in python.
<p align="center">
  <img src="https://user-images.githubusercontent.com/126679979/223150375-aeb987d4-94f5-4167-9fbc-2d6ae54f81b6.png" width="850">
</p>

## Features

#### Easy access to data via dropdown menu
Using the argument `local_vars` it is possible to pass multiple arrays to the GUI. From two dropdown menus (**X Data** and **Y Data**) it is possible to select, which x and y data the program should fit tp.

#### Automatic fitting
Everytime the program detects a change in any option, it will automatically fit a new line using the new parameters. This can be disabled by unchecking **Auto Fit** in the top right corner. By doing so, a fit will only be computed once the **Fit** Button, located undero **Auto Fit** is pressed.

#### Multiple equation support
From a dropdown menu it is possible to use a wide variety of predefined functions. So far these include
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
The fit will by default contain as many datapoints as the input data. When using a small amount of datapoints or fitting "pointy" functions, checking **Interpolation** will give the fit line a smoother appearence. Note that this ONLY influences how the fit appear on the graph and not any of the actual fit parameters.

#### Display Window
The left hand side contains a window that displays the current fit model, best fit parameters and GOF analysis.<br>
The best fit parameters come with a one $\sigma$ (68%) confidence interval. The program calculates the fits **SSE, R-Squard, Adjusted R-Squared, RMSE** and 
and if weights have been given the $\chi$**-squared** of the model*.<br>
*Currently weights have no function in pyCftool, but will come in future updates. 


#### To be implemented

- Save and export
- Weights
- Robust
- Fit options
- Residual options
- Multiple plots
- 2D curve fit

## Usage
Import the function `from pyCftool import pyCftool`.

### pyCftool
#### pyCftool(x ,y ,weights=None ,local_vars=None)
&nbsp; Opens an interactive fitting program.<br>

&nbsp;&nbsp;  **Parameters:&nbsp;&nbsp; x :&nbsp;&nbsp; one-dimensional numpy array of length N**<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The x component of the data that will emidiatly be displayed and fitted for once the GUI opens. <br><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**y :&nbsp;&nbsp; one-dimensional numpy array of length N**<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The y component of the data that will emidiatly be displayed and fitted for once the GUI opens. <br><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**weights :&nbsp;&nbsp; float or one-dimensional numpy arrayof length N, optional**<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The weights of each datapoint. Currently has no function, but will in the future be used to give better &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;fit estimate and calculate $\chi$-squared.<br><br>
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
  <img src="https://user-images.githubusercontent.com/126679979/223142476-ce64f1be-8425-4d75-8ac6-3852d485a9c6.png" width="750">
</p>
The 'best-fit' parameters are listed in the window on the right hand side. The bounds are within one σ (68%) standard deviation. Since this is a second order polynomial, we can select order = 2 from the dropdown menu. The fitter will automatically fit unless `Auto Fit` has been unchecked. The resultning image will become
<p align="center">
  <img src="https://user-images.githubusercontent.com/126679979/223142544-2cb848c2-d10e-4f64-9940-1dcb8dcbfca8.png" width="750">
</p>

### Costum Fit
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
  <img src="https://user-images.githubusercontent.com/126679979/223147572-3dd52297-9254-404b-9c0e-e82a0a0a9470.png" width="800">
</p>
