import numpy as np
import re

def create_string(s,p,pv,s_clean,w,bound,method):
    str0 = "import numpy as np\nfrom scipy import optimize as optim\nfrom matplotlib import pyplot as plt\n\n"
    str1 = f"def pyCFfunc(x,{','.join(p)}):\n"
    str2 = "    return " + s + "\n\n"
    if isinstance(w,np.ndarray):
        str3 = "#x = X_DATA HERE\n#y = Y_DATA HERE\n#sigma = WEIGHTS HERE #(weights=1/sigma)\n\n"
        str4 = f"popt, pcov = optim.curve_fit(pyCFfunc,x,y,p0=[{','.join([str(v) for v in pv])}], sigma=sigma"
    else:
        str3 = "#x = X_DATA HERE\n#y = Y_DATA HERE\n\n"
        str4 = f"popt, pcov = optim.curve_fit(pyCFfunc,x,y,p0=[{','.join([str(v) for v in pv])}]"
    if bound:
        str5 = f", bounds=({','.join(str(b).replace('inf','np.inf') for b in bound[0])},{','.join(str(b).replace('inf','np.inf') for b in bound[1])})"
    else:
        str5=""
    if method:
        str6 = f", method='{method}')\n\n"
    else:
        str6 = ")\n\n"
    str7 = f"fig, ax = plt.subplots()\nf1 = ax.plot(x,y,'.',c='black',label='Raw data')\nf2 = ax.plot(x,pyCFfunc(x,*popt),c='b', label='Best fit: y={s_clean}')\nax.legend()\nax.grid()\nplt.show()"
    return str0+str1+str2+str3+str4+str5+str6+str7

def export_fit(eq_str,params,paramvals,w, dir,bound,method):
    eq_split = re.split(r'(\W+)', eq_str)
    np_words = ['sqrt','sin','cos','exp','pi']
    for i in range(len(eq_split)):
        if eq_split[i] =='^':
            eq_split[i]='**'
        elif eq_split[i] in np_words:
            eq_split[i] = 'np.'+eq_split[i]

    eq_export = ''.join(eq_split)
    export_str = create_string(eq_export,params,paramvals,eq_str,w,bound,method)
    with open(dir,'w') as f:
        f.write(export_str)
        f.close()
