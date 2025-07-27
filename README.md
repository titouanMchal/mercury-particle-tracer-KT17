# mercury-particle-tracer-KT17
Test particle tracing code at Mercury using KT17 model

### Required packages : 
- numpy
- scipy
- matplotlib
- KT17
  
Note : the KT17 library uses np.bool8, which is deprecated in recent NumPy versions. You may need to manually replace it with np.bool_ to ensure compatibility.

### Usage :

To run a basic example and visualize particle trajectories in Mercury’s magnetosphere, execute the script examples.py, which demonstrates how to initialize particles, compute their trajectories, and plot diagnostics.

### References :

**KT17 model**
Korth, H., N. A. Tsyganenko, C. L. Johnson, L. C. Philpott, B. J. Anderson, M. M. Al Asad, S. C. Solomon, and R. L. McNuttJr. (2015), Modular model for Mercury's magnetospheric magnetic field confined within the average observed magnetopause. J. Geophys. Res. Space Physics, 120, 4503–4518. doi: 10.1002/2015JA021022.
https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2015JA021022

**KT17 library**
https://github.com/mattkjames7/KT17
