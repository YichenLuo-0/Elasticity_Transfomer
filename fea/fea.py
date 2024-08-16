import matplotlib.pyplot as plt
from solidspy import solids_GUI

# Run the Finite Element Analysis
disp = solids_GUI(compute_strains=True, folder="./model/")
plt.show()
