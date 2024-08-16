import meshio
import numpy as np

mesh = meshio.read("model/triangular_beam.msh")

points = mesh.points
cells = mesh.cells
point_data = mesh.point_data
cell_data = mesh.cell_data

# Element data
eles = cells["quad"]
els_array = np.zeros([eles.shape[0], 7], dtype=int)
els_array[:, 0] = range(eles.shape[0])
els_array[:, 1] = 1
els_array[:, 3] = eles[:, 3]
els_array[:, 4] = eles[:, 2]
els_array[:, 5] = eles[:, 1]
els_array[:, 6] = eles[:, 0]

# Nodes
nodes_array = np.zeros([points.shape[0], 5])
nodes_array[:, 0] = range(points.shape[0])
nodes_array[:, 1:3] = points[:, :2]

# Boundary conditions
bound_bc = nodes_array[:, 1] == 2
nodes_array[:, 3] = nodes_array[:, 3] - bound_bc
nodes_array[:, 4] = nodes_array[:, 4] - bound_bc

# Loads
q0 = -10 / 2
bound_ab = nodes_array[:, 2] == 0
id_ab = nodes_array[:, 0][bound_ab]
q0_ab = nodes_array[:, 1][bound_ab] / 2 * q0
n_ab = len(id_ab)

bound_ac = nodes_array[:, 1] == nodes_array[:, 2]
id_ac = nodes_array[:, 0][bound_ac]
n_ac = len(id_ac)

loads_array = np.zeros((n_ab + n_ac, 3))
loads_array[:n_ab, 0] = id_ab
loads_array[:n_ab, 1] = 0
loads_array[:n_ab, 2] = q0_ab
loads_array[n_ab:n_ab + n_ac, 0] = id_ac

# Create files
np.savetxt("model/eles.txt", els_array, fmt="%d")
np.savetxt("model/nodes.txt", nodes_array, fmt=("%d", "%.4f", "%.4f", "%d", "%d"))
np.savetxt("model/loads.txt", loads_array, fmt=("%d", "%.6f", "%.6f"))
