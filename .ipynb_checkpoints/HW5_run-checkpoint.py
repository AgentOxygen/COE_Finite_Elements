import json
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.patches import FancyArrowPatch
import copy


class FiniteTrussElement3D:
    """
    3D Finite Truss Element Code
    
    Written by Cameron Cummins (csc3323)
    
    2/24/2023
    """
    
    
    def genLocal3DTrussStiffnessMatrix(self, n1_pos: np.array, n2_pos: np.array, youngs: float, area: float) -> np.array:
        """
        Computes the local stiffness matrix for a 3D truss element.
        n1_pos --> Position array for local node 1
        n2_pos --> Position array for local ndoe 2
        youngs --> Youngs Modulus for this element
        area --> Cross-sectional area of this element

        Returns 6x6 matrix of local siffness matrix k
        """
        dx = n2_pos[0] - n1_pos[0]
        dy = n2_pos[1] - n1_pos[1]
        dz = n2_pos[2] - n1_pos[2]
        L = (dx**2 + dy**2 + dz**2)**0.5
        c1 = dx/L
        c2 = dy/L
        c3 = dz/L

        k = (1/L)*(youngs*area)*np.array([
            [c1**2, c1*c2, c1*c3, -1*(c1**2), -1*c1*c2, -1*c1*c3],
            [c1*c2, c2**2, c2*c3, -1*c1*c2, -1*(c2**2), -1*c2*c3],
            [c1*c3, c2*c3, c3**2, -1*c1*c3, -1*c2*c3, -1*(c3**2)],
            [-1*(c1**2), -1*c1*c2, -1*c1*c3, c1**2, c1*c2, c1*c3],
            [-1*c1*c2, -1*(c2**2), -1*c2*c3, c1*c2, c2**2, c2*c3],
            [-1*c1*c3, -1*c2*c3, -1*(c3**2), c1*c3, c2*c3, c3**2]
        ])
        return k


    def genExternalForcesVector(self, global_index: dict, ext_forces_input: dict) -> np.array:
        """
        Matches the external forces from the input to the global index.
        global_index --> Global index of nodal displacements to column position in the global
                         stiffness matrix
        ext_forces_input --> Node to forces dictionary from input JSON

        Returns vector dimensionally equal to the columns of the global stiffness matrix
        """
        ext_forces = np.zeros(len(global_index))
        for node_name in ext_forces_input:
            ext_forces[global_index[(node_name, "u")]] = ext_forces_input[node_name][0]
            ext_forces[global_index[(node_name, "v")]] = ext_forces_input[node_name][1]
            ext_forces[global_index[(node_name, "w")]] = ext_forces_input[node_name][2]
        return ext_forces


    def genLocaltoGlobalDict(self, n1_name: str, n2_name: str, global_index: dict) -> dict:
        """
        Generates a dictionary for transferring values from a local stiffness matrix into
        the global matrix.
        n1_name --> Global name of local node 1
        n2_name --> Global name of local node 2
        global_index --> Global index containing the global names and their respective indices

        Returns dictionary that converts local indices to global indices
        """
        local_index = {
            0 : global_index[(n1_name, "u")],
            1 : global_index[(n1_name, "v")],
            2 : global_index[(n1_name, "w")],
            3 : global_index[(n2_name, "u")],
            4 : global_index[(n2_name, "v")],
            5 : global_index[(n2_name, "w")],
        }

        return local_index


    def genDisplacementsAndMask(self, global_index: dict, fixed_disps: dict) -> tuple:
        """
        Generates the known displacements and boolean array where each entry corresponds to whether 
        or not that particular displacement is known. This is used to reduce the global stiffness 
        matrix and thus has a matching index.
        global_index --> Global index of nodal displacements to column position in the global
                       stiffness matrix
        fixed_disps --> List fixed/known nodal displacements from input JSON file

        Returns tuple:
        [0] np.array --> vector of boolean values dimensionally equal to the global index
                         True = known
                         False = unknown
        [1] np.array --> vector of displacements, note that unknown displacements are 0 by default
        """
        mask = np.zeros(len(global_index), dtype=bool)
        displacements = np.zeros(len(global_index))
        for node_name in fixed_disps.keys():
            for component in fixed_disps[node_name]:
                mask[global_index[(node_name, component)]] = True
                displacements[global_index[(node_name, component)]] = fixed_disps[node_name][component]

        return (mask, displacements)


    def reduceGlobalK(self, global_k: np.array, mask: np.array) -> tuple:
        """
        Reduces the global stiffness matrix to solve for unknown displacements
        global_k --> The original global stiffness matrix
        mask --> Displacement masking vector, boolean array, indicating which column 
                    indices contain a known values

        Returns a tuple:
            [0] np.array --> Reduced stiffness, square array of size equal to number 
                             of unknown displacements
            [1] dict --> Global-to-reduced index transformation
        """
        num_unk = np.sum(1-mask)
        k_reduced = np.zeros((num_unk, num_unk))

        global_to_reduced = {}
        local_index = 0
        for global_index, known in enumerate(mask):
            if not known:
                global_to_reduced[global_index] = local_index
                local_index += 1

        for i in range(0, global_k.shape[0]):
            for j in range(0, global_k.shape[1]):
                if not mask[i] and not mask[j]:
                    local_i = global_to_reduced[i]
                    local_j = global_to_reduced[j]
                    k_reduced[local_i][local_j] = global_k[i][j]
        return (k_reduced, global_to_reduced)


    def solveDisplacements(self, global_k: np.array, reduced_k: np.array, reduced_index: dict, disps: np.array, ext_forces: np.array) -> np.array:
        """
        Solves for the unknown displacements and replaces their values in the displacement vector.
        global_k --> The global stiffness matrix
        reduced_k --> The reduced stiffness matrix
        reduced_index --> The global-to-reduced index transformation
        disps --> The known and unknown displacements vector with index corresponding to the global stiffness matrix
        ext_forces --> The external force components with index corresponding to the global stiffness matrix

        Returns a displacement vector dimensionally equivalent to the displacement vector parameter but
            with the unknown displacements replaced with computed values.
        """
        subtract_known_disps = np.zeros(len(reduced_index))

        for i in range(0, global_k.shape[0]):
            if i in reduced_index.keys():
                for j in range(0, global_k.shape[1]):
                    if j not in reduced_index.keys():
                        subtract_known_disps[reduced_index[i]] += global_k[i][j]*disps[j]
        
        known_forces = np.zeros(len(reduced_index))
        for force_index in reduced_index.keys():
            known_forces[reduced_index[force_index]] = ext_forces[force_index]
            
        solved_displacements = np.linalg.inv(reduced_k).dot(known_forces - subtract_known_disps)
        for disp_index in reduced_index.keys():
            disps[disp_index] = solved_displacements[reduced_index[disp_index]]
        
        return disps


    def solveNodalReactionForces(self, global_k: np.array, disps: np.array) -> np.array:
        """
        Computes the reaction forces at each node, with components matching the index of the displacements vector.
        global_k --> The global stiffness matrix
        disps --> The known and unknown displacements vector with index corresponding to the global stiffness matrix

        Returns a vector array of nodal reaction forces.
        """
        return global_k.dot(disps)


    def solveInternalForces(self, pos: dict, conns: dict, disps: np.array, global_index: dict) -> np.array:
        """
        Computes the reaction forces at each node, with components matching the index of the displacements vector.
        pos --> Nodal positions dictionary from input JSON
        conns --> Nodal connections dictionary from input JSON
        disps --> The solved displacements vector with index corresponding to the global stiffness matrix
        global_index --> Global index of nodal displacements to column position in the global
                         stiffness matrix

        Returns a tuple:
            [0] --> vector array of internal forces with length equal to the number of connections
            [1] --> Indexing of the internal foces array to node names
        """
        int_forces = np.zeros(len(conns))
        int_forces_index = {}
        for index, conn in enumerate(conns):
            int_forces_index[index] = conn["labels"]
            n1_name = conn["labels"][0]
            n2_name = conn["labels"][1]
            n1_pos = pos[n1_name]
            n2_pos = pos[n2_name]

            dx = n2_pos[0] - n1_pos[0]
            dy = n2_pos[1] - n1_pos[1]
            dz = n2_pos[2] - n1_pos[2]
            L = (dx**2 + dy**2 + dz**2)**0.5
            c1 = dx/L
            c2 = dy/L
            c3 = dz/L

            n1_u = disps[global_index[(n1_name, "u")]]
            n1_v = disps[global_index[(n1_name, "v")]]
            n1_w = disps[global_index[(n1_name, "w")]]
            n2_u = disps[global_index[(n2_name, "u")]]
            n2_v = disps[global_index[(n2_name, "v")]]
            n2_w = disps[global_index[(n2_name, "w")]]
            
            int_forces[index] = c1*(n2_u-n1_u)/(L) + c2*(n2_v-n1_v)/(L) + c3*(n2_w-n1_w)/(L)

        return (int_forces, int_forces_index)


    def buildGlobalStiffnessMatrix(self, pos: dict, conns: dict, global_index: dict) -> np.array:
        """
        Constructs the global stiffness matrix using local element stiffness matrices.
        pos --> Nodal positions dictionary from input JSON
        conns --> Nodal connections dictionary from input JSON
        global_index --> Global index of nodal displacements to column position in the global stiffness matrix

        Returns a square, symmetric global stiffness matrix, dimensionally consistent with the global index
        """
        global_k = np.zeros((len(global_index), len(global_index)))

        for conn in conns:
            n1_name = conn["labels"][0]
            n2_name = conn["labels"][1]
            n1_pos = pos[n1_name]
            n2_pos = pos[n2_name]

            local_k = self.genLocal3DTrussStiffnessMatrix(n1_pos, n2_pos, conn["youngs_mod"], conn["area"])
            local_to_global = self.genLocaltoGlobalDict(n1_name, n2_name, global_index)

            for i in range(0, local_k.shape[0]):
                g_i = local_to_global[i]
                for j in range(0, local_k.shape[1]):
                    g_j = local_to_global[j]
                    global_k[g_i][g_j] += local_k[i][j]
        return global_k
    
    
    def validateResults(self) -> str:
        """
        Preforms assertion tests to validate the results at a high level and outputs a summary.
        
        Returns a string containing the summary in a printable format.
        """
        assert np.allclose(self.global_stiffness, self.global_stiffness.T), "Global stiffness matrix is not symmetric."
        assert np.diagonal(self.global_stiffness).all() > 0, "Global stiffness matrix is not positive along the diagonal."
        
        input_json = self.input_structure
        for node_name in input_json["node_xy_fixed"]:
            for component in input_json["node_xy_fixed"][node_name]:
                index = self.node_index[(node_name, component)]
                fixed_value = input_json["node_xy_fixed"][node_name][component]
                assert self.displacements[index] == fixed_value, \
                    f"Node '{node_name}', '{component}' == {self.displacements[index]} and isn't fixed to {fixed_value}"
        
        assert len(self.element_internal_forces) == len(input_json["node_connections"]), \
            f"{len(self.element_internal_forces)} internal forces, but {len(input_json['node_connections'])} connections specified."
        
        
        date = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        
        output_str = ""
        output_str += f"========================= Results Validation (Assertion Tests) =========================\n"
        output_str += f"Date: {date}\n"
        output_str += f"Input JSON File: '{self.__input_file_path}'\n\n"
        output_str += f"Global stiffness natrix is symmetric.\n"
        output_str += f"Global stiffness natrix is positive along the diagonal.\n"
        output_str += f"Fixed nodes have still have known displacement values.\n"
        output_str += f"Correct number of internal forces calculated.\n"
        return output_str
    
    
    def calculateLengthDeltaAndStrains(self, element_index: dict, input_structure: dict, output_structure: dict) -> tuple:
        """
        Calculates the changes in length and internal strains of the elements as a result of the structural deformation.
        element_index --> Global element index for matching values to elements
        input_structure --> dictionary from JSON data for initial structure
        output_structure --> dictionary of equivalent format to input_structure for structure after deformation
        
        Returns a tuple:
            [0] --> change in length for each element
            [1] --> strains associated with each element
        """
        length_deltas = np.zeros((len(element_index)))
        strains = np.zeros((len(element_index)))
        for index in element_index:
            na, nb = element_index[index]
            x1, y1, z1 = input_structure["node_positions"][na]
            x2, y2, z2 = input_structure["node_positions"][nb]
            l1 = pow(pow(x2-x1, 2) + pow(y2-y1, 2) + pow(z2-z1, 2), 0.5)
            x1, y1, z1 = output_structure["node_positions"][na]
            x2, y2, z2 = output_structure["node_positions"][nb]
            l2 = pow(pow(x2-x1, 2) + pow(y2-y1, 2) + pow(z2-z1, 2), 0.5)
            
            length_deltas[index] = l2-l1
            strains[index] = (l2-l1)/l1
        return length_deltas, strains
    
    
    def calculateElementStresses(self, element_index: dict, element_forces: np.array, output_structure: dict) -> np.array:
        """
        Calculates the element stresses as a result of the final deformation
        element_index --> Global element index for matching values to elements
        element_forces --> Internal forces associated with each element, following the index
        output_structure --> dictionary of equivalent format to input_structure for structure after deformation
        
        Returns an array of element stresses
        """
        stresses = np.zeros(element_forces.shape)
        for conn in output_structure["node_connections"]:
            for index in element_index:
                if element_index[index] == conn["labels"]:
                    stresses[index] = element_forces[index] / conn["area"]
                    break
        return stresses
    
    
    def __init__(self, input_file_path: str) -> None:
        with open(input_file_path) as input_file:
            input_json = json.load(input_file)

        # Build node index
        global_node_index = {}
        index = 0
        for node_name in input_json["node_positions"]:
            global_node_index[(node_name, "u")] = index
            global_node_index[(node_name, "v")] = index + 1
            global_node_index[(node_name, "w")] = index + 2
            index += 3

        # Build the global stiffness matrix
        global_k = self.buildGlobalStiffnessMatrix(input_json["node_positions"], input_json["node_connections"], global_node_index)
        # Generate reduction vector for solving for unknown displacements
        mask, displacements = self.genDisplacementsAndMask(global_node_index, input_json["node_xy_fixed"])
        # Reduce K
        reduced_k, reduced_index = self.reduceGlobalK(global_k, mask)
        # Pull external forces
        external_forces = self.genExternalForcesVector(global_node_index, input_json["external_node_forces"])
        # Calculate and replace the unknown displacements with the solutions
        displacements = self.solveDisplacements(global_k, reduced_k, reduced_index, displacements, external_forces)
        # Compute nodal reaction forces
        reaction_forces = self.solveNodalReactionForces(global_k, displacements)
        # Compute internal forces
        int_forces, int_forces_index = self.solveInternalForces(input_json["node_positions"], input_json["node_connections"], displacements, global_node_index)
        
        self.__input_file_path = input_file_path
        self.input_structure = input_json
        self.global_stiffness = global_k
        self.node_index = global_node_index
        self.displacements = displacements
        self.node_external_forces = external_forces
        self.node_reaction_forces = reaction_forces
        self.element_internal_forces = int_forces
        self.element_index = int_forces_index
        
        # Run assertion tests to check for any obvious mistakes
        self.validation = self.validateResults()
        
        # Generate deformed structure output
        self.output_structure = copy.deepcopy(input_json)
        for node, comp in self.node_index:
            index = self.node_index[(node, comp)]
            if comp == "u":
                self.output_structure["node_positions"][node][0] += self.displacements[index]
            elif comp == "v":
                self.output_structure["node_positions"][node][1] += self.displacements[index]
            elif comp == "w":
                self.output_structure["node_positions"][node][2] += self.displacements[index]
                
        self.element_length_deltas, self.element_strains = self.calculateLengthDeltaAndStrains(self.element_index, self.input_structure, self.output_structure)
        self.element_stresses = self.calculateElementStresses(self.element_index, self.element_internal_forces, self.output_structure)

        
if __name__ == "__main__":
    # Run the finite element analysis
    analysis = FiniteTrussElement3D("HW5_3D_truss.json")
    
    # Output some of the results
    for node_comp in analysis.node_index.keys():
        index = analysis.node_index[node_comp]
        print(f"{node_comp} = {analysis.displacements[index]}")


    # Plot the results
    f, ax1 = plt.subplots(1, 1, figsize=(14, 12), facecolor='w', subplot_kw={'projection': "3d"})

    class Arrow3D(FancyArrowPatch):

        def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
            super().__init__((0, 0), (0, 0), *args, **kwargs)
            self._xyz = (x, y, z)
            self._dxdydz = (dx, dy, dz)

        def draw(self, renderer):
            x1, y1, z1 = self._xyz
            dx, dy, dz = self._dxdydz
            x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

            xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
            super().draw(renderer)

        def do_3d_projection(self, renderer=None):
            x1, y1, z1 = self._xyz
            dx, dy, dz = self._dxdydz
            x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

            xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

            return np.min(zs) 

    def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
        arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
        ax.add_artist(arrow)

    setattr(Axes3D, 'arrow3D', _arrow3D)

    # Add external forces
    put_label = True
    for node in analysis.input_structure["external_node_forces"]:
        x, y, z = analysis.input_structure["node_positions"][node]
        fu, fv, fw = analysis.input_structure["external_node_forces"][node]
        if put_label:
            ax1.arrow3D(x, y, z, fu, fv, fw, color="Blue", mutation_scale=20, shrinkA=100, shrinkB=1, label="External Force")
            put_label = False
        else:
            ax1.arrow3D(x, y, z, fu, fv, fw, color="Blue", mutation_scale=20, shrinkA=100, shrinkB=1)

    # Add points and connections
    for data, color, ls, lw, label in [(analysis.input_structure, "Green", "-", 4, "Initial"), ((analysis.output_structure, "Red", "--", 2, "Final"))]:
        pts_x = []
        pts_y = []
        pts_z = []
        pts_labels = []

        # Plot points
        for pt in data["node_positions"].keys():
            pts_labels.append(pt)
            x, y, z = data["node_positions"][pt]
            pts_x.append(x)
            pts_y.append(y)
            pts_z.append(z)

        # Plot connections
        put_label = True
        for conn in data["node_connections"]:
            a, b = conn["labels"]
            x1, y1, z1 = data["node_positions"][a]
            x2, y2, z2 = data["node_positions"][b]
            if put_label:
                ax1.plot([x1, x2], [y1, y2], [z1, z2], linewidth=lw, color=color, linestyle=ls, label=label)
                put_label = False
            else:
                ax1.plot([x1, x2], [y1, y2], [z1, z2], linewidth=lw, color=color, linestyle=ls)

        ax1.scatter(pts_x, pts_y, pts_z, color=color, s=80)

        for i, label in enumerate(pts_labels):
            ax1.text(pts_x[i], pts_y[i], pts_z[i], label, zdir=None, color="Black", fontsize=20)

    ax1.grid()
    ax1.legend(fontsize=15)
    ax1.set_xlabel("x", fontsize=15)
    ax1.set_ylabel("y", fontsize=15)
    ax1.set_zlabel("z", fontsize=15)
    ax1.set_title("HW5 Structure Deformation", fontsize=25)

    print("See 'HW5_deform.png' for reference")
    f.savefig("HW5_deform.png")