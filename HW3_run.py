import json
import numpy as np
from datetime import datetime


class FiniteElement2D:
    """
    2D Finite Element Code
    
    Written by Cameron Cummins (csc3323)
    
    2/15/2023
    """
    
    
    def genLocal2DTrussStiffnessMatrix(self, n1_pos: np.array, n2_pos: np.array, material: float) -> np.array:
        """
        Computes the local stiffness matrix for a 2D truss element.
        n1_pos --> Position array for local node 1
        n2_pos --> Position array for local ndoe 2
        material --> Material behavior associated with this particular 2D element,
                        dependent on Young's modulus and the cross sectional area

        Returns 4x4 matrix of local siffness matrix k
        """
        dx = n2_pos[0] - n1_pos[0]
        dy = n2_pos[1] - n1_pos[1]
        L = (dx**2 + dy**2)**0.5
        cos = dx/L
        sin = dy/L

        k = (1/L)*material*np.array([
            [cos**2, cos*sin, -1*(cos**2), -1*cos*sin],
            [cos*sin, sin**2, -1*cos*sin, -1*(sin**2)],
            [-1*(cos**2), -1*cos*sin, cos**2, cos*sin],
            [-1*cos*sin, -1*(sin**2), sin*cos, sin**2]
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
            2 : global_index[(n2_name, "u")],
            3 : global_index[(n2_name, "v")],
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
            L = (dx**2 + dy**2)**0.5
            cos = dx/L
            sin = dy/L

            n1_u = disps[global_index[(n1_name, "u")]]
            n1_v = disps[global_index[(n1_name, "v")]]
            n2_u = disps[global_index[(n2_name, "u")]]
            n2_v = disps[global_index[(n2_name, "v")]]

            int_forces[index] = cos*(n2_u-n1_u)/(L) + sin*(n2_v-n1_v)/(L)

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

            local_k = self.genLocal2DTrussStiffnessMatrix(n1_pos, n2_pos, conn["material"])
            local_to_global = self.genLocaltoGlobalDict(n1_name, n2_name, global_index)

            for i in range(0, 4):
                g_i = local_to_global[i]
                for j in range(0, 4):
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
        
    
    def __init__(self, input_file_path: str) -> None:
        with open(input_file_path) as input_file:
            input_json = json.load(input_file)

        # Build node index
        global_node_index = {}
        index = 0
        for node_name in input_json["node_positions"]:
            global_node_index[(node_name, "u")] = index
            global_node_index[(node_name, "v")] = index + 1
            index += 2

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


if __name__ == "__main__":
    # Run the finite element analysis
    analysis = FiniteElement2D("four_node_truss.json")
    # Output the displacements for each node's component
    for node, component in analysis.node_index.keys():
        print(f"{node}{component} = {round(analysis.displacements[analysis.node_index[(node, component)]], 3)}")