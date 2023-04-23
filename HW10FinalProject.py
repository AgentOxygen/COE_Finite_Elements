import numpy as np

class FiniteElement2D:
    """
    2D Finite Element Code
    
    Written by Cameron Cummins (csc3323)
    
    4/23/2023
    """
    
    def getInputData(self, input_dict: dict) -> tuple:
        """
        Handles processing input files into arrays and dictionaries.
        
        input_dict --> Dictionary containing paths to the various input files.
        
        Returns a tuple:
            [0] np.array --> Array of nodes with their respective x, y positions
            [1] dict --> Maps node components to global indices for stiffness matrix assembly
            [2] np.array --> Array of known and unknown node component displacements consistent with [0]
            [3] np.array --> Boolean array indicating which node component displacements are known consistent with [2]
            [4] np.array --> Array of elements, globally indexed, with local nodes identified by index consistent with [0]
            [5] np.array --> Array of elements, globally indexed, with Young's Modulus E and Poissons Ratio nu
            [6] np.array --> Array of applied component forces consistent with [0]
        """
        # Check to make sure paths included
        assert "nodes" in input_dict, "Include path to 'nodes' file in input dictionary"
        assert "displacements" in input_dict, "Include path to 'displacements' file in input dictionary"
        assert "elements" in input_dict, "Include path to 'elements' file in input dictionary"
        assert "forces" in input_dict, "Include path to 'forces' file in input dictionary"
        
        def getCleanInput(path_to_input_file: str) -> list:
            """
            Cleans up arrays from input text files to avoid errors when interpreting data
            path_to_input_file -> path to input text file to read

            Returns:
                ret_list -> same array but with dirty entries removed
            """
            ret_list = []
            with open(path_to_input_file) as f:
                for string in f.readlines():
                    if len(string.split()) > 0:
                        ret_list.append(string)
            return ret_list

        ########################################################################
        #                     Reading Node Position Inputs                     #
        ########################################################################
        nodes_input = getCleanInput(input_dict["nodes"])
        # Create array storing node (x, y) positions
        node_positions = np.zeros((int(nodes_input[0]), 2))
        for node_def in nodes_input[1:]:
            index, x, y = node_def.split()
            node_positions[int(index)-1] = [x, y]
        # Create an index map for each positional (x and y) component of every node
        nodal_pos_map = {}
        global_pos_index = 0
        for node_index in range(node_positions.shape[0]):
            nodal_pos_map[(node_index, "x")] = global_pos_index
            global_pos_index += 1
            nodal_pos_map[(node_index, "y")] = global_pos_index
            global_pos_index += 1

        ########################################################################
        #                   Reading Node Displacement Inputs                   #
        ########################################################################
        disp_input = getCleanInput(input_dict["displacements"])
        # Create displacement array and indicate whether or not the values are known in a boolean mask
        displacements = np.zeros(node_positions.shape)
        displacements_mask = np.zeros(node_positions.shape, dtype=bool)
        for disp_def in disp_input[1:]:
            node_index, comp_index, disp = disp_def.split()
            displacements[int(node_index)-1][int(comp_index)-1] = float(disp)
            displacements_mask[int(node_index)-1][int(comp_index)-1] = True

        ########################################################################
        #                  Reading Element Definition Inputs                   #
        ########################################################################
        element_input = getCleanInput(input_dict["elements"])
        # Create array for storing three nodal indices (triangular)
        element_nodes = np.zeros((len(element_input)-1, 3), dtype=int)
        # In this case, E and nu are constants for all elements
        E, nu = (float(element_input[0].split()[1]), float(element_input[0].split()[2]))
        # But we will create an array anyways
        element_properties = np.zeros((len(element_input)-1, 2), dtype=float)

        for elmnt_def in element_input[1:]:
            index, n1, n2, n3 = elmnt_def.split()
            element_nodes[int(index)-1] = [int(n1)-1, int(n2)-1, int(n3)-1]
            element_properties[int(index)-1] = [E, nu] # Same for all elements


        ########################################################################
        #                 Reading Applied Nodal Forces Inputs                  #
        ########################################################################
        forces_input = getCleanInput(input_dict["forces"])
        # Create array for storing applied forces at each node
        applied_forces = np.zeros(node_positions.shape)
        for force in forces_input[1:]:
            index, comp, force = force.split()
            applied_forces[int(index)-1][int(comp)-1] = float(force)

        # Return input products
        return node_positions, nodal_pos_map, displacements, displacements_mask, element_nodes, element_properties, applied_forces


    def genTriangularElementStiffnessK(self, node_pos: np.array, node_pos_map: dict, element_node_indices: np.array, E: float, nu: float) -> tuple:
        """
        node_pos --> Global list of nodes and their respective positional coordinates
        node_pos_map --> Index of each coordinate in a flattened node_pos, used to coordinate local to global index transformations
        element_node_indices --> Global node indices associated with this particular element
        E --> Young's Modulus for this particular element
        nu --> Poisson's ratio
        
        Returns a tuple:
            [0] np.array --> Computed local stiffness matrix for this element
            [1] np.array --> Computed B matrix for this element
            [2] np.array --> Computed C matrix for this element
        """
        # Create a map to convert the local indices to global ones. This scheme follows the "flatten" method.
        local_to_global = np.array([
            node_pos_map[(element_node_indices[0], "x")],
            node_pos_map[(element_node_indices[0], "y")],
            node_pos_map[(element_node_indices[1], "x")],
            node_pos_map[(element_node_indices[1], "y")],
            node_pos_map[(element_node_indices[2], "x")],
            node_pos_map[(element_node_indices[2], "y")],
        ])

        x1, y1 = node_pos[element_node_indices[0]]
        x2, y2 = node_pos[element_node_indices[1]]
        x3, y3 = node_pos[element_node_indices[2]]

        # Compute area of the triangular element
        t_area = 0.5*(x2*y3-x3*y2+x3*y1-x1*y3+x1*y2-x2*y1)

        b1 = (y2-y3)/(2*t_area)
        b2 = (y3-y1)/(2*t_area)
        b3 = (y1-y2)/(2*t_area)

        c1 = (x3-x2)/(2*t_area)
        c2 = (x1-x3)/(2*t_area)
        c3 = (x2-x1)/(2*t_area)

        b_matrix = np.array(
        [
            [b1, 0, b2, 0, b3, 0],
            [0, c1, 0, c2, 0, c3],
            [c1, b1, c2, b2, c3, b3],
        ])

        c_matrix = np.array(
        [
            [E/(1-nu**2), (nu*E)/(1-nu**2), 0],
            [(nu*E)/(1-nu**2), E/(1-nu**2), 0],
            [0, 0, E/(2*(1+nu))]
        ])

        return t_area*np.transpose(b_matrix)@c_matrix@b_matrix, local_to_global, b_matrix, c_matrix


    def assembleMatrices(self, node_pos: np.array, nodal_pos_map: dict, element_nodes: np.array, element_props: np.array) -> tuple:
        """
        Assembles the global stiffness matrix, local B matrices, and local C matrices.
        node_pos --> Global list of nodes and their respective positional coordinates
        node_pos_map --> Index of each coordinate in a flattened node_pos, used to coordinate local to global index transformations
        element_nodes --> List of elements with their respective nodes identified by global node index
        element_nodes --> List of elements with their respective properties identified by global node index
        
        Returns a tuple:
            [0] np.array --> global stiffness matrix, indexed by node_pos_map
            [1] np.array --> the local B matrices, globally indexed by element
            [2] np.array --> the local C matrices, globally indexed by element
        """
        global_k = np.zeros((len(nodal_pos_map), len(nodal_pos_map)))
        local_b_matrices = np.zeros((len(element_nodes), 3, 6))
        local_c_matrices = np.zeros((len(element_nodes), 3, 3))

        for element_index, nodes in enumerate(element_nodes):
            E, nu = element_props[element_index]

            local_k, local_k_map, b_matrix, c_matrix = self.genTriangularElementStiffnessK(node_pos, nodal_pos_map, nodes, E, nu)
            # Store these matrices for calculating the element stresses/strains later
            local_c_matrices[element_index] = c_matrix
            local_b_matrices[element_index] = b_matrix
            # Assemble Global K matrix
            for i in range(local_k.shape[0]):
                for j in range(local_k.shape[1]):
                    global_k[local_k_map[i]][local_k_map[j]] += local_k[i][j]

        return global_k, local_b_matrices, local_c_matrices


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
        disps_shape = disps.shape
        disps = disps.flatten()
        ext_forces = ext_forces.flatten()

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

        # Reformat to multidimensional array for intuitive indexing
        sol_disps = np.zeros(disps_shape)
        for node_index in range(0, sol_disps.shape[0]):
            for comp_index in range(0, sol_disps.shape[1]):
                sol_disps[node_index][comp_index] = disps[node_index*2 + comp_index]

        return sol_disps


    def computeElementStrainsAndStresses(self, local_b_matrices: np.array, local_c_matrices: np.array, sol_disps: np.array, element_nodes: np.array) -> tuple:
        """
        Computes the strains and stresses associated with each element
        local_b_matrices --> B matrices associated with each element, indexed globally (this is output by genTriangularElementStiffnessK)
        local_c_matrices --> C matrices associated with each element, indexed globally (this is output by genTriangularElementStiffnessK)
        sol_disps --> Solved displacements output by solveDisplacements
        element_nodes --> List of elements with their respective nodes identified by global node index
        
        Returns a tuple:
            [0] np.array --> Array of element strains, indexed globally
            [1] np.array --> Array of element stresses, indexed globally
        """
        element_strains = np.zeros((local_b_matrices.shape[0], 3))
        element_stresses = np.zeros((local_b_matrices.shape[0], 3))
        for element_index in range(local_b_matrices.shape[0]):
            n1, n2, n3 = element_nodes[element_index]
            local_disps = np.zeros(6)
            local_disps[0:2] = sol_disps[n1]
            local_disps[2:4] = sol_disps[n2]
            local_disps[4:6] = sol_disps[n3]
            element_strains[element_index] = local_b_matrices[element_index]@local_disps
            element_stresses[element_index] = local_c_matrices[element_index]@element_strains[element_index]
            
        return element_strains, element_stresses
    
    
    def __init__(self, input_files_dictionary: dict) -> None:
        # Read inputs
        node_pos, nodal_pos_map, disp, disp_mask, element_nodes, element_props, applied_forces = self.getInputData(input_files_dictionary)
    
        # Compute Finite Elements
        k_global, local_b_matrices, local_c_matrices = self.assembleMatrices(node_pos, nodal_pos_map, element_nodes, element_props)
        k_reduced, global_to_reduced = self.reduceGlobalK(k_global, disp_mask.flatten())
        sol_disps = self.solveDisplacements(k_global, k_reduced, global_to_reduced, disp, applied_forces)
        sol_node_forces = k_global @ sol_disps.flatten()
        
        # Store outputs
        self.element_strains, self.element_stresses = self.computeElementStrainsAndStresses(local_b_matrices, local_c_matrices, sol_disps, element_nodes)
        self.node_displacements = sol_disps
        self.node_forces = sol_node_forces
        self.stiffness_matrix = k_global
        
        
if __name__ == "__main__":
    suffix = "6"
    paths = {
        "nodes": f"HW 10 Input Files/nodes{suffix}",
        "displacements": f"HW 10 Input Files/displacements{suffix}",
        "elements": f"HW 10 Input Files/elements{suffix}",
        "forces": f"HW 10 Input Files/forces{suffix}"
    }
    analysis = FiniteElement2D(paths)
    print("Displacements: ")
    print(analysis.node_displacements)
    print("Strains: ")
    print(analysis.element_strains)
    print("Stresses: ")
    print(analysis.element_stresses)
    