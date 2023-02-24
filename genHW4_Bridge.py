"""
Generates the Input JSON for the HW4 Bridge Structure

Written by Cameron Cummins (csc3323)

2/23/2023
"""
import json

if __name__ == "__main__":
    youngs =  210*(10**9)
    cross_area = 0.01

    nodes = {}
    conns = []
    nindex = 0

    # Create the points from the ground up, connecting points in each row
    row1_labels = []
    for row in range(8):
        label = chr(65+nindex)
        row1_labels.append(label)
        nodes[label] = [row*5, 0]
        nindex += 1

    for index in range(0, len(row1_labels)):
        conns.append({"labels": [row1_labels[index], row1_labels[index-1]], "youngs_mod": youngs, "area": cross_area})

    row2_labels = []
    for row in range(1, 7):
        label = chr(65+nindex)
        row2_labels.append(label)
        nodes[label] = [row*5, 5]
        nindex += 1

    for index in range(1, len(row2_labels)):
        conns.append({"labels": [row2_labels[index], row2_labels[index-1]], "youngs_mod": youngs, "area": cross_area})

    row3_labels = []
    for row in range(2, 6):
        label = chr(65+nindex)
        row3_labels.append(label)
        nodes[label] = [row*5, 10]
        nindex += 1

    for index in range(1, len(row3_labels)):
        conns.append({"labels": [row3_labels[index], row3_labels[index-1]], "youngs_mod": youngs, "area": cross_area})

    # Connect the rows to each other
    for index in range(1, len(row1_labels)-1):
        conns.append({"labels": [row1_labels[index], row2_labels[index-1]], "youngs_mod": youngs, "area": cross_area})
    for index in range(1, len(row2_labels)-1):
        conns.append({"labels": [row2_labels[index], row3_labels[index-1]], "youngs_mod": youngs, "area": cross_area})

    # Lower diagonals
    for index in range(0, len(row2_labels)-1):
        conns.append({"labels": [row2_labels[index], row1_labels[index+2]], "youngs_mod": youngs, "area": cross_area})
    # Upper diagonals
    for index in range(1, len(row3_labels)):
        conns.append({"labels": [row2_labels[index], row3_labels[index]], "youngs_mod": youngs, "area": cross_area})
    
    # Add edges
    conns.append({"labels": [row1_labels[0], row2_labels[0]], "youngs_mod": youngs, "area": cross_area})
    conns.append({"labels": [row2_labels[0], row3_labels[0]], "youngs_mod": youngs, "area": cross_area})
    conns.append({"labels": [row1_labels[-1], row2_labels[-1]], "youngs_mod": youngs, "area": cross_area})
    conns.append({"labels": [row2_labels[-1], row3_labels[-1]], "youngs_mod": youngs, "area": cross_area})

    # Fixed points
    fixed = {}
    fixed[row1_labels[0]] = {"u": 0, "v": 0}
    fixed[row1_labels[-1]] = {"u": 0, "v": 0}

    # Add external forces
    forces = {}
    for index in range(1, len(row1_labels)-1):
        forces[row1_labels[index]] = [0, -1]

    output = {
        "node_positions": nodes,
        "node_connections": conns,
        "node_xy_fixed": fixed,
        "external_node_forces": forces
    }

    jsonified = json.dumps(output, indent=4)

    # Writing to sample.json
    with open("HW4_bridge.json", "w") as outfile:
        outfile.write(jsonified)