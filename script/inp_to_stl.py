import meshio
import gmsh
import numpy as np

"""
Script to convert finite element mesh files (such as Abaqus .inp files) to STL format.

Note: Sometimes the given surface element types might not be in the provided element types, 
please add them manually if needed.
The code supports tetrahedrons, hexahedrons, triangles, and quadrilaterals.
Tested successfully with Abaqus mesh of type C3D8R.
"""

def convert_to_stl_with_gmsh(input_file, output_file):
    """Convert mesh file to STL format using gmsh"""

    # First convert inp to msh format using meshio
    print("Reading input file...")
    msh = meshio.read(input_file, file_format="abaqus")
    temp_msh_file = "temp_conversion.msh"
    meshio.write(temp_msh_file, msh, file_format="gmsh")
    print(f"Converted to temporary file: {temp_msh_file}")

    # Initialize gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)  # Show terminal output

    try:
        # Open msh file
        gmsh.open(temp_msh_file)

        # Get all nodes
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        print(f"Number of nodes: {len(node_tags)}")

        # Reorganize node coordinates
        coords = node_coords.reshape(-1, 3)

        # Get all surface elements (triangles and quadrilaterals)
        surface_elements = []
        element_types = gmsh.model.mesh.getElementTypes()

        for elem_type in element_types:
            print("Element type:", elem_type)
            # Get all elements of this type
            elem_tags, elem_connectivity = gmsh.model.mesh.getElementsByType(elem_type)
            # Check element type
            elem_name = gmsh.model.mesh.getElementProperties(elem_type)[0]
            print(f"Element type: {elem_name}, Count: {len(elem_tags)}")
            # If it's a volume element, need to extract its surface
            if elem_name in ['4-node tetrahedron', 'Tetrahedron', 'tetra']:
                # Four faces of a tetrahedron
                connectivity = elem_connectivity.reshape(-1, 4)
                for tet in connectivity:
                    # Four triangular faces of a tetrahedron
                    faces = [
                        [tet[0], tet[1], tet[2]],
                        [tet[0], tet[1], tet[3]],
                        [tet[0], tet[2], tet[3]],
                        [tet[1], tet[2], tet[3]]
                    ]
                    surface_elements.extend(faces)

            elif elem_name in ['8-node hexahedron', 'Hexahedron', "Hexahedron 8", 'hexahedron']:
                # Six faces of a hexahedron
                print("Starting extraction")
                connectivity = elem_connectivity.reshape(-1, 8)
                for hex_elem in connectivity:
                    # Six quadrilateral faces of a hexahedron, converted to triangles
                    faces = [
                        # Bottom face (divided into two triangles)
                        [hex_elem[0], hex_elem[1], hex_elem[2]],
                        [hex_elem[0], hex_elem[2], hex_elem[3]],
                        # Top face
                        [hex_elem[4], hex_elem[5], hex_elem[6]],
                        [hex_elem[4], hex_elem[6], hex_elem[7]],
                        # Side faces
                        [hex_elem[0], hex_elem[1], hex_elem[5]],
                        [hex_elem[0], hex_elem[5], hex_elem[4]],
                        [hex_elem[1], hex_elem[2], hex_elem[6]],
                        [hex_elem[1], hex_elem[6], hex_elem[5]],
                        [hex_elem[2], hex_elem[3], hex_elem[7]],
                        [hex_elem[2], hex_elem[7], hex_elem[6]],
                        [hex_elem[3], hex_elem[0], hex_elem[4]],
                        [hex_elem[3], hex_elem[4], hex_elem[7]]
                    ]
                    surface_elements.extend(faces)

            elif elem_name in ['3-node triangle', 'Triangle', 'triangle']:
                # Directly triangular faces
                connectivity = elem_connectivity.reshape(-1, 3)
                for tri in connectivity:
                    surface_elements.append([tri[0], tri[1], tri[2]])

            elif elem_name in ['4-node quadrangle', 'Quadrangle', 'quad']:
                # Quadrilateral faces, split into two triangles
                connectivity = elem_connectivity.reshape(-1, 4)
                for quad in connectivity:
                    surface_elements.extend([
                        [quad[0], quad[1], quad[2]],
                        [quad[0], quad[2], quad[3]]
                    ])
            else:
                raise ValueError(f"Unknown element type: {elem_name}")

        if not surface_elements:
            print("Warning: No surface elements found, attempting to extract all elements directly...")
            # If no surface elements found, all elements might be volume elements, need to extract outer surface
            all_faces = {}

            for elem_type in element_types:
                elem_tags, elem_connectivity = gmsh.model.mesh.getElementsByType(elem_type)
                elem_name = gmsh.model.mesh.getElementProperties(elem_type)[0]

                if '4-node tetrahedron' in elem_name or 'Tetrahedron' in elem_name:
                    connectivity = elem_connectivity.reshape(-1, 4)
                    for tet in connectivity:
                        faces = [
                            tuple(sorted([tet[0], tet[1], tet[2]])),
                            tuple(sorted([tet[0], tet[1], tet[3]])),
                            tuple(sorted([tet[0], tet[2], tet[3]])),
                            tuple(sorted([tet[1], tet[2], tet[3]]))
                        ]
                        for face in faces:
                            if face in all_faces:
                                del all_faces[face]  # Internal face, remove
                            else:
                                all_faces[face] = list(face)  # External face, keep

            surface_elements = list(all_faces.values())

        print(f"Number of extracted faces: {len(surface_elements)}")

        if not surface_elements:
            raise ValueError("Could not extract any surface elements")

        # Convert node indices (gmsh nodes start from 1, need to convert to start from 0)
        min_node_tag = min(node_tags)
        max_node_tag = max(node_tags)

        # Create node mapping
        node_map = {tag: i for i, tag in enumerate(node_tags)}

        # Convert face node indices
        triangles = []
        for face in surface_elements:
            try:
                mapped_face = [node_map[node_tag] for node_tag in face]
                triangles.append(mapped_face)
            except KeyError as e:
                print(f"Warning: Node {e} does not exist, skipping this face")
                continue

        triangles = np.array(triangles)
        print(f"Number of valid triangles: {len(triangles)}")

        # Write STL file using meshio
        if len(triangles) > 0:
            cells = [("triangle", triangles)]
            output_mesh = meshio.Mesh(coords, cells)
            meshio.write(output_file, output_mesh, file_format="stl")
            print(f"✅ Successfully generated STL file: {output_file}")
            print(f"   Number of nodes: {len(coords)}")
            print(f"   Number of triangles: {len(triangles)}")
        else:
            raise ValueError("No valid triangles to write")

    except Exception as e:
        print(f"Error: {e}")
        raise
    finally:
        gmsh.finalize()
        # Clean up temporary file
        import os
        if os.path.exists(temp_msh_file):
            os.remove(temp_msh_file)


# Main program
if __name__ == "__main__":
    convert_to_stl_with_gmsh("Job.inp", "Job_surface.stl")