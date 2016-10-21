import pytracers
import h5py
import argparse
import numpy as np


def combinations(arr):
    """
    Sorted array of all combinations of drawing n-1 items from arr.

    >>> combinations(range(3))
    array([[1, 2],
           [0, 2],
           [0, 1]])
    """
    n = len(arr)
    perms = np.zeros((n, n-1), dtype=int)
    for i in xrange(n):
        perms[i, :i] = arr[:i]
        perms[i, i:] = arr[i+1:]
    return perms


def build_faces(elems):
    """
    Build faces and elem2faces structures

    >>> elems = np.asarray([[0, 1, 2, 3], [0, 2, 3, 4], \
                           [0, 1, 2, 5], [0, 2, 4, 5]], dtype=int)
    >>> faces, elem2faces, face2elems = build_faces(elems)
    >>> faces
    array([[1, 2, 3],
           [0, 2, 3],
           [0, 1, 3],
           [0, 1, 2],
           [2, 3, 4],
           [0, 3, 4],
           [0, 2, 4],
           [1, 2, 5],
           [0, 2, 5],
           [0, 1, 5],
           [2, 4, 5],
           [0, 4, 5]])
    >>> elem2faces
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  1],
           [ 7,  8,  9,  3],
           [10, 11,  8,  6]])
    >>> face2elems
    [[0], [0, 1], [0], [0, 2], [1], [1], [1, 3], [2], [2, 3], [2], [3], [3]]
    """
    n_elems = elems.shape[0]
    dim = elems.shape[1]-1

    face_dict = dict()
    elem2faces = [[] for i in xrange(n_elems)]

    comb_ids = combinations(range(dim+1))

    n_face = 0
    for i_elem, elem in enumerate(elems):
        for ids in comb_ids:
            face = elem[ids].tolist()
            face.sort()
            face = tuple(face)
            if face in face_dict:
                elem2faces[i_elem].append(face_dict[face])
            else:
                face_dict[face] = n_face
                elem2faces[i_elem].append(n_face)
                n_face += 1
    faces = np.asarray(invert_dict(face_dict), dtype=int)
    face2elems = invert_list_of_lists(elem2faces)
    return faces, np.asarray(elem2faces, dtype=int), face2elems


def invert_dict(key2val):
    """
    Invert a dict with tuple keys and integer index to a list

    >>> d = {(1, 2, 3): 1, (2, 3, 4): 2, (1, 2, 4): 0}
    >>> invert_dict(d)
    [[1, 2, 4], [1, 2, 3], [2, 3, 4]]
    """
    n = len(key2val)
    val2key = [[] for _ in xrange(n)]
    for key, val in key2val.iteritems():
        val2key[val] = list(key)
    return val2key


def invert_list_of_lists(key2vals):
    """
    Invert a list of lists containing ids.

    >>> a = [[0, 1, 2, 3], [4, 5, 6, 1], [7, 8, 9, 3], [10, 11, 8, 6]]
    >>> invert_list_of_lists(a)
    [[0], [0, 1], [0], [0, 2], [1], [1], [1, 3], [2], [2, 3], [2], [3], [3]]
    """
    n = int(np.max(np.asarray(key2vals, dtype=int).flatten())+1)
    val2keys = [[] for _ in xrange(n)]
    for key, vals in enumerate(key2vals):
        for val in vals:
            val2keys[val].append(key)
    return val2keys


def build_boundary_face_normals(nodes, elems, faces,
                                face2elems, boundary_faces):
    """
    Build normals of external facets.

    >>> elems = np.array([[0, 1, 2, 5], \
                          [0, 2, 3, 5], \
                          [0, 3, 4, 5], \
                          [0, 4, 1, 5]], dtype=int)
    >>> nodes = np.array([[0., 0., -1.], \
                          [-1., -1., 0.], \
                          [1., -1., 0.], \
                          [1., 1., 0.], \
                          [-1., 1., 0.], \
                          [0., 0., 1.]], dtype=float)
    >>> faces, elem2faces, face2elems = build_faces(elems)
    >>> boundary_faces, interior_faces = \
            find_boundary_and_interior_faces(face2elems)
    >>> n, A, x_face_mean = build_boundary_face_normals(nodes, elems, faces, \
                                                        face2elems, boundary_faces)
    >>> n
    array([[ 0.        , -0.70710678,  0.70710678],
           [-0.        , -0.70710678, -0.70710678],
           [ 0.70710678, -0.        ,  0.70710678],
           [ 0.70710678, -0.        , -0.70710678],
           [ 0.        ,  0.70710678,  0.70710678],
           [-0.        ,  0.70710678, -0.70710678],
           [-0.70710678, -0.        ,  0.70710678],
           [-0.70710678,  0.        , -0.70710678]])
    >>> A
    array([ 2.82842712,  2.82842712,  2.82842712,  2.82842712,  2.82842712,
            2.82842712,  2.82842712,  2.82842712])
    """
    n_nodes, dim = nodes.shape
    n_bfaces = len(boundary_faces)
    A = np.zeros(n_bfaces)
    n = np.zeros((n_bfaces, dim))
    x_face_mean = np.zeros((n_bfaces, dim))
    for i_bface, bface in enumerate(boundary_faces):
        nodes_from_face = faces[bface, :]
        nodes_from_elem = elems[face2elems[bface][0], :]
        x_elem = nodes[nodes_from_elem, :]
        x_face = nodes[nodes_from_face, :]
        x_elem_mean = np.mean(x_elem, 0)
        x_face_mean[i_bface, :] = np.mean(x_face, 0)
        An = np.cross(x_face[1, :] - x_face[0, :], x_face[2, :] - x_face[0, :])
        A[i_bface] = np.linalg.norm(An)
        n[i_bface, :] = An/A[i_bface] * np.sign(
            np.dot(An, x_face_mean[i_bface, :] - x_elem_mean))
    return n, A, x_face_mean


def find_boundary_and_interior_faces(face2elems):
    """
    Build boundary faces.

    >>> face2elems = [[0], [0, 1], [0], [0, 2], [1], [1], \
                      [1, 3], [2], [2, 3], [2], [3], [3]]
    >>> boundary_faces, interior_faces = \
             find_boundary_and_interior_faces(face2elems)
    >>> boundary_faces
    array([ 0,  2,  4,  5,  7,  9, 10, 11])
    >>> interior_faces
    array([1, 3, 6, 8])
    """
    boundary_faces = []
    interior_faces = []
    for face, connected_elems in enumerate(face2elems):
        n_connected = len(connected_elems)
        if n_connected == 1:
            boundary_faces.append(face)
        elif n_connected == 2:
            interior_faces.append(face)
        else:
            raise ValueError("Wrong length of element in face2elems list.")
    return (np.asarray(boundary_faces, dtype=int),
            np.asarray(interior_faces, dtype=int))


def dot_arrays_of_vectors(a, b):
    """
    Dot two arrays of vectors together... is there a numpy function for this?

    >>> a = np.array([[1., 2., 3.], [3., 2., 1.]])
    >>> b = np.array([[2., 1., 0.], [4., 2., 3.]])
    >>> dot_arrays_of_vectors(a, b)
    array([  4.,  19.])
    """
    n, dim = a.shape
    c = np.zeros(n)
    for i in xrange(dim):
        c[:] += a[:, i] * b[:, i]
    return c


def build_u_normal_face(u, faces, boundary_faces, face_normals):
    n_faces, dim = faces.shape
    u_face = np.zeros((len(boundary_faces), dim))
    for i_bface, bface in enumerate(boundary_faces):
        for node in faces[bface]:
            u_face[i_bface, :] += u[node, :]/3
    u_normal_face = dot_arrays_of_vectors(u_face, face_normals)
    return u_normal_face


def get_extrema(nodes):
    """
    Return tuple of vectors of max and min of a set of coordinates.

    >>> a = np.array([[0., 1., -1.], [1., -2., 4.]])
    >>> get_extrema(a)
    (array([ 1.,  1.,  4.]), array([ 0., -2., -1.]))
    """
    return np.max(nodes, axis=0), np.min(nodes, axis=0)


def find_boundary_and_interior_nodes_from_faces(faces,
                                                boundary_faces, n_nodes):
    boundary_nodes = sorted(np.unique(faces[boundary_faces, :].flatten()))
    interior_nodes = np.asarray(list(set(range(n_nodes))-set(boundary_nodes)),
                                dtype=int)
    return boundary_nodes, interior_nodes


def get_flow_direction(vector):
    """
    Get closest flow direction from a vector of normal velocity. Axis
    denotes x,y,z, Sign denotes direction.

    >>> get_flow_direction(np.array([0., 1., 0.]))
    (1, 1.0)
    """
    axis = np.argmax(vector**2)
    sign = np.sign(vector[axis])
    return axis, sign


def get_boundary_inlet_face_ids(conditon, face_normals):
    inlet_ids_1 = np.where(conditon)[0]

    flow_axis, flow_sign = get_flow_direction(
        np.mean(face_normals[inlet_ids_1, :], 0))

    inlet_ids_2 = np.where(
        face_normals[inlet_ids_1.flatten(), flow_axis] == flow_sign)[0]

    inlet_ids = inlet_ids_1[inlet_ids_2]
    return inlet_ids


def main():
    """
    Prototype
    """
    parser = argparse.ArgumentParser(description="Trace particles")
    parser.add_argument("input_file", type=str,
                        help="Path to input .h5 file")
    args = parser.parse_args()

    with h5py.File(args.input_file, "r") as h5f:
        nodes = np.asarray(h5f["Mesh/0/coordinates"])
        elems = np.asarray(h5f["Mesh/0/topology"], dtype=int)
        n_nodes, dim = nodes.shape
        n_elems, dimplusone = elems.shape
        assert dim == dimplusone-1
        u = np.asarray(h5f["VisualisationVector/0"])
        n_unodes, udim = u.shape
        assert n_nodes == n_unodes
        assert dim == udim

    faces, elem2faces, face2elems = build_faces(elems)
    n_faces = len(faces)

    print "n_nodes:  ", n_nodes
    print "n_elems:  ", n_elems
    print "n_faces:  ", n_faces

    assert elem2faces.shape[0] == n_elems
    assert len(face2elems) == n_faces

    boundary_faces, interior_faces = \
        find_boundary_and_interior_faces(face2elems)
    boundary_nodes, interior_nodes = \
        find_boundary_and_interior_nodes_from_faces(faces,
                                                    boundary_faces,
                                                    n_nodes)

    n_bnodes = len(boundary_nodes)
    n_inodes = len(interior_nodes)
    print "n_bnodes: ", n_bnodes
    print "n_inodes: ", n_inodes
    assert n_bnodes + n_inodes == n_nodes

    face_normals, face_areas, x_face_mean = build_boundary_face_normals(
        nodes, elems, faces, face2elems, boundary_faces)
    u_normal_face = build_u_normal_face(u, faces, boundary_faces, face_normals)

    thresh = 1e-6

    inlet_face_ids = get_boundary_inlet_face_ids(u_normal_face < -thresh,
                                                 face_normals)

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(nodes[:, 0], nodes[:, 1], nodes[:, 2],
                    triangles=faces[boundary_faces[inlet_face_ids], :],
                    array=u_normal_face[inlet_face_ids],
                    antialiased=False)
    plt.show()


if __name__ == "__main__":
    import doctest
    doctest.testmod()
    main()
