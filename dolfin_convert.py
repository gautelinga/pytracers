import dolfin as df
import argparse


def main():
    parser = argparse.ArgumentParser(description="Trace particles")
    parser.add_argument("mesh_file", type=str,
                        help="Path to mesh file")
    parser.add_argument("u_file", type=str,
                        help="Path to velocity field file")
    parser.add_argument("out_file", type=str,
                        help="Path to .xdmf file to save to")
    args = parser.parse_args()

    mesh = df.Mesh()
    with df.HDF5File(mesh.mpi_comm(), args.mesh_file, "r") as h5f_mesh:
        h5f_mesh.read(mesh, "/mesh", False)

    V = df.VectorFunctionSpace(mesh, "CG", 1)
    u = df.Function(V)

    with df.HDF5File(mesh.mpi_comm(), args.u_file, "r") as h5f_u:
        h5f_u.read(u, "/velocity")

    xdmff = df.XDMFFile(mesh.mpi_comm(), args.out_file)
    xdmff.parameters["rewrite_function_mesh"] = False
    xdmff.parameters["flush_output"] = True
    xdmff.write(u, float(0.))


if __name__ == "__main__":
    main()
