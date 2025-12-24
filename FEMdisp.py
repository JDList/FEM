
import meshio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import sys, pprint

def extract_cell_stress(mesh, tri_cells):
    """Return sxx, syy, sxy arrays of length Nc.
       Tries to handle common layouts: (Nc,3,3), (Nc,9), (Nc,3)."""
    Nc = tri_cells.shape[0]
    for key, block in mesh.cell_data_dict.items():
        for bname, data in block.items():
            arr = np.array(data)
            if arr.shape[0] != Nc:
                continue
            if arr.ndim == 3 and arr.shape[1] == 3 and arr.shape[2] == 3:
                sxx = arr[:,0,0].astype(float)
                sxy = arr[:,0,1].astype(float)
                syy = arr[:,1,1].astype(float)
                return sxx, syy, sxy
            if arr.ndim == 2 and arr.shape[1] == 9:
                sxx = arr[:,0].astype(float)
                sxy = arr[:,1].astype(float)
                syy = arr[:,4].astype(float)
                return sxx, syy, sxy
            if arr.ndim == 2 and arr.shape[1] == 3:
                sxx = arr[:,0].astype(float)
                syy = arr[:,1].astype(float)
                sxy = arr[:,2].astype(float)
                return sxx, syy, sxy
    if hasattr(mesh, "cell_data") and isinstance(mesh.cell_data, dict):
        for key, arr_list in mesh.cell_data.items():
            for arr in arr_list:
                arrnp = np.array(arr)
                if arrnp.shape[0] != Nc:
                    continue
                if arrnp.ndim == 3 and arrnp.shape[1] == 3 and arrnp.shape[2] == 3:
                    return arrnp[:,0,0].astype(float), arrnp[:,1,1].astype(float), arrnp[:,0,1].astype(float)
                if arrnp.ndim == 2 and arrnp.shape[1] == 9:
                    return arrnp[:,0].astype(float), arrnp[:,4].astype(float), arrnp[:,1].astype(float)
                if arrnp.ndim == 2 and arrnp.shape[1] == 3:
                    return arrnp[:,0].astype(float), arrnp[:,1].astype(float), arrnp[:,2].astype(float)
    raise RuntimeError("Could not locate cell stress data in mesh.cell_data_dict or mesh.cell_data")

def nodal_averaging_scalar_per_cell(tri_cells, cell_values, Nnodes):
    """Average a scalar cell value to nodes (simple arithmetic mean)."""
    nodal = np.zeros(Nnodes, dtype=float)
    counts = np.zeros(Nnodes, dtype=int)
    for c in range(tri_cells.shape[0]):
        for v in tri_cells[c]:
            nodal[v] += cell_values[c]
            counts[v] += 1
    counts[counts == 0] = 1
    nodal /= counts
    return nodal

def main():
    fname = "results.vtk"
    print("Reading:", fname)
    mesh = meshio.read(fname)

    tri_cells = None
    for cb in mesh.cells:
        if cb.type.lower() in ("triangle", "tri"):
            tri_cells = cb.data
            break
    if tri_cells is None:
        print("No triangle connectivity found in file.")
        sys.exit(1)

    Nc = tri_cells.shape[0]
    Np = mesh.points.shape[0]
    print(f"Found {Np} points and {Nc} triangles.")

    sxx_cell, syy_cell, sxy_cell = extract_cell_stress(mesh, tri_cells)
    print("Extracted cell stress shapes:", sxx_cell.shape, syy_cell.shape, sxy_cell.shape)
    print("sxx stats:", sxx_cell.min(), sxx_cell.mean(), sxx_cell.max())

    sxx_nodal = nodal_averaging_scalar_per_cell(tri_cells, sxx_cell, Np)
    syy_nodal = nodal_averaging_scalar_per_cell(tri_cells, syy_cell, Np)
    sxy_nodal = nodal_averaging_scalar_per_cell(tri_cells, sxy_cell, Np)

    vm_nodal = np.sqrt(np.maximum(0.0, sxx_nodal**2 - sxx_nodal*syy_nodal + syy_nodal**2 + 3.0*sxy_nodal**2))
    pts = mesh.points[:, :2] 
    triang = tri.Triangulation(pts[:,0], pts[:,1], triangles=tri_cells)

    fig, axes = plt.subplots(2, 2, figsize=(12,10))
    levels = 50

    def plot_scalar(ax, values, title, cmap='viridis'):
        tcf = ax.tricontourf(triang, values, levels=levels, cmap=cmap)
        ax.triplot(triang, color='k', linewidth=0.4, alpha=0.4)
        ax.set_title(title)
        ax.set_aspect('equal', 'box')
        cb = fig.colorbar(tcf, ax=ax, orientation='vertical')
        return cb

    plot_scalar(axes[0,0], sxx_nodal, r'$\sigma_{xx}$ (nodal-averaged)')
    plot_scalar(axes[0,1], syy_nodal, r'$\sigma_{yy}$ (nodal-averaged)')
    plot_scalar(axes[1,0], sxy_nodal, r'$\tau_{xy}$ (nodal-averaged)')
    plot_scalar(axes[1,1], vm_nodal,     r'von Mises (nodal)')

    plt.suptitle(r"Stress components and von Mises (nodal-averaged)", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()

if __name__ == "__main__":
    main()
plt.show()
