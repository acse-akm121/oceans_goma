"""
A simple example file to set up a problem and QOI.
"""
from firedrake import *
from pyroteus import *
from pyroteus_adjoint import *
import matplotlib.pyplot as plt
from movement import *

plt.rcParams["figure.facecolor"] = "white"  # Ensure  background on plots is white

NU = 0.0001


def get_form(mesh_seq):
    def form(index, solutions):
        u, u_ = solutions["u"]
        P = mesh_seq.time_partition
        dt = Constant(P.timesteps[index])
        nu = Constant(NU)

        v = TestFunction(u.function_space())
        F = (
            inner((u - u_) / dt, v) * dx
            + inner(dot(u, nabla_grad(u)), v) * dx
            + nu * inner(grad(u), grad(v)) * dx
        )
        return F

    return form


def get_solver(mesh_seq):
    def solver(index, ic):
        function_space = mesh_seq.function_spaces["u"][index]
        u = Function(function_space)

        u_ = Function(function_space, name="u_old")
        u_.assign(ic["u"])

        # Define form
        F = mesh_seq.form(index, {"u": (u, u_)})

        # Time integrate from start to end
        P = mesh_seq.time_partition
        t_start, t_end = P.subintervals[index]
        dt = P.timesteps[index]
        t = t_start
        while t < t_end - 1e-5:
            solve(F == 0, u, ad_block_tag="u")
            u_.assign(u)
            t += dt
        return {"u": u}

    return solver


def get_function_spaces(mesh):
    return {"u": VectorFunctionSpace(mesh, "CG", 1), "f": FunctionSpace(mesh, "DG", 0)}


def get_initial_condition(mesh_seq):
    fs = mesh_seq.function_spaces["u"][0]
    x, y = SpatialCoordinate(mesh_seq[0])
    # An initial condition that propagates to both ends
    return {"u": interpolate(as_vector([sin(pi * (x - 0.5)), 0]), fs)}


def get_qoi(mesh_seq, solutions, i):
    def end_time_qoi():
        u = solutions["u"]
        # ds(2) is dy (ds, subdomain 2)
        return inner(u, u) * ds(2)

    return end_time_qoi


def plot_indicators(indicators, time_partition, **kwargs):
    """
    Plot a sequence of snapshots associated with
    ``indicators`` and :class:`TimePartition`
    ``time_partition``.

    Any keyword arguments are passed to ``tricontourf``.

    :arg indicators: list of list of indicators,
        indexed by mesh sequence index, then timestep
    :arg time_partition: the :class:`TimePartition`
        object used to solve the problem
    """
    P = time_partition
    rows = kwargs.pop("rows", (P.exports_per_subinterval[0] - 1) // 2)
    cols = kwargs.pop("cols", (P.num_subintervals * 2))
    steady = rows == cols == 1
    print(rows, cols)
    figsize = kwargs.pop("figsize", (6 * cols, 24 // cols))
    fig, axes = plt.subplots(rows, cols, sharex="col", figsize=figsize)
    tcs = []
    idx = -1
    for i in range(rows):
        tc = []
        for j in range(cols):
            idx += 1
            ax = axes[i, j]
            # Element-wise norm for the indicators
            tc.append(tricontourf(indicators[0][idx], axes=ax, **kwargs))
            if not steady:
                time = (
                    i * P.end_time / cols
                    + j * P.timesteps_per_export[0] * P.timesteps[0]
                )
                ax.annotate(f"t={time:.2f}", (0.05, 0.05), color="white")
        tcs.append(tc)
    plt.tight_layout()
    return fig, axes, tcs


def plot_mesh(mesh, fig=None, axes=None, time_partition=None, **kwargs):
    """
    Plot the given mesh as a surface with edges.
    """
    kwargs.setdefault("interior_kw", {"linewidth": 0.5})
    kwargs.setdefault("boundary_kw", {"linewidth": 2.0})
    if fig is None and axes is None:
        fig, axes = plt.subplots(figsize=(5, 5))
    tp = triplot(mesh, axes=axes, **kwargs)
    axes.axis(False)
    return fig, axes, tp


def standardize(arr):
    """
    Apply standardization based on mean and standard deviation
    """
    normed = (arr - np.mean(arr)) / np.std(arr)
    normed -= np.min(normed)
    normed += 1
    return normed


def lp_normalize(arr, p=2):
    """
    Normalize in L^p space.
    """
    retval = arr**p
    denom = sum(retval) ** (1 / p)
    retval /= denom
    return retval


def reorder_indicators(indicators):
    """
    Temporary function that changes the shape of the indicator list for plotting purposes.
    """
    temp = [k[0] for k in indicators]
    temp_indicators = [temp]
    return temp_indicators


def standardize_indicators(indicators, std_fn=standardize, **kwargs):
    """
    Apply given form of standardization or normalization to the monitor function.
    """
    P0 = indicators[0][0].function_space()
    retval = []
    for i in range(len(indicators)):
        t1 = []
        for j in range(len(indicators[0])):
            t1.append(Function(P0, val=std_fn(indicators[i][j].dat.data, **kwargs)))
        retval.append(t1)
    return retval


def get_get_monitor(mesh, i, indicators):
    """
    Get the given monitor function. This returns something compatible with the mover object.
    """

    def get_monitor(mesh):
        P0 = FunctionSpace(mesh, "DG", 0)
        f = Function(P0)
        f.project(indicators[0][i])
        return f

    return get_monitor


def main():
    n = 32
    fields = ["u"]
    mesh = UnitSquareMesh(n, n, diagonal="left")
    # We use a pyroteus.MeshSeq even though we have only one mesh because it allows us to use MeshSeq.indicate_errors()
    meshes = [mesh]
    end_time = 0.5
    dt = 1 / n
    num_subintervals = len(meshes)
    time_partition = TimePartition(
        end_time, num_subintervals, dt, fields, timesteps_per_export=2
    )
    mesh_seq = GoalOrientedMeshSeq(
        time_partition,
        meshes,
        get_function_spaces=get_function_spaces,
        get_initial_condition=get_initial_condition,
        get_form=get_form,
        get_solver=get_solver,
        get_qoi=get_qoi,
        qoi_type="end_time",
    )
    print("Solving the problem on a single, uniform mesh")
    # Get the forward and adjoint solutions and DWR error indicators
    solutions, indicators = mesh_seq.indicate_errors(
        enrichment_kwargs={"enrichment_method": "p"}
    )

    print("Pre-processing for movement")
    standardized_indicators = standardize_indicators(indicators)
    # Create new meshes to adapt. We  use as many meshes as we have exports of the monitor function
    meshes = [UnitSquareMesh(n, n, diagonal="left") for i in range(len(indicators[0]))]
    num_subintervals = len(meshes)
    time_partition = TimePartition(
        end_time, num_subintervals, dt, fields, timesteps_per_export=1
    )
    rtol = 1e-2
    maxiter = 100
    method = "quasi_newton"
    for i, mesh in enumerate(meshes):
        print("Starting on mesh {}".format(i))
        get_mon = get_get_monitor(mesh, i, standardized_indicators)
        mover = MongeAmpereMover(
            mesh, get_mon, method=method, maxiter=maxiter, rtol=rtol
        )
        mover.move()
        # if i < len(meshes) - 1:
        #     meshes[i + 1].coordinates.assign(mesh.coordinates)
    print("Computing quality metrics")
    quality_measures = {}
    for i, mesh in enumerate(meshes):
        quality_measures[i] = {}
        quality_measures[i]["min_angle"] = get_min_angles2d(mesh)
        quality_measures[i]["aspect_ratio"] = get_aspect_ratios2d(mesh)
        quality_measures[i]["skewness"] = get_skewnesses2d(mesh)
        quality_measures[i]["scaled_jac"] = get_scaled_jacobians2d(mesh)
        print("Mesh {}".format(i))
        print("Metric\t\tMin\t\tMax")
        for key in quality_measures[i].keys():
            print(
                "{}\t\t{:.3f}\t\t{:.3f}".format(
                    key,
                    np.min(quality_measures[i][key].dat.data),
                    np.max(quality_measures[i][key].dat.data),
                )
            )
    figsize = (10, 20)
    fig, axs = plt.subplots(4, 2, figsize=figsize)
    tps = []
    tp = []
    for i in range(len(meshes)):
        row, col = i // (len(meshes) // 4), i % 2
        _, _, plot = plot_mesh(meshes[i], fig=fig, axes=axs[row, col])
        tp.append(plot)
        min_ar, max_ar = np.min(quality_measures[i]["aspect_ratio"].dat.data), np.max(
            quality_measures[i]["aspect_ratio"].dat.data
        )
        if time_partition is not None:
            time = 2 * i * dt
            axs[row, col].set_title(
                "t={:.3f}, Aspect Ration Min/Max:{:.3f}{:.3f}".format(
                    time, min_ar, max_ar
                )
            )
    tps.append(tp)
    fig.tight_layout(pad=0.5)
    plt.suptitle("Adapted meshes")
    adapt_filename = "adapted_meshes"
    fig.savefig(adapt_filename)
    print("Saved adapted mesh figure to: {}".format(adapt_filename))

    for key in quality_measures[0].keys():
        out_file = File(f"quality_measures/{key}.pvd")
        for i in range(len(meshes)):
            out_file.write(quality_measures[i][key])
        print("Wrote to {}".format(out_file))


if __name__ == "__main__":
    main()
