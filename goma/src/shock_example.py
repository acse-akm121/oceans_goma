"""
A simple example file to set up a problem and QOI.
"""
from firedrake import *
from pyroteus import *
from pyroteus_adjoint import *
import matplotlib.pyplot as plt


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


def plot_indicator_snapshots2(indicators, time_partition, **kwargs):
    """
    Plot a sequence of snapshots associated with
    ``indicators`` and :class:`TimePartition`
    ``time_partition``. This modification plots the log of errors.

    Any keyword arguments are passed to ``tricontourf``.

    :arg indicators: list of list of indicators,
        indexed by mesh sequence index, then timestep
    :arg time_partition: the :class:`TimePartition`
        object used to solve the problem
    """
    P = time_partition
    rows = P.exports_per_subinterval[0] - 1
    cols = P.num_subintervals
    steady = rows == cols == 1
    figsize = kwargs.pop("figsize", (6 * cols, 24 // cols))
    fig, axes = plt.subplots(rows, cols, sharex="col", figsize=figsize)
    tcs = []
    for i, indi_step in enumerate(indicators):
        ax = axes if steady else axes[0] if cols == 1 else axes[0, i]
        ax.set_title(f"Mesh[{i}]")
        tc = []
        for j, indi in enumerate(indi_step):
            t_indi = Function(indi.function_space(), val=np.log(indi.dat.data))
            ax = axes if steady else axes[j] if cols == 1 else axes[j, i]
            tc.append(tricontourf(t_indi, axes=ax, **kwargs))
            if not steady:
                time = (
                    i * P.end_time / cols
                    + j * P.timesteps_per_export[i] * P.timesteps[i]
                )
                ax.annotate(f"t={time:.2f}", (0.05, 0.05), color="white")
        tcs.append(tc)
    plt.tight_layout()
    return fig, axes, tcs


def main():
    print("Program starting")
    n = 32
    fields = ["u"]
    meshes = [
        UnitSquareMesh(n, n, diagonal="left"),
        UnitSquareMesh(n, n, diagonal="left"),
    ]
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
    solutions, indicators = mesh_seq.indicate_errors(
        enrichment_kwargs={"enrichment_method": "p"}
    )

    vmax, vmin = -np.inf, np.inf
    for i, mesh_i in enumerate(indicators):
        for j, e_t in enumerate(indicators[i]):
            vmax = np.max((vmax, np.max(np.log(np.abs(e_t.dat.data)))))
            vmin = np.min((vmin, np.min(np.log(np.abs(e_t.dat.data)))))

    figsize = (
        2 * len(solutions["u"]["forward"][0]),
        2 * len(solutions["u"]["forward"][0]),
    )
    # print(vmax, vmin)
    # vmax = -12
    # vmin = -70
    # fig, axs, tcs = plot_indicator_snapshots2(
    #     indicators,
    #     time_partition,
    #     levels=np.linspace(vmin, vmax, 100),
    #     vmax=vmax,
    #     vmin=vmin,
    #     cmap="viridis",
    #     figsize=figsize,
    # )
    # cbar = fig.colorbar(tcs[1][-3], ax=axs, location="top")
    # print(np.exp(-24.6), np.exp(-14))
    # fig.savefig("fig1_dwr.jpg")
    # smoothing of monitor function:
    gamma = 2  # Refinement level
    eta_b = 5e-8
    funcs = []
    P0 = indicators[i][0].function_space()
    for i, mesh_i in enumerate(indicators):
        #     area = areas[i]

        max_errs = np.zeros(indicators[i][0].dat.data.shape)
        fns = []
        for j, e_t in enumerate(indicators[i]):
            val = np.array(e_t.dat.data)
            fn = Function(P0, val=val)
            fns.append(fn)
        funcs.append(fns)
    # print(len(funcs))
    # print(len(funcs[1]))
    # fig, axs, tcs = plot_indicator_snapshots(
    #     funcs, time_partition, cmap="viridis", figsize=figsize
    # )
    # cbar = fig.colorbar(tcs[-1][-1], ax=axs, location='top')
    # plt.suptitle("Monitor function")
    # for i in range(axs.shape[1]):
    #     for j in range(axs.shape[0]):
    #         fig.colorbar(tcs[i][j], ax=axs[j, i])
    # plt.suptitle("Unsmoothed Monitor Function")
    # fig.savefig("fig2_unsmooth.jpg")

    P = mesh_seq.time_partition
    N = 40  # Constant from paper, but see the later comment
    for num_smooth in range(1, 16):
        smoothed_fns = []
        # num_smooth = 16
        print("Starting smoothing #{}".format(num_smooth))
        print(type(CellSize(mesh_seq[i])))
        for i in range(len(funcs)):
            fns = []
            function_space = funcs[i][0].function_space()
            dt = P.timesteps[i]
            delX = Constant(1 / (2 * n))
            K = Constant(N * delX**2 / dt)
            t_start, t_end = P.subintervals[i]
            t = t_start
            f_bound = Function(function_space)
            f_smooth = Function(function_space)
            v = TestFunction(f_bound.function_space())
            F = (
                inner(f_smooth - f_bound / dt, v)
                - K * (inner(dot(nabla_grad(f_smooth), nabla_grad(f_smooth)), v))
            ) * dx
            for j in range(len(funcs[i])):
                # this should be the same as while t < t_end - 1e-5:
                f_bound.assign(funcs[i][j])
                for k in range(num_smooth):
                    # Potentially have a k-loop here and smooth N times?
                    # print(k)
                    solve(F == 0, f_smooth)
                    f_bound.assign(f_smooth)
                t += dt
                fns.append(Function(function_space, val=f_smooth.dat.data))
            smoothed_fns.append(fns)

        # print(len(smoothed_fns), len(smoothed_fns[0]), type(smoothed_fns[0][0]))
        print(
            np.min(smoothed_fns[-1][-1].dat.data), np.max(smoothed_fns[-1][-1].dat.data)
        )

        fig, axs, tcs = plot_indicator_snapshots(
            smoothed_fns, time_partition, cmap="viridis", figsize=figsize
        )
        for i in range(axs.shape[1]):
            for j in range(axs.shape[0]):
                fig.colorbar(tcs[i][j], ax=axs[j, i])
        plt.suptitle("Smoothed Monitor Function ({})".format(num_smooth))
        fig.savefig("fig3_smooth_{}.jpg".format(num_smooth))
        print("Wrote to: {}".format("fig3_smooth_{}.jpg".format(num_smooth)))


if __name__ == "__main__":
    main()
