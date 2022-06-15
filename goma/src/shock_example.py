"""
A simple example file to set up a problem and QOI.
"""
from firedrake import *
from pyroteus import *
from pyroteus_adjoint import *


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
            + inner(dot(u, nabla_grad(u)), v) * dx  # noqa: W503
            + nu * inner(grad(u), grad(v)) * dx  # noqa: W503
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
    return {"u": VectorFunctionSpace(mesh, "CG", 2)}


def get_initial_condition(mesh_seq):
    fs = mesh_seq.function_spaces["u"][0]
    x, y = SpatialCoordinate(mesh_seq[0])
    # An initial condition that propagates to both ends
    return {"u": interpolate(as_vector([sin(pi * (x - 0.5)), 0]), fs)}


def get_qoi(mesh_seq, solutions, i):
    def end_time_qoi():
        u = solutions["u"]
        return inner(u, u) * ds(2)

    return end_time_qoi


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
    solutions = mesh_seq.solve_adjoint()
    # print("QOI: {}".format(mesh_seq.get_qoi()))
    fig, axes, tcs = plot_snapshots(
        solutions,
        time_partition,
        "u",
        "adjoint",
        levels=np.linspace(0, 0.8, 9),
    )
    fig.savefig("burgers2.jpg")
    solutions, indicators = mesh_seq.indicate_errors(
        enrichment_kwargs={"enrichment_method": "h"}
    )
    print(indicators.shape)
    for field, sols in solutions.items():
        fwd_outfile = File(f"test_burgers2/{field}_forward.pvd")
        adj_outfile = File(f"test_burgers2/{field}_adjoint.pvd")
        for i, mesh in enumerate(mesh_seq):
            for sol in sols["forward"][i]:
                fwd_outfile.write(sol)
            for sol in sols["adjoint"][i]:
                adj_outfile.write(sol)

    fig, axes, tcs = plot_indicator_snapshots(
        indicators, time_partition, levels=50
    )  # noqa: E501
    fig.savefig("ee-adj.jpg")


if __name__ == "__main__":
    main()
