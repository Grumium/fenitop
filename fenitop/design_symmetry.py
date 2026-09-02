# SPDX-License-Identifier: GPL-3.0-or-later
"""Symmetric designs on a domain that is not reduced.

Written 2026 by Jan-David Förster for the TopoloGUI fork
(https://github.com/Grumium/fenitop).  Not part of upstream FEniTop.

Exploiting a symmetry plane means solving on half the domain, which delivers
a symmetric design as a by-product but demands that the load be symmetric or
antisymmetric about the plane.  When it cannot be done -- an uncertain load
direction, boundary conditions the plane does not mirror -- the design can
still be *made* symmetric, by projecting it onto the symmetric subspace at
every iteration.  That is this module.  It costs one sparse matrix-vector
product per iteration and constrains nothing about the loads, because it
never touches them.

The projector is the average over the symmetry group: a cell's density is
replaced by the mean over its orbit.  Applied to the design variable it is a
projection; applied to the sensitivity it is the same operator, which is what
the chain rule asks for, since the projector is symmetric.
"""

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

try:
    from scipy.spatial import cKDTree
except ImportError:                 # pragma: no cover - scipy is a dependency
    cKDTree = None


def _plane_transforms(planes, gdim):
    """Affine ``(R, t, name)`` maps for the planes worth mirroring about.

    Rotational symmetry is skipped: it comes with a quarter domain and a
    cyclic constraint of its own, so there is no full domain left to mirror.
    A plane whose type is not understood is skipped rather than guessed at.
    """
    transforms = []
    for plane in planes or []:
        ptype = plane.get("type", "axis_aligned")
        if ptype == "diagonal":
            normal = np.asarray(plane["normal"], dtype=float)[:gdim]
            norm = np.linalg.norm(normal)
            if norm == 0.0:
                continue
            normal = normal/norm
            # Mirroring about {x : n.(x - p) = 0} is x -> R x + 2 (n.p) n.  The
            # offset matters: detection places a diagonal plane through the
            # bounding-box centre, and taking it through the origin instead
            # mirrors a part that does not sit at the origin onto empty space.
            point = np.asarray(plane.get("point", np.zeros(gdim)),
                               dtype=float)[:gdim]
            transforms.append((np.eye(gdim) - 2.0*np.outer(normal, normal),
                               2.0*float(normal @ point)*normal,
                               plane.get("name") or
                               f"diagonal plane with normal {normal.round(3)}"))
        elif ptype == "axis_aligned":
            axis = {"x": 0, "y": 1, "z": 2}[plane["axis"]]
            if axis >= gdim:
                continue
            R = np.eye(gdim)
            R[axis, axis] = -1.0
            t = np.zeros(gdim)
            t[axis] = 2.0*float(plane["value"])
            transforms.append((R, t, plane.get("name") or
                               f"{plane['axis']} = {plane['value']:g}"))
    return transforms


def _close_group(transforms, gdim, limit=16):
    """The group the transforms generate, identity included.

    Two orthogonal mirrors generate four elements, three generate eight; a
    diagonal plane together with an axis-aligned one generates a rotation as
    well.  Closing the set is what makes the orbit average an actual
    projection -- averaging over the generators alone is not idempotent.
    """
    group = [(np.eye(gdim), np.zeros(gdim))]

    def key(element):
        R, t = element
        # ``+ 0.0`` turns -0.0 into 0.0.  Without it the identity reached by
        # composing a mirror with itself is not recognized as the identity
        # already in the set -- a mirror built as I - 2nn^T carries diagonal
        # entries a whisker below zero, which square to a negative zero -- and
        # the duplicate leaves the orbit average weighting one image twice.
        # It is then not a projection, and the design does not come out
        # symmetric: measured on a diagonal plane, a group of three where two
        # was right, and a residual asymmetry of 0.72 instead of round-off.
        return ((np.round(R, 9) + 0.0).tobytes(),
                (np.round(t, 9) + 0.0).tobytes())

    seen = {key(group[0])}
    changed = True
    while changed and len(group) < limit:
        changed = False
        for R1, t1 in list(group):
            for R2, t2 in ((R, t) for R, t, _ in transforms):
                candidate = (R2 @ R1, R2 @ t1 + t2)
                if key(candidate) not in seen:
                    seen.add(key(candidate))
                    group.append(candidate)
                    changed = True
                    if len(group) >= limit:
                        break
            if len(group) >= limit:
                break
    return group


class DesignSymmetry:
    """Averages the design over the orbit of a symmetry group.

    Built once per run: the map from a cell to its mirror images depends only
    on the mesh, so the per-iteration cost is a single ``MatMult``.
    """

    def __init__(self, comm, rho_field, planes):
        self.comm = comm
        self.mat = None
        space = rho_field.function_space
        gdim = space.mesh.geometry.dim
        transforms = _plane_transforms(planes, gdim)
        if not transforms:
            return

        index_map = space.dofmap.index_map
        num_local = index_map.size_local
        centroids = space.tabulate_dof_coordinates()[:num_local, :gdim]

        # The design lives in DG0, one value per cell, so the global index of
        # an owned cell is its offset in the rank-ordered concatenation --
        # which is what an allgather produces.
        gathered = comm.allgather(centroids)
        all_centroids = np.concatenate(gathered) if gathered else centroids
        num_global = all_centroids.shape[0]

        if cKDTree is None:
            raise ImportError("design symmetry needs scipy.spatial.cKDTree")
        tree = cKDTree(all_centroids)

        # A cell is matched to its image by proximity, so the tolerance has to
        # be well inside one cell.  The nearest neighbour distance is that
        # scale, and taking a quarter of the smallest one keeps a distorted
        # mesh from matching the wrong cell.
        spacing, _ = tree.query(all_centroids, k=2)
        tol = 0.25*float(np.min(spacing[:, 1])) if num_global > 1 else 1e-12

        def maps_onto_itself(R, t):
            """Whether every cell has an image under this map, on every rank."""
            distance, columns = tree.query(centroids @ R.T + t, k=1)
            far = distance > tol
            local = (int(np.sum(far)), float(np.max(distance)) if far.any()
                     else 0.0)
            counts = comm.allreduce(local[0], op=MPI.SUM)
            worst = comm.allreduce(local[1], op=MPI.MAX)
            return counts == 0, counts, worst, columns

        # A plane the mesh does not answer to is dropped, not fatal.  The
        # usual reason is a second plane the run reduced the domain on: after
        # halving, a plane whose normal is a different axis still maps what is
        # left onto itself, but a diagonal one folds it off the mesh
        # altogether.  Which of the two it is depends on the geometry, so it
        # is settled by asking the mesh rather than by a rule.
        usable = []
        for R, t, name in transforms:
            ok, count, worst, _ = maps_onto_itself(R, t)
            if ok:
                usable.append((R, t, name))
            elif comm.rank == 0:
                print(f"  ⚠️  Symmetric design about {name} skipped: "
                      f"{count} cells have no mirror image on this mesh "
                      f"(worst distance {worst:.3g}, tolerance {tol:.3g}).",
                      flush=True)
        if not usable:
            return

        # Compositions of maps that each send the mesh to itself do the same,
        # so closing the group cannot introduce a new failure -- but the
        # matches are found by proximity, so the assertion below is kept
        # rather than assumed.
        group = _close_group(usable, gdim)
        if len(group) < 2:
            return

        rows, cols = [], []
        for R, t in group:
            ok, count, worst, columns = maps_onto_itself(R, t)
            if not ok:
                raise ValueError(
                    "the mesh is not symmetric about a combination of the "
                    f"planes selected for a symmetric design: {count} cells "
                    f"have no mirror image within {tol:.3g} (worst distance "
                    f"{worst:.3g}).")
            rows.append(np.arange(num_local, dtype=np.int32)
                        + index_map.local_range[0])
            cols.append(columns.astype(np.int32))
        weight = 1.0/len(group)

        mat = PETSc.Mat().createAIJ(
            size=((num_local, num_global), (num_local, num_global)),
            nnz=(len(group), len(group)), comm=comm)
        mat.setUp()
        for row_block, col_block in zip(rows, cols):
            for row, col in zip(row_block, col_block):
                mat.setValue(int(row), int(col), weight,
                             addv=PETSc.InsertMode.ADD_VALUES)
        mat.assemble()
        self.mat = mat
        self.group_size = len(group)
        self._in = mat.createVecRight()
        self._out = mat.createVecLeft()

    def __bool__(self):
        return self.mat is not None

    def apply(self, values):
        """Average *values* -- owned design DOFs -- over the orbit."""
        if self.mat is None or values is None:
            return values
        self._in.array[:] = values
        self.mat.mult(self._in, self._out)
        return self._out.array.copy()
