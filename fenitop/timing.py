# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jan-David Förster


import time
from collections import defaultdict

class TimingStats:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TimingStats, cls).__new__(cls)
            cls._instance.reset()
        return cls._instance
    
    def reset(self):
        self.timers = defaultdict(float)
        self.counts = defaultdict(int)
        self.current_starts = {}
        
    def start(self, name):
        self.current_starts[name] = time.perf_counter()
        
    def stop(self, name):
        if name in self.current_starts:
            elapsed = time.perf_counter() - self.current_starts[name]
            self.timers[name] += elapsed
            self.counts[name] += 1
            del self.current_starts[name]
            return elapsed
        return 0.0
        
    def get_summary(self):
        return dict(self.timers)

# Global instance
stats = TimingStats()


# Tasks DOLFINx times internally that dominate problem set-up: mesh
# construction, entity/topology creation and dofmap building all happen before
# the optimization loop starts, so the loop timers above never see them.
# "Compute entities of dim = N" is the one to watch -- that is what dolfinx
# 0.11 parallelises via Topology.create_entities(num_threads=...).
#
# DOLFINx exposes no way to enumerate registered timers (list_timings only
# prints, from C++, so its output cannot be captured from Python), and
# timing() raises for a task that never ran.  So probe a candidate list
# instead: the dimension-parameterised names are generated rather than
# hardcoded, because they are exactly the interesting ones and their spelling
# depends on the mesh.
_DOLFINX_FIXED_TASKS = (
    "Build dofmap data",
    "Compute dof reordering map",
    "Compute local-to-local map",
    "Compute-local-to-global links for global/local adjacency list",
    "Distribute row-wise data (scalable)",
    "Gibbs-Poole-Stockmeyer ordering",
    "GPS: create_level_structure",
    "Init dofmap from element dofmap",
    "Topology: create",
    "Topology: determine shared index ownership",
    "Topology: determine vertex ownership groups (owned, undetermined, unowned)",
)


def _candidate_tasks(max_dim=3):
    tasks = list(_DOLFINX_FIXED_TASKS)
    for d in range(max_dim + 1):
        tasks.append(f"Compute entities of dim = {d}")
        for d1 in range(max_dim + 1):
            tasks.append(f"Compute connectivity {d}-{d1}")
    # Mesh builders and the dual-graph timer carry a suffix naming the cell
    # type, so spell out the variants rather than guessing one.
    for shape in ("BoxMesh (hexahedra)", "BoxMesh (tetrahedra)",
                  "RectangleMesh (triangles)", "RectangleMesh (quadrilaterals)"):
        tasks.append(f"Build {shape}")
    for kind in ("", " (mixed)", " (simplex)"):
        tasks.append(f"Compute local part of mesh dual graph{kind}")
        tasks.append(f"Compute non-local part of mesh dual graph{kind}")
    return tasks


def dolfinx_timings(tasks=None):
    """Return ``{task: {"count": int, "seconds": float}}`` for DOLFINx timers.

    Only tasks that actually ran are included, so the result is empty until a
    mesh has been built.  Never raises: a task this build never ran is skipped.
    """
    try:
        from dolfinx.common import timing
    except ImportError:
        return {}

    out = {}
    for task in (tasks if tasks is not None else _candidate_tasks()):
        try:
            count, delta = timing(task)
        except Exception:
            continue  # task never ran in this process
        if count:
            out[task] = {"count": int(count),
                         "seconds": delta.total_seconds()}
    return out


def report(comm=None, include_dolfinx=True):
    """Print the optimization-loop breakdown and DOLFINx's own set-up timers.

    Safe to call from a stand-alone script or under MPI; with a communicator
    only rank 0 prints.
    """
    if comm is not None and comm.rank != 0:
        return

    loop = stats.get_summary()
    if loop:
        total = sum(loop.values())
        print("\n  optimization loop:")
        for name in sorted(loop, key=loop.get, reverse=True):
            share = 100 * loop[name] / total if total else 0
            print(f"    {name:<14} {loop[name]:9.3f} s  "
                  f"({stats.counts[name]:>5} calls, {share:4.1f}%)")
        print(f"    {'TOTAL':<14} {total:9.3f} s")

    if include_dolfinx:
        internal = dolfinx_timings()
        if internal:
            print("\n  dolfinx set-up (mesh, topology, dofmap):")
            for name, d in sorted(internal.items(),
                                  key=lambda kv: -kv[1]["seconds"]):
                print(f"    {name:<58} {d['seconds']:8.3f} s  "
                      f"({d['count']:>4}x)")
