"""Load pickles written by the cluster container's numpy under the host's numpy.

The container (train_q.sif) ships numpy 2.x, which pickles array reconstructors
as `numpy._core.*`. The host venv is on numpy 1.26, where those live at
`numpy.core.*`, so a straight `pickle.load` raises
`ModuleNotFoundError: No module named 'numpy._core.numeric'`.

The remap is done inside a custom Unpickler rather than by aliasing entries in
`sys.modules`: a global alias is visible to every later import, and pandas --
which is compiled against numpy 1.x and probes `numpy._core` itself -- then fails
with `ImportError: numpy._core.multiarray failed to import`. Scoping the remap to
the unpickling call keeps it invisible to everything else.

Use `say_no.cp._compat.load(path)` wherever a cluster-produced pkl is read.
"""

import pickle


class _NumpyCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            try:
                return super().find_class(module, name)
            except (ModuleNotFoundError, ImportError):
                # numpy 1.x host: the same symbols live under numpy.core
                return super().find_class(module.replace("numpy._core", "numpy.core", 1), name)
        return super().find_class(module, name)


def load(path):
    """pickle.load for a dump written by a possibly-newer numpy."""
    with open(path, "rb") as f:
        return _NumpyCompatUnpickler(f).load()
