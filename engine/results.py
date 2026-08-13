import datetime
import numpy as np

# Container for one run's raw per-trial data + metadata, with lossless npz save/load (one key per L).
# The published p_c comes from the raw crossing arrays (raw_SI/SU/BI/BU) via extrapolate_pc_raw; the
# direction-bias (pR/pD) and d_f (s_union/s_inter) arrays are OPTIONAL companions that now live IN this
# one file rather than in separate _iso.npz / _exp.npz sidecars (older runs kept them outside; load()
# reads them here, and gui_backend still falls back to the old sidecars for those older files).


def _aslist(x):
    """Coerce a per-L sequence of trial arrays to a list of float arrays; None (or any missing entry)
    -> None, so a partially-absent optional channel is simply treated as absent."""
    if x is None or any(a is None for a in x):
        return None
    return [np.asarray(a, dtype=float) for a in x]


class PercolationResults:
    # all the properties of each entry of the results
    def __init__(
        self,
        tiling_type: str,
        seed: int,
        trials: int,
        L_values,
        raw_SI,
        raw_SU,
        raw_BI=None,
        raw_BU=None,
        # direction-bias crossings (was the _iso.npz companion)
        raw_pR=None,
        raw_pD=None,
        raw_bond_pR=None,
        raw_bond_pD=None,
        # d_f incipient-cluster sizes, union/intersection onset, site & bond (was the _exp.npz companion)
        raw_smax=None,
        raw_smax_inter=None,
        raw_bond_smax=None,
        raw_bond_smax_inter=None,
        extra_meta: dict | None = None,
    ):
        self.tiling_type = tiling_type
        self.seed = seed
        self.timestamp = datetime.datetime.now().isoformat(timespec="seconds")
        self.trials = trials
        self.L_values = np.asarray(L_values, dtype=float)
        self.raw_SI = [np.asarray(r, dtype=float) for r in raw_SI]
        self.raw_SU = [np.asarray(r, dtype=float) for r in raw_SU]
        self.raw_BI = _aslist(raw_BI)
        self.raw_BU = _aslist(raw_BU)
        self.raw_pR = _aslist(raw_pR)
        self.raw_pD = _aslist(raw_pD)
        self.raw_bond_pR = _aslist(raw_bond_pR)
        self.raw_bond_pD = _aslist(raw_bond_pD)
        self.raw_smax = _aslist(raw_smax)
        self.raw_smax_inter = _aslist(raw_smax_inter)
        self.raw_bond_smax = _aslist(raw_bond_smax)
        self.raw_bond_smax_inter = _aslist(raw_bond_smax_inter)
        self.extra_meta = extra_meta or {}

    # per-L key prefix <-> attribute, for the optional channels (saved/loaded uniformly)
    _OPT = [("pR", "raw_pR"), ("pD", "raw_pD"), ("bpR", "raw_bond_pR"), ("bpD", "raw_bond_pD"),
            ("smax", "raw_smax"), ("smaxI", "raw_smax_inter"),
            ("bsmax", "raw_bond_smax"), ("bsmaxI", "raw_bond_smax_inter")]

    @property
    def has_bond(self) -> bool:
        return self.raw_BI is not None and self.raw_BU is not None

    @property
    def n_L(self) -> int:
        return len(self.L_values)

    # Saves the results to the paper_results/npz folder
    def save(self, path: str) -> str:

        arrays: dict[str, np.ndarray] = {}

        # Scalar metadata
        arrays["meta_tiling_type"]        = np.array(self.tiling_type)
        arrays["meta_seed"]               = np.array(self.seed)
        arrays["meta_timestamp"]          = np.array(self.timestamp)
        arrays["meta_trials"]             = np.array(self.trials)
        arrays["meta_has_bond"]           = np.array(self.has_bond)

        for k, v in self.extra_meta.items():
            arrays[f"meta_{k}"] = np.array(v)

        arrays["L_values"] = self.L_values

        for i, (si, su) in enumerate(zip(self.raw_SI, self.raw_SU)):
            arrays[f"raw_SI_{i}"] = si
            arrays[f"raw_SU_{i}"] = su

        if self.has_bond:
            for i, (bi, bu) in enumerate(zip(self.raw_BI, self.raw_BU)):
                arrays[f"raw_BI_{i}"] = bi
                arrays[f"raw_BU_{i}"] = bu

        # optional channels (direction-bias + d_f) -- only written when present
        for prefix, attr in self._OPT:
            data = getattr(self, attr)
            if data is not None:
                for i, a in enumerate(data):
                    arrays[f"{prefix}_{i}"] = a

        np.savez(path, **arrays)
        # np.savez appends .npz if not present
        if not path.endswith(".npz"):
            path = path + ".npz"
        print(f"[results] Saved to {path}")
        return path

    # Loads a file from paper_results/npz folder
    # Inverse of save(): read scalars, L values, and the raw per-trial arrays back. Array-valued
    # meta (e.g. the center tuple) is kept as-is rather than .item()'d, which throws on non-scalars.
    @classmethod
    def load(cls, path: str) -> "PercolationResults":
        """Load from a .npz file produced by save()."""
        data = np.load(path, allow_pickle=True)

        tiling_type = str(data["meta_tiling_type"])
        seed        = int(data["meta_seed"])
        trials      = int(data["meta_trials"])
        has_bond    = bool(data["meta_has_bond"])
        L_values    = data["L_values"]
        n_L         = len(L_values)

        raw_SI = [data[f"raw_SI_{i}"] for i in range(n_L)]
        raw_SU = [data[f"raw_SU_{i}"] for i in range(n_L)]
        raw_BI = [data[f"raw_BI_{i}"] for i in range(n_L)] if has_bond else None
        raw_BU = [data[f"raw_BU_{i}"] for i in range(n_L)] if has_bond else None

        # optional channels: present only if their per-L keys were written
        def opt(prefix):
            return [data[f"{prefix}_{i}"] for i in range(n_L)] if f"{prefix}_0" in data.files else None
        kw = {attr: opt(prefix) for prefix, attr in cls._OPT}

        # Collect any remaining meta_ keys as extra_meta
        extra_meta = {}
        skip = {"meta_tiling_type", "meta_seed", "meta_timestamp",
                "meta_trials", "meta_has_bond"}
        for k in data.files:
            if k.startswith("meta_") and k not in skip:
                v = data[k]
                try:
                    extra_meta[k[5:]] = v.item()      # scalar metadata
                except (ValueError, AttributeError):
                    extra_meta[k[5:]] = v             # array-valued metadata (e.g. center tuple)

        obj = cls(
            tiling_type=tiling_type,
            seed=seed,
            trials=trials,
            L_values=L_values,
            raw_SI=raw_SI,
            raw_SU=raw_SU,
            raw_BI=raw_BI,
            raw_BU=raw_BU,
            extra_meta=extra_meta,
            **kw,
        )
        obj.timestamp = str(data["meta_timestamp"])
        return obj
