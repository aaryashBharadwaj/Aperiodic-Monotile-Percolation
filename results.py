import datetime
import numpy as np

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
        extra_meta: dict | None = None,
    ):
        self.tiling_type = tiling_type
        self.seed = seed
        self.timestamp = datetime.datetime.now().isoformat(timespec="seconds")
        self.trials = trials
        self.L_values = np.asarray(L_values, dtype=float)
        self.raw_SI = [np.asarray(r, dtype=float) for r in raw_SI]
        self.raw_SU = [np.asarray(r, dtype=float) for r in raw_SU]
        self.raw_BI = [np.asarray(r, dtype=float) for r in raw_BI] if raw_BI is not None else None
        self.raw_BU = [np.asarray(r, dtype=float) for r in raw_BU] if raw_BU is not None else None
        self.extra_meta = extra_meta or {}

    @property
    def has_bond(self) -> bool:
        return self.raw_BI is not None and self.raw_BU is not None

    @property
    def n_L(self) -> int:
        return len(self.L_values)
    # used to get the means and deviations used for plotting
    def means_stds(self):
        mSI = np.array([r.mean() for r in self.raw_SI])
        sSI = np.array([r.std()  for r in self.raw_SI])
        mSU = np.array([r.mean() for r in self.raw_SU])
        sSU = np.array([r.std()  for r in self.raw_SU])
        if self.has_bond:
            mBI = np.array([r.mean() for r in self.raw_BI])
            sBI = np.array([r.std()  for r in self.raw_BI])
            mBU = np.array([r.mean() for r in self.raw_BU])
            sBU = np.array([r.std()  for r in self.raw_BU])
        else:
            zeros = np.zeros(self.n_L)
            mBI = sBI = mBU = sBU = zeros
        return mSI, sSI, mSU, sSU, mBI, sBI, mBU, sBU

    # Saves the results to the results_output folder
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

        np.savez(path, **arrays)
        # np.savez appends .npz if not present
        if not path.endswith(".npz"):
            path = path + ".npz"
        print(f"[results] Saved to {path}")
        return path

    # Loads a file from results_output folder
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

        # Collect any remaining meta_ keys as extra_meta
        extra_meta = {}
        skip = {"meta_tiling_type", "meta_seed", "meta_timestamp",
                "meta_trials", "meta_has_bond"}
        for k in data.files:
            if k.startswith("meta_") and k not in skip:
                extra_meta[k[5:]] = data[k].item()

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
        )
        obj.timestamp = str(data["meta_timestamp"])
        return obj