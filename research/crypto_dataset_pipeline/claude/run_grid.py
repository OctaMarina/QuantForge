"""Pre-declared modelling grid (written before any out-of-sample result was seen)."""
import warnings
warnings.filterwarnings("ignore")
import mad_lib as M

JOBS = []
for h in ["15m", "2h", "1d"]:
    JOBS.append(("PV", "mean", h, f"y_{h}"))
    for s in M.SETS:
        for m in ["ridge", "lgbm"]:
            JOBS.append((s, m, h, f"y_{h}"))
# bounce control for the short horizon: entry at t+60s
JOBS.append(("PV", "mean", "15m", "y_15m_d60"))
for s in ["PV", "OWN", "ALL"]:
    for m in ["ridge", "lgbm"]:
        JOBS.append((s, m, "15m", "y_15m_d60"))

if __name__ == "__main__":
    d = M.build_panel()
    M.run_grid(d, JOBS, tag="main")
    print("DONE")
