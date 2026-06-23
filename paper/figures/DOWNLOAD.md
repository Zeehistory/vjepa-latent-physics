# Figures to download from the cluster

The paper (`paper/main.tex`) references 3 figures. They are produced by the cluster runs and are **not
in this repo** — pull them via `scp` into this `paper/figures/` directory with the exact filenames
below. Cluster host base path: `zss8@<cluster>:/home/zss8/project_pi_jks79/zss8/vjepa`.

| Figure in paper | Save as (in `paper/figures/`) | Cluster source path |
|---|---|---|
| Fig. 1 — Step-1 layerwise accuracy | `category_accuracy_by_layer.png` | `outputs/analysis/physics_iq/category_probe/raw/category_accuracy_by_layer.png` |
| Fig. 2 — fluid/solid cosine heatmap | `category_cosine_layer23.png` | `outputs/analysis/physics_iq/category_probe/raw/category_cosine_layer23.png` |
| Fig. 3 — steering filmstrip | `steer_filmstrip.png` | `outputs/analysis/steer_category/fluid_dynamics_to_solid_mechanics_diff_means/<scenario>_filmstrip.png` |

## One-liner (edit `<cluster>` and the scenario id)

```bash
BASE=zss8@<cluster>:/home/zss8/project_pi_jks79/zss8/vjepa
DST=paper/figures

scp $BASE/outputs/analysis/physics_iq/category_probe/raw/category_accuracy_by_layer.png $DST/
scp $BASE/outputs/analysis/physics_iq/category_probe/raw/category_cosine_layer23.png    $DST/
# chosen scenario: juice-in-water. Adjust the exact id if the filename differs on disk.
scp "$BASE/outputs/analysis/steer_category/fluid_dynamics_to_solid_mechanics_diff_means/"*juice*water*_filmstrip.png $DST/steer_filmstrip.png
```

## Notes

- The cosine heatmap layer suffix (`layer23`) is the deepest raw-probe layer; if your run's deepest
  layer differs, rename accordingly and update the `\includegraphics` line in `main.tex`.
- If the `*_filmstrip.png` were not written (older run), re-run with `NUM_SAMPLES=6 sbatch
  scripts/slurm_steer_category.sh`, which also writes `controllability.png` (an optional 4th figure
  showing the readout-saturation step function).
- The optional confusion matrices live alongside Fig. 1 as `confusion_layer<NN>.png` if you want one.
