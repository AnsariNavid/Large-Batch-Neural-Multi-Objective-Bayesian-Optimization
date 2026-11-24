# LBN-MOBO

**Large-Batch Neural Multi-Objective Bayesian Optimization**

[Paper](https://arxiv.org/abs/2306.01095) · [Open in Colab](https://colab.research.google.com/github/AnsariNavid/lbn_mobo/blob/main/LBN-MOBO.ipynb) · [Demo notebook](LBN-MOBO.ipynb) · [Citation](#citation)

Engineering design often involves several competing objectives and an expensive process for evaluating candidate designs. When simulations or experiments can evaluate thousands of candidates in parallel, the number of sequential optimization rounds becomes an important practical constraint. LBN-MOBO addresses this setting by combining neural surrogate models with a scalable acquisition strategy for large batches.

The method learns both predicted performance and epistemic uncertainty, then uses non-dominated sorting to select candidates across these quantities. This supports exploration of uncertain regions alongside improvement of the current design trade-offs. The repository provides a **30-variable, two-objective ZDT3 demonstration**, saved baseline results, and examples of the parallel workflows used for engineering applications.

## Research paper

**[Large-Batch, Iteration-Efficient Neural Bayesian Design Optimization](https://arxiv.org/abs/2306.01095)**

Navid Ansari, Alireza Javanmardi, Eyke Hüllermeier, Hans-Peter Seidel, and Vahid Babaei.

arXiv:2306.01095, first submitted in 2023; revised version v4, September 2024.

The paper evaluates LBN-MOBO on ZDT and DTLZ benchmarks, OpenFOAM airfoil design, and simulated printer-gamut exploration. The engineering experiments include batches of **15,000 airfoil candidates** and **20,000 printer candidates** per round. The printer experiment uses a learned forward model of a 44-ink printing system; these batch sizes describe the paper's experimental configurations, not the default notebook.

### What is included

| Component | Scope |
| --- | --- |
| ZDT3 notebook | Main entry point. Generates its own initial data, trains a deep ensemble, acquires candidates, and plots hypervolume and objective-space results. |
| Baseline data | Precomputed DGEMO, TSEMO, USeMO, and NSGA-II results used by the notebook. The notebook does not rerun these methods. |
| Parallel examples | SLURM and application-specific scripts for ZDT3, airfoil design, and printer gamut. They require configuration; the engineering examples also depend on external assets that are not included. |

## Quick start

### Option 1: Google Colab

1. [Download the repository ZIP](https://github.com/AnsariNavid/lbn_mobo/archive/refs/heads/main.zip) and extract it.
2. Upload the extracted **`lbn_mobo-main`** folder to the root of **My Drive**, preserving its subfolders. The notebook expects:

   ```text
   /content/drive/MyDrive/lbn_mobo-main
   ```

3. [Open the notebook in Colab](https://colab.research.google.com/github/AnsariNavid/lbn_mobo/blob/main/LBN-MOBO.ipynb), or open the uploaded `LBN-MOBO.ipynb` from Google Drive.
4. Run the Drive-mount and working-directory cells. If your folder has a different name or location, update the `os.chdir(...)` cell.
5. Run the dependency-installation cell, then the optimization and plotting cells in order.

The notebook trains the ten ensemble members sequentially. Training uses CUDA when PyTorch detects a GPU, so a GPU runtime can speed up that stage. Acquisition runs on the CPU.

For a shorter first run, set `iter_max = 1` and keep `Batch_size = 1000`. This still trains ten networks and performs a full acquisition search. The original README reported approximately 1.6 hours on a Colab CPU for its demonstration; treat that as a historical reference, since runtime depends on the configuration and environment.

### Option 2: Local Jupyter

Clone the repository and create an isolated Python environment:

```bash
git clone https://github.com/AnsariNavid/lbn_mobo.git
cd lbn_mobo
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch numpy scipy pandas matplotlib pymoo pygmo jupyterlab
jupyter lab LBN-MOBO.ipynb
```

The activation command above is for Linux/macOS. On Windows PowerShell, use `.venv\Scripts\Activate.ps1`.

Start Jupyter from the repository root. Skip the `google.colab` Drive-mount cell and the Colab-specific `os.chdir(...)` cell. With the dependencies installed in this environment, the notebook's installation cell can also be skipped. Run the optimization and plotting cells in order.

The dependencies above follow the imports used by the main demo. The repository does not provide a pinned environment or lockfile, so these commands are a setup starting point, not a verified historical environment. The notebook additionally installs `Platypus-Opt`, although the main demo does not import it. The separate `find_usefull_pareto.py` helper imports `oapackage`; that helper is not used by the notebook.

## Configuration

The main notebook cell contains:

```python
Batch_size = 1000
iter_max = 5
problem_name = "zdt3"
```

| Setting | Default | Meaning |
| --- | --- | --- |
| `Batch_size` | `1000` | Initial random sample count and nominal number of new candidates requested per round. |
| `iter_max` | `5` | Number of training and acquisition rounds after initialization. |
| `problem_name` | `"zdt3"` | Problem used by the initial-data generator. The acquisition oracle is configured separately in `query_oracle.py`. |
| Ensemble size | `10` | Networks trained in the notebook and loaded by the acquisition surrogate. |
| Hidden layers | `[100, 50, 100]` | Surrogate architecture in `Forward_BNN.py` and `BO_surrogate_function_uncertainty.py`. |
| Training epochs | `60` | Epochs per network per round in `Forward_BNN.py`. |
| Training minibatch size | `10` | Up to ten samples per optimizer step; separate from the acquisition `Batch_size`. |
| NSGA-II population / generations | `1000` / `100` | Acquisition-search settings in `Acquisition_4_D.py`. |

**Use `Batch_size` values of 1,000, 2,000, 3,000, and so on.** The notebook launches `int(Batch_size / 1000)` independent acquisition searches. Values below 1,000 launch no acquisition search, and other values are truncated when calculating the number of searches.

Each search returns the nondominated solutions in `res.X`, which can contain fewer points than the population size. The number of evaluated candidates can therefore be smaller than `Batch_size`. Check the saved array sizes when accounting for evaluation budgets.

Changing `problem_name` alone does not switch the whole workflow to another problem. See [Adapting the method](#adapting-the-method).

## How it works

For a problem with $M$ objectives, the paper's **2MD acquisition** considers a $2M$-dimensional space containing predicted objective values and their epistemic uncertainties. NSGA-II searches for trade-offs in this augmented space. Independent searches with different seeds can be combined to construct larger batches.

The supplied ZDT3 implementation uses an ensemble of ten neural networks with different initializations and activation functions. For each design $x$, it estimates an objective mean $\mu_j(x)$ and an ensemble variance $v_j(x)$. Since ZDT3 is a minimization problem, the acquisition code minimizes:

$$
\left[\mu_1(x),\ \mu_2(x),\ -v_1(x),\ -v_2(x)\right].
$$

The negative variance terms favor exploration of uncertain regions. The `4_D` in `Acquisition_4_D.py` refers to these four acquisition objectives; the design space still has 30 variables.

Each optimization round performs the following steps:

1. Fit the ensemble using the accumulated observations, with a separate 90/10 training and validation split for each member.
2. Search the augmented objective space with NSGA-II.
3. Evaluate the selected designs with the ZDT3 oracle.
4. Append their true objective values to the training data and repeat.

In the notebook, both ensemble training and repeated acquisition searches run sequentially. The parallel examples show how these independent jobs were distributed in the research workflow. The paper studies additional surrogate and noise-handling variants; the main notebook demonstrates the deep-ensemble ZDT3 configuration.

## Repository guide

| File or directory | Purpose |
| --- | --- |
| [`LBN-MOBO.ipynb`](LBN-MOBO.ipynb) | Main demonstration, optimization loop, and plots. |
| [`Random_dataset_gen.py`](Random_dataset_gen.py) | Samples the initial 30-dimensional designs uniformly in `[0, 1]` and evaluates the selected pymoo problem. |
| [`Forward_BNN.py`](Forward_BNN.py) | Trains one ensemble member and saves its PyTorch state dictionary. |
| [`layer_config_forward.py`](layer_config_forward.py) | Defines the multilayer perceptron and activation-function choices. |
| [`BO_surrogate_function_uncertainty.py`](BO_surrogate_function_uncertainty.py) | Loads the ensemble and defines the four-objective acquisition problem. |
| [`Acquisition_4_D.py`](Acquisition_4_D.py) | Runs NSGA-II, evaluates selected candidates, and saves each acquisition result. |
| [`acquisition_organizeer.py`](acquisition_organizeer.py) | Combines acquisition runs and saves their design and objective arrays. |
| [`query_oracle.py`](query_oracle.py) | Wraps the ground-truth ZDT3 evaluator. |
| [`hv_calculator.py`](hv_calculator.py) | Computes cumulative normalized hypervolume for the plots. |
| [`find_usefull_pareto.py`](find_usefull_pareto.py) | Additional two-objective Pareto-filtering helper using `oapackage`. |
| [`Dataset/`](Dataset/), [`Models/`](Models/), [`Acquisition/`](Acquisition/), [`surrogate_pareto/`](surrogate_pareto/) | Output locations populated during a run; initially contain placeholder files. |
| [`DGEMO_data/`](DGEMO_data/), [`TSEMO_data/`](TSEMO_data/), [`USeMO_data/`](USeMO_data/), [`NSGA2_data/`](NSGA2_data/) | Saved baseline objective arrays. |
| [`GT_data.mat`](GT_data.mat) | Reference ZDT3 front used in the final plot. |
| [`execution_times.mat`](execution_times.mat) | Saved timing example, overwritten by the notebook with timings from the current run. |
| [`Parallel_computing/`](Parallel_computing/) | Application and cluster examples described below. |

## Data and outputs

**No external dataset or pretrained checkpoint is needed for the ZDT3 demo.** The initial observations are generated analytically and the ensemble is trained during execution. Keep the placeholder output directories when copying the repository.

| Generated path | Contents |
| --- | --- |
| `Dataset/dset_0.mat` | Initial observations in a MATLAB structure named `dset`, with fields `X` and `Y`. |
| `Dataset/dset_<k>.mat` | Accumulated observations after `k` acquisition rounds. |
| `Models/iter_<i>/mu_net_<n>.ckpt` | Ensemble member `n`, numbered 0 through 9 in the main notebook. |
| `Acquisition/iter_<i>/run_<r>.mat` | One acquisition run, stored as structure `data` with fields `X` and `Y`. |
| `surrogate_pareto/ParetoSet_<i>.mat` | Acquired designs under key `ParetoSet`. |
| `surrogate_pareto/ParetoFront_<i>.mat` | Their true objective evaluations under key `ParetoFront`. |
| `execution_times.mat` | Per-round wall-clock durations under key `elapsed_times`. |

Iteration indices `i` start at zero. For ZDT3, `X` has shape `(N, 30)` and `Y` has shape `(N, 2)`. A dataset can be loaded as follows:

```python
from scipy.io import loadmat

dataset = loadmat("Dataset/dset_0.mat", simplify_cells=True)["dset"]
X = dataset["X"]
Y = dataset["Y"]
print(X.shape, Y.shape)
```

Despite their names, the generated `ParetoFront_<i>.mat` files store oracle evaluations of candidates selected in the augmented acquisition space. Those candidates need not all be nondominated in the true two-objective space.

### Interpreting the plots

The notebook plots cumulative hypervolume and compares the acquired solutions with the reference ZDT3 front. `hv_calculator.py` uses reference point **`[11, 11]`** and divides the hypervolume by **121**. Its LBN-MOBO curve accumulates the acquired candidates; the initial random dataset is not included in those per-round files.

The baseline curves use the committed `.mat` files. DGEMO iterations 6 and 7 and USeMO iteration 7 contain empty arrays. Their cumulative curves therefore carry forward the previous value at those entries; these files do not establish additional successful optimization rounds.

The notebook's final plot uses the rounds actually run through `iter_max`, even though its existing markdown caption mentions ten iterations. Exact reproduction of the paper also requires matching its experiment-specific budgets, seeds, environment, and hardware.

### Re-running experiments

Use a fresh repository copy or a separate working directory for each experiment, and preserve previous outputs before rerunning. The notebook writes to fixed paths and uses exact file counts to detect completion. Stale files can mix outputs from different configurations or leave the loop waiting indefinitely.

The code sets some PyTorch and NSGA-II seeds, but initial NumPy sampling is not seeded by the notebook. Record the initial dataset, random seeds, package versions, and hardware if you need repeatable comparisons.

## Adapting the method

For a new design problem, update the following components consistently:

1. **Oracle:** implement the true objective evaluation in `query_oracle.py`, returning an `(N, M)` array. Keep the initial-data generator consistent with this evaluator.
2. **Design space:** change the sampling strategy and bounds in `Random_dataset_gen.py`, and `n_var`, `xl`, and `xu` in the acquisition problem.
3. **Surrogate:** update input and output sizes in both training and model-loading code. Keep architecture definitions and ensemble sizes consistent.
4. **Acquisition:** use `2 * M` objectives and update all hard-coded two-objective slices and array shapes. Check the signs for any objectives to be maximized.
5. **Evaluation:** replace the ZDT3 reference front, hypervolume reference point and normalization, and plotting code with choices appropriate to the new objectives and their scales.

The main notebook assumes 30 input variables, two objectives, and bounds `[0, 1]`. Supporting a different number of objectives or general design constraints requires code changes.

## Parallel and engineering examples

These directories preserve examples of the research execution workflows. They are starting points for adapting the method to a cluster.

| Directory | Workflow | Configuration or assets needed |
| --- | --- | --- |
| [`Parallel_computing/ZDT3/`](Parallel_computing/ZDT3/) | SLURM-based ensemble training and acquisition for ZDT3. | Set SSH host, working directory, environment, partitions, job-array sizes, and output directories. Align script names and completion counts. |
| [`Parallel_computing/Airfoil_design/`](Parallel_computing/Airfoil_design/) | Bézier-GAN geometry generation with simulation-based airfoil evaluation. | Trained GAN files under `trained_gan/5.0_0.2`, the imported `surrogate` package, `Airfoil_simulation_1/2/3` modules and CFD setup, plus the referenced container image and site configuration. These assets are not included. The GAN code uses TensorFlow 1-style APIs. |
| [`Parallel_computing/Printer_gamut/`](Parallel_computing/Printer_gamut/) | Gamut exploration with a learned 44-ink printer oracle. | Ten pretrained oracle checkpoints, `oracle_0.ckpt` through `oracle_9.ckpt`, referenced at `../Oracle_Relu_ensemble/Models_44_inks/`, plus cluster configuration. The checkpoints are not included. |

For example, the ZDT3 orchestration scripts refer to `Forward_UANA_diverse_activation.sh`, while the supplied training launcher is named `Forward_BNN.sh`. Resolve this mismatch before submitting jobs. Inspect each application's paths and dependencies rather than assuming the root notebook's configuration applies to it.

The folder name **`Printer_gamut`** is case-sensitive. The ZDT3 notebook is the recommended starting point for understanding the algorithm before adapting these workflows.

## Citation

If you use LBN-MOBO in your research, please cite the paper. The entry below refers to the revised 2024 manuscript:

```bibtex
@misc{ansari2024lbnmobo,
  title         = {Large-Batch, Iteration-Efficient Neural Bayesian Design Optimization},
  author        = {Ansari, Navid and Javanmardi, Alireza and H{\"u}llermeier, Eyke
                   and Seidel, Hans-Peter and Babaei, Vahid},
  year          = {2024},
  eprint        = {2306.01095},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  doi           = {10.48550/arXiv.2306.01095},
  url           = {https://arxiv.org/abs/2306.01095v4}
}
```

## Questions and license

For implementation questions or missing research assets, please [open an issue](https://github.com/AnsariNavid/lbn_mobo/issues). A code license has not been specified in this repository.
