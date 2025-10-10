import json
import logging
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributed
from torchmetrics import Metric


plt.rcParams.update({"font.size": 8})
mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)  # Only show WARNING and above

colors = [
    "#1f77b4",  # muted blue
    "#d62728",  # brick red
    "#7f7f7f",  # middle gray
    "#2ca02c",  # cooked asparagus green
    "#ff7f0e",  # safety orange
    "#9467bd",  # muted purple
    "#8c564b",  # chestnut brown
    "#e377c2",  # raspberry yogurt pink
    "#bcbd22",  # curry yellow-green
    "#17becf",  # blue-teal
]

error_type = {
    "TotalRMSE": (
        [("rmse_e", "RMSE E [meV]"), ("rmse_f", "RMSE F [meV / A]")],
        [("energy", "Energy per atom [eV]"), ("force", "Force [eV / A]")],
    ),
    "PerAtomRMSE": (
        [("rmse_e_per_atom", "RMSE E/atom [meV]"), ("rmse_f", "RMSE F [meV / A]")],
        [("energy", "Energy per atom [eV]"), ("force", "Force [eV / A]")],
    ),
    "PerAtomRMSEstressvirials": (
        [
            ("rmse_e_per_atom", "RMSE E/atom [meV]"),
            ("rmse_f", "RMSE F [meV / A]"),
            ("rmse_stress", "RMSE Stress [meV / A^3]"),
        ],
        [
            ("energy", "Energy per atom [eV]"),
            ("force", "Force [eV / A]"),
            ("stress", "Stress [eV / A^3]"),
        ],
    ),
    "PerAtomMAEstressvirials": (
        [
            ("mae_e_per_atom", "MAE E/atom [meV]"),
            ("mae_f", "MAE F [meV / A]"),
            ("mae_stress", "MAE Stress [meV / A^3]"),
        ],
        [
            ("energy", "Energy per atom [eV]"),
            ("force", "Force [eV / A]"),
            ("stress", "Stress [eV / A^3]"),
        ],
    ),
    "TotalMAE": (
        [("mae_e", "MAE E [meV]"), ("mae_f", "MAE F [meV / A]")],
        [("energy", "Energy per atom [eV]"), ("force", "Force [eV / A]")],
    ),
    "PerAtomMAE": (
        [("mae_e_per_atom", "MAE E/atom [meV]"), ("mae_f", "MAE F [meV / A]")],
        [("energy", "Energy per atom [eV]"), ("force", "Force [eV / A]")],
    ),
    "DipoleRMSE": (
        [
            ("rmse_mu_per_atom", "RMSE MU/atom [mDebye]"),
            ("rel_rmse_f", "Relative MU RMSE [%]"),
        ],
        [("dipole", "Dipole per atom [Debye]")],
    ),
    "DipoleMAE": (
        [("mae_mu", "MAE MU [mDebye]"), ("rel_mae_f", "Relative MU MAE [%]")],
        [("dipole", "Dipole per atom [Debye]")],
    ),
    "EnergyDipoleRMSE": (
        [
            ("rmse_e_per_atom", "RMSE E/atom [meV]"),
            ("rmse_f", "RMSE F [meV / A]"),
            ("rmse_mu_per_atom", "RMSE MU/atom [mDebye]"),
        ],
        [
            ("energy", "Energy per atom [eV]"),
            ("force", "Force [eV / A]"),
            ("dipole", "Dipole per atom [Debye]"),
        ],
    ),
    "EffectiveCouplingLoss": (
        [
            ("mae_graph_wide_coupling","MAE Graph Wide Coupling [meV]"),
            ("rmse_graph_wide_coupling","RMSE Graph Wide Coupling [meV]")

        ],
        [   
            ("effective_coupling", "Effective Coupling [meV]"),

        ]
    ),
    "ClassifierAccuracy" : (
        [
            ("accuracy", "Accuracy of Classifier [%]"),
        ],
        [
            ("coupling_class", "Coupling Class"),

        ]
    ),
    "GatedEffectiveCouplingLoss": (
        [
            ("accuracy", "Accuracy of Classifier [%]"),
            ("mae_graph_wide_coupling","MAE Graph Wide Coupling [meV]"),
            ("rmse_graph_wide_coupling","RMSE Graph Wide Coupling [meV]")
        ],
        # [
        #     ("coupling_class", "Coupling Class"),

        # ],
        [
            ("effective_coupling","Effective Coupling")
        ]
    )

}


KIND_MAP = {0: "mapper", 1: "attn", 2: "pooled"}

def _activations_to_df(named_results: dict) -> "pd.DataFrame":
    """
    named_results: {dataset_name: results_dict_from_model_inference}
    returns a long-form DataFrame with columns:
      ['dataset','task','kind','layer','step','val']
    Missing activations -> empty DataFrame.
    """
    rows = []
    for dataset_name, res in named_results.items():
        acts = res.get("activations")
        if not acts:
            continue
        for task in ("cls", "regress"):
            a = acts.get(task)
            if not a:
                continue
            # tensors -> cpu numpy
            val   = a["val"].detach().cpu().numpy()
            layer = a["layer"].detach().cpu().numpy()
            kind  = a["kind"].detach().cpu().numpy()
            step  = a["step"].detach().cpu().numpy()
            for v, l, k, s in zip(val, layer, kind, step):
                rows.append({
                    "dataset": dataset_name,
                    "task": task,
                    "kind": KIND_MAP.get(int(k), f"kind_{int(k)}"),
                    "layer": int(l),
                    "step":  int(s),
                    "val":   float(v),
                })
    import pandas as pd
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["dataset","task","kind","layer","step","val"]
    )

def _plot_activation_norms(axs, train_df, test_df, title_prefix=""):
    """
    axs: a dict like {'cls': ax1, 'regress': ax2} or a list [ax1, ax2]
    Plots mean(val) vs layer for each kind, with train/test styles.
    """
    import numpy as np

    def _agg(df):
        if df.empty:
            return df
        # If you logged step==-1 in inference, step adds no value; we just mean over step & dataset
        return (df.groupby(["task","kind","layer"])["val"]
                  .mean().reset_index())

    train_g = _agg(train_df)
    test_g  = _agg(test_df)

    tasks = ["cls","regress"]
    for i, task in enumerate(tasks):
        ax = axs[i] if isinstance(axs, (list, tuple)) else axs[task]
        ax.set_title(f"{title_prefix}{task} activation norms")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Mean L2 activation")
        # one line per kind; train solid, test dashed
        for kind in ["mapper","attn","pooled"]:
            td = train_g[(train_g.task==task) & (train_g.kind==kind)]
            sd = test_g[(test_g.task==task)  & (test_g.kind==kind)]
            if not td.empty:
                ax.plot(td["layer"], td["val"], label=f"train {kind}")
            if not sd.empty:
                ax.plot(sd["layer"], sd["val"], linestyle="--", label=f"test {kind}")
        ax.legend(loc="best")

class TrainingPlotter:
    def __init__(
        self,
        results_dir: str,
        heads: List[str],
        table_type: str,
        train_valid_data: Dict,
        test_data: Dict,
        output_args: str,
        device: str,
        plot_frequency: int,
        distributed: bool = False,
        swa_start: Optional[int] = None,
        loss_fn = None, 
        plot_interaction_e: bool = False,

    ):
        self.results_dir = results_dir
        self.heads = heads
        self.table_type = table_type
        self.train_valid_data = train_valid_data
        self.test_data = test_data
        self.output_args = output_args
        self.device = device
        self.plot_frequency = plot_frequency
        self.distributed = distributed
        self.swa_start = swa_start
        self.loss_fn = loss_fn
        self.plot_interaction_e = plot_interaction_e



    def plot(self, model_epoch: str, model: torch.nn.Module, rank: int) -> None:

        # All ranks process data through model_inference
        train_valid_dict = model_inference(
            self.train_valid_data,
            model,
            self.output_args,
            self.device,
            self.distributed,
            self.loss_fn,
        )
        test_dict = model_inference(
            self.test_data, model, self.output_args, self.device, self.distributed, self.loss_fn,
        )

                # ----- collect activations (DataFrames) -----
        train_act_df = _activations_to_df(train_valid_dict)
        test_act_df  = _activations_to_df(test_dict)
        have_acts = (not train_act_df.empty) or (not test_act_df.empty)

        # ----- (optional) persist full results for debugging -----
        # your JSON dump is fine; tensors are now converted to lists by earlier compute(), but if not,
        # make sure to convert to python types before dumping.

        if rank != 0:
            return

        data = pd.DataFrame(results for results in parse_training_results(self.results_dir))
        labels, quantities = error_type[self.table_type]

        for head in self.heads:
            fig = plt.figure(layout="constrained", figsize=(11, 9 if have_acts else 6))
            fig.suptitle(f"Model loaded from epoch {model_epoch} ({head} head)", fontsize=16)

            # If we have activations, make 3 subfig rows; else keep 2
            if have_acts:
                subfigs = fig.subfigures(3, 1, height_ratios=[1, 1, 1], hspace=0.06)
            else:
                subfigs = fig.subfigures(2, 1, height_ratios=[1, 1], hspace=0.06)

            # row 1: epoch dependence (unchanged)
            axsTop = subfigs[0].subplots(1, 2, sharey=False)
            plot_epoch_dependence(axsTop, data, head, model_epoch, labels)

            # row 2: inference scatter(s) (unchanged)
            axsBottom = subfigs[1].subplots(1, len(quantities), sharey=False, squeeze=False)
            axsBottom = axsBottom.ravel()
            plot_inference_from_results(
                axsBottom, train_valid_dict, test_dict, head, quantities,
                plot_interaction_e=self.plot_interaction_e
            )

            # row 3: NEW — activation norms (2 axes: cls & regress)
            if have_acts:
                ax_act = subfigs[2].subplots(1, 2, sharey=False)
                _plot_activation_norms(ax_act, train_act_df, test_act_df, title_prefix="")

            if self.swa_start is not None:
                for ax in axsTop:
                    ax.axvline(self.swa_start, color="black", linestyle="dashed", linewidth=1, alpha=0.6,
                            label="Stage Two Starts")
                stage = "stage_two" if self.swa_start < model_epoch else "stage_one"
            else:
                stage = "stage_one"

            axsTop[0].legend(loc="best")
            filename = f"{self.results_dir[:-4]}_{head}_{stage}.png"
            fig.savefig(filename, dpi=300, bbox_inches="tight")
            plt.close(fig)

        #MONITORING-----
        import json
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)
        with open(f'epoch_{model_epoch}_test.json','w+') as file:
            json.dump(test_dict, fp=file, cls=NumpyEncoder)
        with open(f'epoch_{model_epoch}_train.json','w+') as file:
            json.dump(train_valid_dict, fp=file, cls=NumpyEncoder)
        #MONITORING------

        # Only rank 0 creates and saves plots
        if rank != 0:
            return

        data = pd.DataFrame(
            results for results in parse_training_results(self.results_dir)
        )
        labels, quantities = error_type[self.table_type]

        logging.info("error table type")
        logging.info(self.table_type)

        logging.info("quantities:")
        logging.info(quantities)


        for head in self.heads:
            logging.info(f"heads in plotter: {head}")
            fig = plt.figure(layout="constrained", figsize=(10, 6))
            fig.suptitle(
                f"Model loaded from epoch {model_epoch} ({head} head)", fontsize=16
            )

            subfigs = fig.subfigures(2, 1, height_ratios=[1, 1], hspace=0.05)
            axsTop = subfigs[0].subplots(1, 2, sharey=False)
            axsBottom = subfigs[1].subplots(1, len(quantities), sharey=False, squeeze=False)
            axsBottom = axsBottom.ravel()  # now always a 1D array of Axes

            plot_epoch_dependence(axsTop, data, head, model_epoch, labels)

            # Use the pre-computed results for plotting
            plot_inference_from_results(
                axsBottom, train_valid_dict, test_dict, head, quantities, plot_interaction_e=self.plot_interaction_e
            )

            if self.swa_start is not None:
                # Add vertical lines to both axes
                for ax in axsTop:
                    ax.axvline(
                        self.swa_start,
                        color="black",
                        linestyle="dashed",
                        linewidth=1,
                        alpha=0.6,
                        label="Stage Two Starts",
                    )
                stage = "stage_two" if self.swa_start < model_epoch else "stage_one"
            else:
                stage = "stage_one"
            axsTop[0].legend(loc="best")
            # Save the figure using the appropriate stage in the filename
            filename = f"{self.results_dir[:-4]}_{head}_{stage}.png"

            fig.savefig(filename, dpi=300, bbox_inches="tight")
            plt.close(fig)


def parse_training_results(path: str) -> List[dict]:
    results = []
    with open(path, mode="r", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line.strip())  # Ensure it's valid JSON
                results.append(d)
            except json.JSONDecodeError:
                print(
                    f"Skipping invalid line: {line.strip()}"
                )  # Handle non-JSON lines gracefully
    return results


def plot_epoch_dependence(
    axes: np.ndarray, data: pd.DataFrame, head: str, model_epoch: str, labels: List[str]
) -> None:

    valid_data = (
        data[data["mode"] == "eval"]
        .groupby(["mode", "epoch", "head"])
        .agg(["mean", "std"])
        .reset_index()
    )
    valid_data = valid_data[valid_data["head"] == head]
    train_data = (
        data[data["mode"] == "opt"]
        .groupby(["mode", "epoch"])
        .agg(["mean", "std"])
        .reset_index()
    )

    # ---- Plot loss ----
    ax = axes[0]
    ax.plot(
        train_data["epoch"], train_data["loss"]["mean"], color=colors[1], linewidth=1
    )
    ax.set_ylabel("Training Loss", color=colors[1])
    ax.set_yscale("log")

    ax2 = ax.twinx()
    ax2.plot(
        valid_data["epoch"], valid_data["loss"]["mean"], color=colors[0], linewidth=1
    )
    ax2.set_ylabel("Validation Loss", color=colors[0])
    ax2.set_yscale("log")

    ax.axvline(
        model_epoch,
        color="black",
        linestyle="solid",
        linewidth=1,
        alpha=0.8,
        label="Loaded Model",
    )
    ax.set_xlabel("Epoch")
    ax.grid(True, linestyle="--", alpha=0.5)

    # ---- Plot selected keys ----
    ax = axes[1]
    twin_axes = []
    for i, label in enumerate(labels):
        color = colors[(i + 3)]
        key, axis_label = label
        if i == 0:
            main_ax = ax
        else:
            main_ax = ax.twinx()
            main_ax.spines.right.set_position(("outward", 60 * (i - 1)))
            twin_axes.append(main_ax)

        main_ax.plot(
            valid_data["epoch"],
            valid_data[key]["mean"] * 1e3
              if not axis_label == "Accuracy of Classifier [%]" else valid_data[key]["mean"] * 1e2,
            color=color,
            label=label,
            linewidth=1,
        )
        if axis_label == "Accuracy of Classifier [%]":
            main_ax.set_ylim(0, 100)  # Set y-axis limits for accuracy
        else:
            main_ax.set_yscale("log")
        main_ax.set_ylabel(axis_label, color=color)
        main_ax.tick_params(axis="y", colors=color)
    ax.axvline(
        model_epoch,
        color="black",
        linestyle="solid",
        linewidth=1,
        alpha=0.8,
        label="Loaded Model",
    )
    ax.set_xlabel("Epoch")
    ax.grid(True, linestyle="--", alpha=0.5)


# INFERENCE=========
def plot_inference_from_results(
    axes: np.ndarray,
    train_valid_dict: dict,
    test_dict: dict,
    head: str,
    quantities: List[str],
    plot_interaction_e: bool = False,
) -> None:

    logging.info(f"axis: {axes}")
    for ax, quantity in zip(axes, quantities):
        key, label = quantity

        # Store legend handles to avoid duplicates
        legend_labels = {}

        # Plot train/valid data (each entry keeps its own name)
        for name, result in train_valid_dict.items():
            if "train" in name:
                fixed_color_train_valid = colors[1]
                marker = "x"
            else:
                fixed_color_train_valid = colors[0]
                marker = "+"
            if head not in name:
                continue

            # Initialize scatter to None
            scatter = None
            logging.info("result keys:")
            logging.info(result.keys())

            if key == "energy" and "energy" in result:
                scatter = ax.scatter(
                    result["energy"]["reference_per_atom"],
                    result["energy"]["predicted_per_atom"],
                    marker=marker,
                    color=fixed_color_train_valid,
                    label=name,
                )

            elif key == "force" and "forces" in result:
                scatter = ax.scatter(
                    result["forces"]["reference"],
                    result["forces"]["predicted"],
                    marker=marker,
                    color=fixed_color_train_valid,
                    label=name,
                )

            elif key == "stress" and "stress" in result:
                scatter = ax.scatter(
                    result["stress"]["reference"],
                    result["stress"]["predicted"],
                    marker=marker,
                    color=fixed_color_train_valid,
                    label=name,
                )

            elif key == "virials" and "virials" in result:
                scatter = ax.scatter(
                    result["virials"]["reference_per_atom"],
                    result["virials"]["predicted_per_atom"],
                    marker=marker,
                    color=fixed_color_train_valid,
                    label=name,
                )

            elif key == "dipole" and "dipole" in result:
                scatter = ax.scatter(
                    result["dipole"]["reference_per_atom"],
                    result["dipole"]["predicted_per_atom"],
                    marker=marker,
                    color=fixed_color_train_valid,
                    label=name,
                )

            elif key == "coupling_class" and "coupling_class" in result:
                y_true = result["coupling_class"]["reference"]
                y_pred = result["coupling_class"]["predicted"]
                h = ax.hist2d(y_pred, y_true, bins=[3000, 2], cmap="Blues")  # 50 bins over prob, 2 over label
                plt.colorbar(h[3], ax=ax, label="count")
                ax.set_xlabel("Predicted probability")
                ax.set_ylabel("Reference label")
                ax.set_yticks([0, 1])

            
            elif key == "effective_coupling" and "effective_coupling" in result:
                # if plot_interaction_e:
                #     scatter = ax.scatter(
                #        y=result["interaction_energies"]["reference"],

                # else:
                    scatter = ax.scatter(
                        result["effective_coupling"]["reference"],
                        result["effective_coupling"]["predicted"],
                        marker=marker,
                        color=fixed_color_train_valid,
                        label=name,  
                    )

            # Add each train/valid dataset's name to the legend if scatter was assigned
            if scatter is not None:
                legend_labels[name] = scatter

        fixed_color_test = colors[2]  # Color for test dataset

        # Plot test data (single legend entry)
        for name, result in test_dict.items():
            # Initialize scatter to None to avoid possibly used before assignment
            scatter = None

            if key == "energy" and "energy" in result:
                scatter = ax.scatter(
                    result["energy"]["reference_per_atom"],
                    result["energy"]["predicted_per_atom"],
                    marker="o",
                    color=fixed_color_test,
                    label="Test",
                )

            elif key == "force" and "forces" in result:
                scatter = ax.scatter(
                    result["forces"]["reference"],
                    result["forces"]["predicted"],
                    marker="o",
                    color=fixed_color_test,
                    label="Test",
                )

            elif key == "stress" and "stress" in result:
                scatter = ax.scatter(
                    result["stress"]["reference"],
                    result["stress"]["predicted"],
                    marker="o",
                    color=fixed_color_test,
                    label="Test",
                )

            elif key == "virials" and "virials" in result:
                scatter = ax.scatter(
                    result["virials"]["reference_per_atom"],
                    result["virials"]["predicted_per_atom"],
                    marker="o",
                    color=fixed_color_test,
                    label="Test",
                )

            elif key == "dipole" and "dipole" in result:
                scatter = ax.scatter(
                    result["dipole"]["reference_per_atom"],
                    result["dipole"]["predicted_per_atom"],
                    marker="o",
                    color=fixed_color_test,
                    label="Test",
                )

            elif key == "effective_coupling" and "effective_coupling" in result:
                scatter = ax.scatter(
                    result["effective_coupling"]["reference"],
                    result["effective_coupling"]["predicted"],
                    marker=marker,
                    color=fixed_color_train_valid,
                    label=name,  
                )

            # Only add to legend_labels if scatter was assigned
            if scatter is not None:
                legend_labels["Test"] = scatter

        # Add diagonal line for guide
        min_val = min(ax.get_xlim()[0], ax.get_ylim()[0])
        max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            linestyle="--",
            color="black",
            alpha=0.7,
        )

        # Set legend with unique entries (Test + individual train/valid names)
        if legend_labels:
            ax.legend(
                handles=legend_labels.values(), labels=legend_labels.keys(), loc="best"
            )
        ax.set_xlabel(f"Reference {label}")
        ax.set_ylabel(f"MACE {label}")
        ax.grid(True, linestyle="--", alpha=0.5)


def model_inference(
    all_data_loaders: dict,
    model: torch.nn.Module,
    output_args: Dict[str, bool],
    device: str,
    distributed: bool = False,
    loss_fn = None,
):

    for param in model.parameters():
        param.requires_grad = False

    results_dict = {}

    for name in all_data_loaders:
        data_loader = all_data_loaders[name]
        logging.debug(f"Running inference on {name} dataset")
        scatter_metric = InferenceMetric(loss_fn).to(device)

        for batch in data_loader:
            batch = batch.to(device)
            batch_dict = batch.to_dict()
            output = model(
                batch_dict,
                training=False,
                compute_force=output_args.get("forces", False),
                compute_virials=output_args.get("virials", False),
                compute_stress=output_args.get("stress", False),
            )

            results = scatter_metric(batch, output) #this calls update()

        if distributed:
            torch.distributed.barrier()

        results = scatter_metric.compute()
        results_dict[name] = results
        scatter_metric.reset()

        del data_loader

    for param in model.parameters():
        param.requires_grad = True

    return results_dict


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.cpu().detach().numpy()

class InferenceMetric(Metric):
    """Metric class for collecting reference and predicted values for scatterplot visualization."""

    def __init__(self, loss_fn):
        super().__init__()
        # Raw values
        self.add_state("ref_energies", default=[], dist_reduce_fx="cat")
        self.add_state("pred_energies", default=[], dist_reduce_fx="cat")
        self.add_state("ref_forces", default=[], dist_reduce_fx="cat")
        self.add_state("pred_forces", default=[], dist_reduce_fx="cat")
        self.add_state("ref_stress", default=[], dist_reduce_fx="cat")
        self.add_state("pred_stress", default=[], dist_reduce_fx="cat")
        self.add_state("ref_virials", default=[], dist_reduce_fx="cat")
        self.add_state("pred_virials", default=[], dist_reduce_fx="cat")
        self.add_state("ref_dipole", default=[], dist_reduce_fx="cat")
        self.add_state("pred_dipole", default=[], dist_reduce_fx="cat")
        self.add_state("ref_coupling_class", default=[], dist_reduce_fx="cat")
        self.add_state("pred_coupling_class", default=[], dist_reduce_fx="cat")
        self.add_state("ref_effective_coupling", default=[], dist_reduce_fx="cat")
        self.add_state("pred_effective_coupling", default=[], dist_reduce_fx="cat")

        # Per-atom normalized values
        self.add_state("ref_energies_per_atom", default=[], dist_reduce_fx="cat")
        self.add_state("pred_energies_per_atom", default=[], dist_reduce_fx="cat")
        self.add_state("ref_virials_per_atom", default=[], dist_reduce_fx="cat")
        self.add_state("pred_virials_per_atom", default=[], dist_reduce_fx="cat")
        self.add_state("ref_dipole_per_atom", default=[], dist_reduce_fx="cat")
        self.add_state("pred_dipole_per_atom", default=[], dist_reduce_fx="cat")

        # Store atom counts for each configuration
        self.add_state("atom_counts", default=[], dist_reduce_fx="cat")

        # Counters
        self.add_state("n_energy", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_forces", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_stress", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_virials", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_dipole", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_classifiers", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_couplings", default=torch.tensor(0.0), dist_reduce_fx="sum")

        # Electronic coupling metrics
        self.add_state("coupling_accuracy", default = [], dist_reduce_fx="cat")

        #monitor node light up
        self.add_state("n_iter", default = [], dist_reduce_fx="cat")
        self.add_state("act_val_cls",   default=torch.tensor([], dtype=torch.float32), dist_reduce_fx="cat")
        self.add_state("act_layer_cls", default=torch.tensor([], dtype=torch.int32),   dist_reduce_fx="cat")
        self.add_state("act_kind_cls",  default=torch.tensor([], dtype=torch.int8),    dist_reduce_fx="cat")
        self.add_state("act_step_cls",  default=torch.tensor([], dtype=torch.int32),   dist_reduce_fx="cat")

        self.add_state("act_val_regress",   default=torch.tensor([], dtype=torch.float32), dist_reduce_fx="cat")
        self.add_state("act_layer_regress", default=torch.tensor([], dtype=torch.int32),   dist_reduce_fx="cat")
        self.add_state("act_kind_regress",  default=torch.tensor([], dtype=torch.int8),    dist_reduce_fx="cat")
        self.add_state("act_step_regress",  default=torch.tensor([], dtype=torch.int32),   dist_reduce_fx="cat")

        #Pass through the loss function
        self.loss_fn = loss_fn

        # self.add_state("coupling predicted", default = [], dist_reduce_fx="cat")

    @staticmethod
    def _stack_layer_means(per_layer_list):
        # per_layer_list: list of tensors shaped [B] (per-batch norms per layer)
        # returns [L] means across batch
        return torch.stack([v.detach().float().mean() for v in per_layer_list]) if per_layer_list else torch.tensor([])

    def _accum_long_form(self, *, target: str, mapper_L, attn_L, pool_L, step_idx: int):
        """
        target: 'cls' or 'regress'
        mapper_L/attn_L/pool_L: [L] or empty tensor
        Writes into act_*_{target} states.
        """
        chunks = []

        if mapper_L is not None and mapper_L.numel() > 0:
            L = mapper_L.numel()
            chunks.append((
                mapper_L.reshape(-1),
                torch.arange(L, dtype=torch.int32, device=mapper_L.device),
                torch.full((L,), 0, dtype=torch.int8, device=mapper_L.device),  # kind 0 = mapper
            ))
        if attn_L is not None and attn_L.numel() > 0:
            L = attn_L.numel()
            chunks.append((
                attn_L.reshape(-1),
                torch.arange(L, dtype=torch.int32, device=attn_L.device),
                torch.full((L,), 1, dtype=torch.int8, device=attn_L.device),    # kind 1 = attn
            ))
        if pool_L is not None and pool_L.numel() > 0:
            L = pool_L.numel()
            chunks.append((
                pool_L.reshape(-1),
                torch.arange(L, dtype=torch.int32, device=pool_L.device),
                torch.full((L,), 2, dtype=torch.int8, device=pool_L.device),    # kind 2 = pooled
            ))

        if not chunks:
            return

        val   = torch.cat([c[0] for c in chunks], dim=0).cpu()
        layer = torch.cat([c[1] for c in chunks], dim[0]).cpu()
        kind  = torch.cat([c[2] for c in chunks], dim=0).cpu()
        step  = torch.full_like(layer, -1 if step_idx is None else int(step_idx), dtype=torch.int32)

        if target == "cls":
            self.act_val_cls   = torch.cat([self.act_val_cls,   val])
            self.act_layer_cls = torch.cat([self.act_layer_cls, layer])
            self.act_kind_cls  = torch.cat([self.act_kind_cls,  kind])
            self.act_step_cls  = torch.cat([self.act_step_cls,  step])
        else:
            self.act_val_regress   = torch.cat([self.act_val_regress,   val])
            self.act_layer_regress = torch.cat([self.act_layer_regress, layer])
            self.act_kind_regress  = torch.cat([self.act_kind_regress,  kind])
            self.act_step_regress  = torch.cat([self.act_step_regress,  step])

    def update(
        self,
        batch, 
        output,
        mapper_L: torch.Tensor = None,   # [L]
        attn_L:   torch.Tensor = None,   # [L]
        pool_L:   torch.Tensor = None,   # [L]
        step_idx: int = None,        
        ):  # pylint: disable=arguments-differ
        """Update metric states with new batch data."""
        # Calculate number of atoms per configuration
        atoms_per_config = batch.ptr[1:] - batch.ptr[:-1]
        self.atom_counts.append(atoms_per_config)
        mapper_L_cls   = self._stack_layer_means(output.get("mom_mapper_norm_cls", []))
        attn_L_cls     = self._stack_layer_means(output.get("attn_norm_cls", []))
        pool_L_cls     = self._stack_layer_means(output.get("fc_norm_cls", []))

        mapper_L_reg   = self._stack_layer_means(output.get("mom_mapper_norm_regress", []))
        attn_L_reg     = self._stack_layer_means(output.get("attn_norm_regress", []))
        pool_L_reg     = self._stack_layer_means(output.get("fc_norm_regress", []))

        with torch.no_grad():
            self._accum_long_form(target="cls",
                                mapper_L=mapper_L_cls, attn_L=attn_L_cls, pool_L=pool_L_cls,
                                step_idx=step_idx)
            self._accum_long_form(target="regress",
                                mapper_L=mapper_L_reg, attn_L=attn_L_reg, pool_L=pool_L_reg,
                                step_idx=step_idx)

        # Energy
        if output.get("energy") is not None and batch.energy is not None:
            self.n_energy += 1.0
            self.ref_energies.append(batch.energy)
            self.pred_energies.append(output["energy"])
            # Per-atom normalization
            self.ref_energies_per_atom.append(batch.energy / atoms_per_config)
            self.pred_energies_per_atom.append(output["energy"] / atoms_per_config)

        # Forces
        if output.get("forces") is not None and batch.forces is not None:
            self.n_forces += 1.0
            self.ref_forces.append(batch.forces)
            self.pred_forces.append(output["forces"])

        # Stress
        if output.get("stress") is not None and batch.stress is not None:
            self.n_stress += 1.0
            self.ref_stress.append(batch.stress)
            self.pred_stress.append(output["stress"])

        # Virials
        if output.get("virials") is not None and batch.virials is not None:
            self.n_virials += 1.0
            self.ref_virials.append(batch.virials)
            self.pred_virials.append(output["virials"])
            # Per-atom normalization
            atoms_per_config_3d = atoms_per_config.view(-1, 1, 1)
            self.ref_virials_per_atom.append(batch.virials / atoms_per_config_3d)
            self.pred_virials_per_atom.append(output["virials"] / atoms_per_config_3d)

        # Dipole
        if output.get("dipole") is not None and batch.dipole is not None:
            self.n_dipole += 1.0
            self.ref_dipole.append(batch.dipole)
            self.pred_dipole.append(output["dipole"])
            atoms_per_config_3d = atoms_per_config.view(-1, 1)
            self.ref_dipole_per_atom.append(batch.dipole / atoms_per_config_3d)
            self.pred_dipole_per_atom.append(output["dipole"] / atoms_per_config_3d)

        # Coupling class
        if output.get("coupling_class") is not None and batch.coupling_class is not None:
            self.n_classifiers += 1.0
            self.ref_coupling_class.append(batch.coupling_class)
            self.pred_coupling_class.append(output["coupling_class"])

        if (
            output.get("effective_coupling") is not None 
            and batch.effective_coupling is not None 
            and output.get("coupling_class") is not None
        ):
            self.n_couplings += 1.0
            ref = batch.effective_coupling

            # same linearization everywhere
            y_pred_linear = self.loss_fn.to_linear_space(output["effective_coupling"]).squeeze(-1)

            logits = output["coupling_class"]
            probs  = torch.sigmoid(logits).squeeze(-1)             # [B]
            preds  = (probs > 0.5).to(y_pred_linear.dtype)         # [B]
            y_pred_linear = y_pred_linear * preds                  # [B]

            logging.info("ref (linear): %s", ref)
            logging.info("pred (linear, gated): %s", y_pred_linear)

            self.ref_effective_coupling.append(ref.reshape_as(y_pred_linear))
            self.pred_effective_coupling.append(y_pred_linear)

        if (
            output.get("effective_coupling") is not None 
            and batch.effective_coupling is not None 
        ):
            self.n_couplings += 1.0
            y_pred_linear = self.loss_fn.to_linear_space(output["effective_coupling"]).squeeze(-1)  # [B]
            ref = batch.effective_coupling.to(y_pred_linear.device, y_pred_linear.dtype).reshape_as(y_pred_linear)
            self.ref_effective_coupling.append(ref.reshape_as(y_pred_linear))
            self.pred_effective_coupling.append(y_pred_linear)

            logging.info("ref (linear): %s", ref)
            logging.info("pred (linear, gated): %s", y_pred_linear)


    def _process_data(self, ref_list, pred_list):
        # Handle different possible states of ref_list and pred_list in distributed mode

        # Check if this is a list type object
        if isinstance(ref_list, (list, tuple)):
            if len(ref_list) == 0:
                return None, None
            ref = torch.cat(ref_list).reshape(-1)
            pred = torch.cat(pred_list).reshape(-1)
        # Handle case where ref_list is already a tensor (happens after reset in distributed mode)
        elif isinstance(ref_list, torch.Tensor):
            ref = ref_list.reshape(-1)
            pred = pred_list.reshape(-1)
        # Handle other possible types
        else:
            return None, None
        return to_numpy(ref), to_numpy(pred)

    def compute(self):
        """Compute final results for scatterplot."""
        results = {}

        results["activations"] = {
            "cls": {
                "val":   self.act_val_cls,    # [N]
                "layer": self.act_layer_cls,  # [N]
                "kind":  self.act_kind_cls,   # [N] (0 mapper, 1 attn, 2 pooled)
                "step":  self.act_step_cls,   # [N] (-1 if unknown)
            },
            "regress": {
                "val":   self.act_val_regress,
                "layer": self.act_layer_regress,
                "kind":  self.act_kind_regress,
                "step":  self.act_step_regress,
            }
        }

        # Process energies
        if self.n_energy:
            ref_e, pred_e = self._process_data(self.ref_energies, self.pred_energies)
            ref_e_pa, pred_e_pa = self._process_data(
                self.ref_energies_per_atom, self.pred_energies_per_atom
            )
            results["energy"] = {
                "reference": ref_e,
                "predicted": pred_e,
                "reference_per_atom": ref_e_pa,
                "predicted_per_atom": pred_e_pa,
            }

        # Process forces
        if self.n_forces:
            ref_f, pred_f = self._process_data(self.ref_forces, self.pred_forces)
            results["forces"] = {
                "reference": ref_f,
                "predicted": pred_f,
            }

        # Process stress
        if self.n_stress:
            ref_s, pred_s = self._process_data(self.ref_stress, self.pred_stress)
            results["stress"] = {
                "reference": ref_s,
                "predicted": pred_s,
            }

        # Process virials
        if self.n_virials:
            ref_v, pred_v = self._process_data(self.ref_virials, self.pred_virials)
            ref_v_pa, pred_v_pa = self._process_data(
                self.ref_virials_per_atom, self.pred_virials_per_atom
            )
            results["virials"] = {
                "reference": ref_v,
                "predicted": pred_v,
                "reference_per_atom": ref_v_pa,
                "predicted_per_atom": pred_v_pa,
            }

        # Process dipoles
        if self.n_dipole:
            ref_d, pred_d = self._process_data(self.ref_dipole, self.pred_dipole)
            ref_d_pa, pred_d_pa = self._process_data(
                self.ref_dipole_per_atom, self.pred_dipole_per_atom
            )
            results["dipole"] = {
                "reference": ref_d,
                "predicted": pred_d,
                "reference_per_atom": ref_d_pa,
                "predicted_per_atom": pred_d_pa,
            }

        # Process classifier
        if self.n_classifiers:
            ref_c, pred_c = self._process_data(self.ref_coupling_class, self.pred_coupling_class)

            results["coupling_class"] = {
                "reference": ref_c,
                "predicted": pred_c,
            }
        # Process n_couplings
        if self.n_couplings:
            ref_ec, pred_ec = self._process_data(self.ref_effective_coupling, self.pred_effective_coupling)

            results["effective_coupling"] = {
                "reference": ref_ec,
                "predicted": pred_ec
            }

            logging.info(f"results of effective coupling : {results['effective_coupling']}")

        return results
