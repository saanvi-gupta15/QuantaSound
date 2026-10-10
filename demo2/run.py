"""python -m demo2.run: save four scenarios, figures and a Qiskit benchmark."""
import json
from pathlib import Path
from importlib.metadata import version
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .simulation import load_sources, simulate, export_result
from .presentation import plot_result
from demo1.quantum import benchmark

RESULTS = Path(__file__).parent / "results"


def main():
    sources, provenance = load_sources()
    report = {"source_provenance":provenance, "versions":{name:version(name) for name in
              ["numpy","scipy","librosa","scikit-learn","qiskit","ipywidgets"]}, "scenarios":{}}
    for name, selected in [("preserve_all", []), ("reduce_engine", ["engine"]),
                           ("reduce_keyboard", ["keyboard"]), ("reduce_both", ["engine","keyboard"])]:
        print(f"Running {name}", flush=True)
        result = simulate(sources, selected)
        folder = RESULTS/name
        export_result(result, folder)
        figure = plot_result(result)
        figure.savefig(folder/"comparison.png", dpi=120)
        plt.close(figure)
        report["scenarios"][name] = result.metrics
    print("Running separate synthetic audible-hum listening example", flush=True)
    report["keyboard_model_comparison"] = {}
    for model in ("calibrated_lowpass", "calibrated_broadband"):
        result = simulate(sources, ["keyboard"], controller=model)
        folder = RESULTS/model
        export_result(result, folder)
        figure = plot_result(result)
        figure.savefig(folder/"comparison.png", dpi=120)
        plt.close(figure)
        report["keyboard_model_comparison"][model] = result.metrics
    audible, provenance = load_sources(scene="audible_hum")
    result = simulate(audible, ["engine"])
    result.metrics["scene"] = "audible_hum"
    folder = RESULTS/"audible_hum"
    export_result(result, folder)
    figure = plot_result(result)
    figure.suptitle("Synthetic 200 Hz hum listening example — not the recorded-engine benchmark")
    figure.savefig(folder/"comparison.png", dpi=120)
    plt.close(figure)
    report["audible_hum_example"] = {"source_provenance":provenance, "metrics":result.metrics}
    print("Running offline Qiskit QAOA benchmark", flush=True)
    _, _, _, report["qiskit_qaoa"] = benchmark(backend="qiskit")
    (RESULTS/"summary.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    for name, m in report["scenarios"].items():
        print(name, {key:m[key] for key in ["total_noise_reduction_db", "selected_residual_reduction_db",
                                          "speech_counterfactual_distortion_rms_percent"]})


if __name__ == "__main__":
    main()
