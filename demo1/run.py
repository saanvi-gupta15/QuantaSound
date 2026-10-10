"""Run from the repository root: python -m demo1.run --all"""
import argparse
import json
import platform
from importlib.metadata import version
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .plots import demo_a, demo_b, demo_c, RESULTS, save
from .quantum import benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="A/B/C + classifier + QAOA; first run downloads audio")
    parser.add_argument("--audio", action="store_true", help="Include Demo A real audio downloads")
    parser.add_argument("--classifier", action="store_true", help="Download 200 ESC-50 clips and evaluate baseline")
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    report = {"scope":"Review 1 simulation; no live hardware ANC or learned user personalization",
              "python":platform.python_version(),
              "versions":{name:version(name) for name in ["numpy","scipy","matplotlib","librosa","soundfile","scikit-learn"]}}
    if args.audio or args.all:
        print("Demo A: real recordings", flush=True)
        _, report["demo_a"] = demo_a(); plt.close("all")
    print("Demo B: causal digital filtering", flush=True)
    _, report["demo_b"] = demo_b(); plt.close("all")
    for kind in ["tonal", "broadband"]:
        print(f"Demo C: FxLMS {kind}", flush=True)
        _, report[f"demo_c_{kind}"] = demo_c(kind); plt.close("all")
    states, costs, probability, report["quantum_teaser"] = benchmark()
    fig, ax = plt.subplots(figsize=(10,4), layout="constrained")
    ax.bar(range(16), probability)
    ax.set_xticks(range(16), ["".join(str(int(b)) for b in s) for s in states], rotation=45)
    ax.set(xlabel="Block bits: engine, typing, speech, siren", ylabel="QAOA probability",
           title="PRELIMINARY · p=1 exact statevector simulation · no quantum advantage claim")
    save(fig, "preliminary_qaoa.png"); plt.close(fig)
    if args.classifier or args.all:
        from .classifier import run_classifier
        _, report["classifier"] = run_classifier(); plt.close("all")
    output = RESULTS / ("metrics.json" if args.all else "metrics_core.json")
    output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
