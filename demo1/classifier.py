"""PRELIMINARY five-class MFCC baseline with official fold separation."""
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from .data import DATA, CLASSES, prepare_data
from .plots import save


def run_classifier():
    rows = prepare_data(classifier=True)
    names = np.array([r["filename"] for r in rows])
    cache = DATA / "features.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as saved:
            valid = np.array_equal(saved["names"], names)
            features = saved["features"] if valid else None
    else:
        features = None
    if features is None:
        features = []
        for i, row in enumerate(rows):
            x, sr = librosa.load(DATA/"audio"/row["filename"], sr=16000)
            mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=20, n_fft=1024, hop_length=256)
            features.append(np.r_[mfcc.mean(axis=1), mfcc.std(axis=1)])
            if (i+1) % 40 == 0:
                print(f"MFCC features: {i+1}/{len(rows)}", flush=True)
        features = np.array(features)
        np.savez(cache, names=names, features=features)
    labels = np.array([r["category"] for r in rows])
    test = np.array([r["fold"] == "5" for r in rows])
    train_sources = {r["src_file"] for r, held in zip(rows, test) if not held}
    test_sources = {r["src_file"] for r, held in zip(rows, test) if held}
    if train_sources & test_sources:
        raise ValueError("Recording sources overlap between training and test")
    model = RandomForestClassifier(n_estimators=200, random_state=7, n_jobs=1)
    model.fit(features[~test], labels[~test])
    prediction = model.predict(features[test])
    matrix = confusion_matrix(labels[test], prediction, labels=list(CLASSES))
    fig, ax = plt.subplots(figsize=(8, 7), layout="constrained")
    ConfusionMatrixDisplay(matrix, display_labels=CLASSES).plot(ax=ax, xticks_rotation=30, colorbar=False)
    ax.set_title("PRELIMINARY · MFCC + random forest · official fold 5 held out")
    save(fig, "preliminary_classifier.png")
    metrics = {"status":"PRELIMINARY single-label isolated-clip classification",
               "classes":list(CLASSES), "train_count":int((~test).sum()), "test_count":int(test.sum()),
               "accuracy":float(accuracy_score(labels[test], prediction)),
               "balanced_accuracy":float(balanced_accuracy_score(labels[test], prediction)),
               "chance_accuracy":0.2, "confusion_matrix":matrix.tolist(),
               "training_folds":[1,2,3,4], "test_fold":5, "source_overlap":0,
               "feature":"20 MFCC means and standard deviations, 16 kHz mono",
               "seed":7, "trees":200, "test_used_for_tuning":False}
    return fig, metrics
