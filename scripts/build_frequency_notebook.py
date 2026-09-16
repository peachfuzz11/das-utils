"""Build notebooks/ship_frequency_analysis.ipynb from code, then execute it.

Kept as a generator script (not hand-authored notebook JSON) so the analysis
is reviewable as plain Python and reproducible - re-run this to regenerate
the notebook with fresh data.
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


md("""\
# Do ship crossings have a distinctive frequency signature in this DAS data?

Following Pedersen et al., *"A Feasibility Study of Automated Detection and
Classification of Signals in Distributed Acoustic Sensing"* (Sensors 2025,
25(17), 5445): characterize each signal by its power spectral density (PSD)
and dominant frequency, then check whether that frequency-domain
representation alone separates ships from background noise (PCA for
separability, clustering metrics for how well-defined that separation is).

**Data**: `annotations_fixed_offset.json` - the curated, per-file
"distance to nearest AIS ship" ground truth built earlier this session,
using the validated fixed cable offset (dlat_m=136.7, dlon_m=1232.9,
derived from a multi-band power sweep across every transiting crossing on
2025-11-22). "Ship" samples are the channel nearest the ship in every file
where a vessel came within 300m; "noise" samples are random channels from
files with no AIS vessel within 2000m of the whole cable.
Extraction: `scripts/extract_ship_psd_features.py` &rarr;
`/tmp/das_eval_cache/ship_psd_features.csv`.
""")

code("""\
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from das_utils import annotate, psd as psd_mod

df = pd.read_csv("/tmp/das_eval_cache/ship_psd_features.csv")
band_cols = [c for c in df.columns if c.startswith("band_")]
print(df["label"].value_counts())
df.head()
""")

md("""\
## 1. Dominant frequency: ships vs. noise

The paper reports vessels/vehicles dominant around ~30 Hz, with earthquakes
and structural signals at lower frequency. Does that hold here?
""")

code("""\
fig, ax = plt.subplots(figsize=(8, 4))
bins = np.linspace(0, 200, 41)
for label, color in [("ship", "tab:red"), ("noise", "tab:gray")]:
    sub = df[df["label"] == label]["dominant_freq_hz"]
    ax.hist(sub, bins=bins, alpha=0.6, label=f"{label} (n={len(sub)})", color=color, density=True)
ax.axvline(30, color="k", linestyle="--", linewidth=1, label="paper's reported ship peak (~30Hz)")
ax.set_xlabel("dominant frequency (Hz)")
ax.set_ylabel("density")
ax.set_title("Dominant PSD frequency: ship vs. noise samples")
ax.legend()
plt.tight_layout()
plt.show()

df.groupby("label")["dominant_freq_hz"].describe()
""")

md("""\
## 2. Averaged PSD shape

Band-power features summarize the spectrum into coarse bins; to see the
actual continuous shape, recompute and average the full Welch PSD (not just
the stored per-band summary) for every sample.
""")

code("""\
FS = 400.0

def full_psd(row):
    data = np.load(row["file"], mmap_mode="r")
    sig = np.asarray(data[:, int(row["channel"])], dtype=np.float64)
    freqs, p = psd_mod.channel_psd(sig, FS, nperseg=2048)
    return freqs, p

freqs_ref = None
curves = {"ship": [], "noise": []}
for _, row in df.iterrows():
    freqs, p = full_psd(row)
    freqs_ref = freqs
    curves[row["label"]].append(p)

fig, ax = plt.subplots(figsize=(8, 5))
for label, color in [("ship", "tab:red"), ("noise", "tab:gray")]:
    arr = np.array(curves[label])
    med = np.median(arr, axis=0)
    lo, hi = np.percentile(arr, [25, 75], axis=0)
    ax.plot(freqs_ref, med, color=color, label=f"{label} median")
    ax.fill_between(freqs_ref, lo, hi, color=color, alpha=0.2, label=f"{label} IQR")
ax.set_yscale("log")
ax.set_xlabel("frequency (Hz)")
ax.set_ylabel("PSD (log scale)")
ax.set_xlim(0, 200)
ax.set_title("Median PSD with interquartile band: ship vs. noise")
ax.legend()
plt.tight_layout()
plt.show()
""")

md("""\
## 3. Band-power profile

Mean fraction of total power in each frequency band (the same feature
vector fed to PCA/clustering below).
""")

code("""\
band_labels = [c.replace("band_", "").replace("hz", " Hz").replace("_", "-") for c in band_cols]
means = df.groupby("label")[band_cols].mean()

fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(len(band_cols))
width = 0.35
for i, (label, color) in enumerate([("ship", "tab:red"), ("noise", "tab:gray")]):
    ax.bar(x + i * width, means.loc[label], width=width, label=label, color=color, alpha=0.8)
ax.set_xticks(x + width / 2)
ax.set_xticklabels(band_labels, rotation=45, ha="right")
ax.set_ylabel("mean fraction of total power")
ax.set_title("Band-power profile: ship vs. noise")
ax.legend()
plt.tight_layout()
plt.show()
""")

md("""\
## 4. PCA separability

Project the band-power feature vectors (the same representation the paper
uses) into 2D and see whether ship/noise separate without using the label -
mirroring the paper's PCA-for-separability step.
""")

code("""\
X = StandardScaler().fit_transform(df[band_cols].to_numpy())
pca = PCA(n_components=2)
coords = pca.fit_transform(X)
df["pc1"], df["pc2"] = coords[:, 0], coords[:, 1]

fig, ax = plt.subplots(figsize=(7, 6))
for label, color in [("ship", "tab:red"), ("noise", "tab:gray")]:
    sub = df[df["label"] == label]
    ax.scatter(sub["pc1"], sub["pc2"], color=color, label=label, alpha=0.7, s=40)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
ax.set_title("PCA of band-power features, colored by true label")
ax.legend()
plt.tight_layout()
plt.show()
""")

md("""\
## 5. Clustering metrics

The paper reports Davies-Bouldin, Silhouette, and Calinski-Harabasz scores
for its HDBSCAN clustering. Compute the same three, both for an
unsupervised clustering (KMeans, k=2, and HDBSCAN) and directly against the
*true* ship/noise label - the latter is the ceiling: how separable the
labels would look if clustering perfectly recovered them.
""")

code("""\
y_true = (df["label"] == "ship").astype(int).to_numpy()

results = []
for name, labels in [
    ("true label (ceiling)", y_true),
    ("kmeans_k2", KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(X)),
    ("hdbscan", HDBSCAN(min_cluster_size=5).fit_predict(X)),
]:
    valid = labels != -1  # HDBSCAN noise points excluded from the metrics, as in the paper
    if len(set(labels[valid])) < 2:
        results.append({"method": name, "silhouette": float("nan"),
                         "davies_bouldin": float("nan"), "calinski_harabasz": float("nan"),
                         "n_clusters": len(set(labels[valid]))})
        continue
    results.append({
        "method": name,
        "silhouette": silhouette_score(X[valid], labels[valid]),
        "davies_bouldin": davies_bouldin_score(X[valid], labels[valid]),
        "calinski_harabasz": calinski_harabasz_score(X[valid], labels[valid]),
        "n_clusters": len(set(labels[valid])),
    })

metrics_df = pd.DataFrame(results)
metrics_df
""")

code("""\
# How well does an unsupervised clustering's split line up with the true ship/noise label?
kmeans_labels = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(X)
pd.crosstab(df["label"], pd.Series(kmeans_labels, name="kmeans_cluster"))
""")

md("""\
## 6. Honest takeaways

- **Not the same as the paper's out of the box.** The paper's ~30Hz vessel
  peak was found on the SHEFA-2 telecom cable with its own gauge length,
  coupling, and vessel population; the histogram and PSD plots above show
  whether this cable's confirmed close crossings (<= 300m) show a
  comparably distinct peak, or whether ship/noise PSDs largely overlap here.
- **PCA/clustering read the same way**: if the ship/noise points overlap in
  the PC1/PC2 scatter and the KMeans/HDBSCAN clustering metrics are far
  from the "true label" ceiling row, band-power alone isn't yet a reliable
  separator for *this* cable/dataset at these ranges - a real, checkable
  result, not a refusal to look.
- **Sample size caveat**: only a few dozen confirmed close (<=300m)
  crossings exist in the file scope used here; that limits how much
  confidence to place in either a positive or negative finding - re-run
  `extract_ship_psd_features.py` over the full (uncapped) file set for a
  larger sample before treating this as conclusive.
""")

nb["cells"] = cells
path = "/home/ph/git/das-utils/notebooks/ship_frequency_analysis.ipynb"
with open(path, "w") as f:
    nbf.write(nb, f)
print(f"wrote {path}")
