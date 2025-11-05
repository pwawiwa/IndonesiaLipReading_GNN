# src/feats/feature_pipeline.py
import os
from pathlib import Path
import math
import torch
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.stats import skew, kurtosis
from scipy.fft import rfft, rfftfreq
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from scipy.stats import f_oneway, kruskal
import warnings
warnings.filterwarnings("ignore")

# -------------------- utils -------------------- #
def safe_load_pt(p):
    return torch.load(p, weights_only=False)

def map_video_to_label(root_dataset_dir="data/IDLRW-DATASET"):
    """
    Scan dataset folders to map filename -> word label.
    Assumes structure data/IDLRW-DATASET/<word>/<split>/*.mp4
    """
    root = Path(root_dataset_dir)
    mapping = {}
    for word_dir in root.iterdir():
        if not word_dir.is_dir():
            continue
        for split_dir in (word_dir / "train", word_dir / "val", word_dir / "test"):
            if not split_dir.exists():
                continue
            for p in split_dir.glob("**/*.mp4"):
                mapping[p.name] = word_dir.name
    return mapping

# -------------------- low-level geometry helpers -------------------- #
def euclid(a, b):
    return np.linalg.norm(a - b)

def frame_center(landmarks):
    # landmarks: (N, 3)
    return np.mean(landmarks[:, :2], axis=0)

def convex_hull_area(points2d):
    try:
        hull = ConvexHull(points2d)
        return hull.volume  # for 2D convex hull, volume is area
    except Exception:
        return 0.0

def compute_EAR(eye_pts):
    # eye_pts: array shape (n_points, 2)
    # generic formula using 6-eye points (approx). For mediapipe eye points choose appropriate indices.
    # We'll fallback to a simple ratio if fewer points.
    if eye_pts.shape[0] < 6:
        return 0.0
    # use two vertical distances divided by horizontal
    A = euclid(eye_pts[1], eye_pts[5])
    B = euclid(eye_pts[2], eye_pts[4])
    C = euclid(eye_pts[0], eye_pts[3]) + 1e-8
    return (A + B) / (2.0 * C)

def compute_MAR(mouth_pts):
    # mouth_pts: should include outer lip corner and vertical lip points
    # We'll use vertical distances (avg) / horizontal width
    left = mouth_pts[0]
    right = mouth_pts[6] if mouth_pts.shape[0] > 6 else mouth_pts[-1]
    horiz = euclid(left, right) + 1e-8
    # vertical points: sample mid-top and mid-bottom
    idx_top = mouth_pts.shape[0]//2 - 1
    idx_bot = mouth_pts.shape[0]//2
    vert = euclid(mouth_pts[idx_top], mouth_pts[idx_bot])
    return vert / horiz

def temporal_deltas(seq, order=1):
    if order == 1:
        return np.diff(seq, axis=0)
    elif order == 2:
        return np.diff(seq, n=2, axis=0)
    else:
        raise ValueError("order only 1 or 2")

def frame_freq_energy(signal, fps, low_hz=0.5, high_hz=8.0):
    # signal: 1D array length T
    if len(signal) < 4:
        return 0.0
    yf = np.abs(rfft(signal))
    xf = rfftfreq(len(signal), 1.0/fps)
    mask = (xf >= low_hz) & (xf <= high_hz)
    return np.sum(yf[mask]) / (np.sum(yf)+1e-8)

# -------------------- AU proxies mapping (indices) --------------------
# NOTE: Mediapipe 468 landmark indices: we need groups. We'll define groups for mouth, eyes, brows, nose, jaw.
# These index groups are approximate; adjust if you use different mapping.
MOUTH_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308]  # outer lip (approx)
MOUTH_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]  # inner
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133]  # approximate
RIGHT_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 362]
LEFT_BROW = [70, 63, 105, 66]
RIGHT_BROW = [300, 293, 334, 296]
NOSE_TIP = [1]  # approximate nose tip index
JAW = [152, 148, 176]  # lower face region sample
CHEEK_L = [50]  # approximate
CHEEK_R = [280]

# -------------------- per-video feature calculator -------------------- #
def video_to_features(entry, fps=25):
    """
    entry: dict with keys 'video' and 'landmarks' where landmarks is (T, 468, 3)
    returns: dict {feature_name: value}
    """
    name = entry["video"]
    lm_seq = entry["landmarks"]  # (T, 468, 3)
    T = lm_seq.shape[0]
    feats = {"video": name, "frames": T}

    # normalize by nose tip (2D) and scale by face size (distance between two eyes)
    nose_idx = NOSE_TIP[0]
    nose = lm_seq[:, nose_idx, :2]  # (T,2)
    left_eye_center = np.mean(lm_seq[:, LEFT_EYE, :2], axis=1)
    right_eye_center = np.mean(lm_seq[:, RIGHT_EYE, :2], axis=1)
    eye_dist = np.linalg.norm(left_eye_center - right_eye_center, axis=1).mean() + 1e-8

    # Flattened PCA: top components of landmark positions over time (to capture dominant motions)
    flat = lm_seq[:, :, :2].reshape(T, -1)  # (T, 468*2)
    try:
        pca = PCA(n_components=3)
        pca_o = pca.fit_transform(flat)  # (T,3)
        feats["pca_var0"] = float(np.var(pca_o[:,0]))
        feats["pca_var1"] = float(np.var(pca_o[:,1]))
        feats["pca_var2"] = float(np.var(pca_o[:,2]))
    except Exception:
        feats["pca_var0"] = feats["pca_var1"] = feats["pca_var2"] = 0.0

    # Node motion features: mean speed and std
    coords2d = lm_seq[:, :, :2]  # (T,468,2)
    coords3d = lm_seq  # include z for some features
    # normalize coords (center at nose)
    coords2d_n = coords2d - nose[:, None, :]
    coords2d_n = coords2d_n / eye_dist  # scale normalized by eye distance

    velocities = np.vstack([np.zeros((1, coords2d_n.shape[1], 2)), np.diff(coords2d_n, axis=0)])  # (T,468,2)
    speeds = np.linalg.norm(velocities, axis=2)  # (T,468)
    feats["node_speed_mean"] = float(np.mean(speeds))
    feats["node_speed_std"] = float(np.std(speeds))

    acc = np.vstack([np.zeros((2, coords2d_n.shape[1], 2)), np.diff(coords2d_n, n=2, axis=0)])
    feats["node_acc_mean"] = float(np.mean(np.linalg.norm(acc, axis=2)))
    feats["node_acc_std"] = float(np.std(np.linalg.norm(acc, axis=2)))

    # global motion energy (per-frame)
    frame_motion = np.linalg.norm(np.mean(velocities, axis=1), axis=1)  # (T,)
    feats["motion_mean_per_frame"] = float(np.mean(frame_motion))
    feats["motion_std_per_frame"] = float(np.std(frame_motion))
    feats["motion_freq_energy"] = float(frame_freq_energy(frame_motion, fps))

    # mouth metrics per frame
    mouth_outer = coords2d_n[:, MOUTH_OUTER, :]  # (T, m, 2)
    mouth_inner = coords2d_n[:, MOUTH_INNER, :] if len(MOUTH_INNER) <= coords2d_n.shape[1] else mouth_outer
    # compute MAR, width, area
    mar_seq = []
    mouth_width_seq = []
    mouth_area_seq = []
    for t in range(T):
        mo = mouth_outer[t]
        try:
            mar = compute_MAR(mo)
        except Exception:
            mar = 0.0
        mouth_width_seq.append(float(euclid(mo[0], mo[-1])))
        mar_seq.append(float(mar))
        try:
            mouth_area_seq.append(convex_hull_area(mo))
        except Exception:
            mouth_area_seq.append(0.0)

    feats["mar_mean"] = float(np.mean(mar_seq))
    feats["mar_std"] = float(np.std(mar_seq))
    feats["mouth_width_mean"] = float(np.mean(mouth_width_seq))
    feats["mouth_area_mean"] = float(np.mean(mouth_area_seq))
    feats["mouth_area_std"] = float(np.std(mouth_area_seq))

    # lip corner pull / AU12 proxy: horizontal pull / smile
    # using mouth outer first and last points as corners approximation
    lip_corner_h = coords2d_n[:, MOUTH_OUTER[0], 0] - coords2d_n[:, MOUTH_OUTER[-1], 0]
    feats["lipcorner_h_mean"] = float(np.mean(np.abs(lip_corner_h)))
    feats["lipcorner_h_std"] = float(np.std(np.abs(lip_corner_h)))

    # AU proxies collective: compute per-frame then aggregate
    au_means = {}
    # AU12: Lip corner puller (horizontal increase)
    au12 = np.abs(coords2d_n[:, MOUTH_OUTER[0], 0] - coords2d_n[:, MOUTH_OUTER[-1], 0])
    au_means["AU12_mean"] = float(au12.mean())
    au_means["AU12_max"] = float(au12.max())
    # AU25: Lips part (MAR)
    au_means["AU25_mean"] = float(np.mean(mar_seq))
    # AU6: cheek raise proxy (distance cheek->eye decreases)
    cheekL = coords2d_n[:, CHEEK_L[0], :] if CHEEK_L[0] < coords2d_n.shape[1] else np.zeros((T,2))
    left_eye_c = np.mean(coords2d_n[:, LEFT_EYE, :], axis=1)
    cheek_dist_seq = np.linalg.norm(cheekL - left_eye_c, axis=1)
    au_means["AU6_mean"] = float(np.mean(cheek_dist_seq))
    # AU1/AU2 eyebrow raise (vertical difference)
    browL = np.mean(coords2d_n[:, LEFT_BROW, :], axis=1)
    browR = np.mean(coords2d_n[:, RIGHT_BROW, :], axis=1)
    au_means["AU1_mean"] = float(np.mean(-browL[:,1]))  # negative because upward is smaller y after normalization? keep sign raw

    # aggregate AU_means into feats
    for k, v in au_means.items():
        feats[k] = v

    # eye blink / EAR proxies
    left_eye_seq = [compute_EAR(coords2d_n[t, LEFT_EYE, :]) for t in range(T)]
    right_eye_seq = [compute_EAR(coords2d_n[t, RIGHT_EYE, :]) for t in range(T)]
    feats["left_EAR_mean"] = float(np.mean(left_eye_seq))
    feats["right_EAR_mean"] = float(np.mean(right_eye_seq))
    feats["blink_rate_proxy"] = float(np.sum(np.array(left_eye_seq) < 0.12)) / (T+1e-8)

    # symmetry: correlation between left & right landmarks (x coordinates)
    left_idxs = np.array(LEFT_EYE + LEFT_BROW + MOUTH_OUTER)
    right_idxs = np.array(RIGHT_EYE + RIGHT_BROW + [468 - i for i in MOUTH_OUTER if (468 - i) < coords2d_n.shape[1]])  # approximate
    # safe fallback
    try:
        left_mean = np.mean(coords2d_n[:, left_idxs, 0], axis=1)
        right_mean = np.mean(coords2d_n[:, right_idxs, 0], axis=1)
        feats["symmetry_corr"] = float(np.corrcoef(left_mean, right_mean)[0,1])
    except Exception:
        feats["symmetry_corr"] = 0.0

    # global temporal statistics for a few key signals
    def agg_stats(x):
        return {
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
            "min": float(np.min(x)),
            "max": float(np.max(x)),
            "median": float(np.median(x)),
            "skew": float(skew(x)) if len(x)>2 else 0.0,
            "kurtosis": float(kurtosis(x)) if len(x)>3 else 0.0
        }
    # speed distribution
    sp = speeds.mean(axis=1) if speeds.shape[0]>0 else np.zeros(T)
    stats = agg_stats(sp)
    for sname, sval in stats.items():
        feats[f"speed_{sname}"] = sval

    # mouth opening peaks (count)
    peaks = np.sum((np.array(mar_seq) > (np.mean(mar_seq) + 1.5*np.std(mar_seq))).astype(int))
    feats["mouth_open_peaks"] = int(peaks)

    # finalize
    return feats

# -------------------- Top-level runner -------------------- #
def process_split_pt(pt_path, map_name_to_label, out_csv_path, fps=25):
    print(f"Loading {pt_path} ...")
    data = safe_load_pt(pt_path)
    rows = []
    for entry in data:
        f = video_to_features(entry, fps=fps)
        # attach label if known
        vid = f["video"]
        label = map_name_to_label.get(vid, "unknown")
        f["label"] = label
        rows.append(f)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv_path, index=False)
    print(f"Saved features to {out_csv_path} ({len(df)} rows)")
    return df

# -------------------- Statistical tests -------------------- #
def run_feature_tests(df, out_prefix):
    # filter labeled
    df = df[df["label"] != "unknown"].copy()
    features = [c for c in df.columns if c not in ("video", "label", "frames")]

    # ANOVA / Kruskal per feature (multi-class)
    anova_rows = []
    kruskal_rows = []
    for feat in features:
        groups = []
        labels = df["label"].unique()
        for lab in labels:
            groups.append(df[df["label"]==lab][feat].dropna().values)
        try:
            stat, p = f_oneway(*groups)
        except Exception:
            stat, p = np.nan, np.nan
        anova_rows.append({"feature": feat, "f_stat": stat, "p_value": p})

        try:
            ks, kp = kruskal(*groups)
        except Exception:
            ks, kp = np.nan, np.nan
        kruskal_rows.append({"feature": feat, "kw_stat": ks, "p_value": kp})

    pd.DataFrame(anova_rows).to_csv(f"{out_prefix}_anova.csv", index=False)
    pd.DataFrame(kruskal_rows).to_csv(f"{out_prefix}_kruskal.csv", index=False)

    # Mutual information (requires numeric label encoding)
    labs, inv = pd.factorize(df["label"])
    X = df[features].fillna(0).values
    mi = mutual_info_classif(X, labs, discrete_features=False, random_state=42)
    mi_df = pd.DataFrame({"feature": features, "mi": mi}).sort_values("mi", ascending=False)
    mi_df.to_csv(f"{out_prefix}_mi.csv", index=False)

    # RandomForest importances (fast feature importance)
    clf = RandomForestClassifier(n_estimators=200, n_jobs=4, random_state=42)
    try:
        clf.fit(StandardScaler().fit_transform(X), labs)
        importances = clf.feature_importances_
    except Exception:
        importances = np.zeros(len(features))
    rf_df = pd.DataFrame({"feature": features, "rf_imp": importances}).sort_values("rf_imp", ascending=False)
    rf_df.to_csv(f"{out_prefix}_rf.csv", index=False)

    print("Stat tests saved: anova / kruskal / mi / rf")

# -------------------- CLI -------------------- #
def main():
    # config
    processed_root = Path("data/processed_all")  # where global train.pt etc are
    dataset_root = Path("data/IDLRW-DATASET")
    out_dir = Path("data/features")
    out_dir.mkdir(parents=True, exist_ok=True)

    map_name_to_label = map_video_to_label(str(dataset_root))

    for split in ["train", "val", "test"]:
        pt_file = processed_root / f"{split}.pt"
        if not pt_file.exists():
            print(f"Skipping missing {pt_file}")
            continue
        out_csv = out_dir / f"features_{split}.csv"
        df = process_split_pt(pt_file, map_name_to_label, out_csv, fps=25)

        # run tests only on train by default (smaller)
        if split == "train":
            run_feature_tests(df, str(out_dir / "train_features"))

if __name__ == "__main__":
    main()
