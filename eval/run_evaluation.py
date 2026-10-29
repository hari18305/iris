"""Full evaluation: subject-disjoint dev/test protocol, ablations, robustness, timing, figures.

Usage:  python eval/build_cache.py   (once)   then   python eval/run_evaluation.py
Outputs: report/results/metrics.json, report/figures/*.png|pdf
"""
from __future__ import annotations

import glob
import json
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from eval import metrics as mt  # noqa: E402
from eval.build_cache import CACHE, DATASET, eye_id  # noqa: E402
from eval.scores import distance_matrix  # noqa: E402
from src import encoding as enc  # noqa: E402
from src import matching  # noqa: E402
from src import normalization as nm  # noqa: E402
from src import segmentation as sg  # noqa: E402
from src.pipeline import load_gray, process_image  # noqa: E402

DEV_MAX_SUBJECT = 60            # subjects 001-060 tune parameters; 061-249 are never used for tuning
OUT_RES, OUT_FIG = "report/results", "report/figures"
os.makedirs(OUT_RES, exist_ok=True)
os.makedirs(OUT_FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "figure.dpi": 150, "savefig.dpi": 300,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
})
C_GEN, C_IMP, C_ACC = "#1f4e9c", "#c0392b", "#555555"


def save(fig, name):
    fig.savefig(os.path.join(OUT_FIG, name + ".png"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT_FIG, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def encode_all(strips, masks, band=(0, 12), all_valid=False):
    old = enc.ROW_BAND
    enc.ROW_BAND = band
    try:
        cs, ms = [], []
        for s, m in zip(strips, masks):
            c, k = enc.encode(s, np.ones_like(m) if all_valid else m)
            cs.append(c)
            ms.append(k)
    finally:
        enc.ROW_BAND = old
    return np.array(cs), np.array(ms)


def rank1(dist, labels):
    first = {}
    for i, e in enumerate(labels):
        first.setdefault(e, i)
    gallery = np.array(sorted(first.values()))
    probes = np.array([i for i in range(len(labels)) if i not in set(gallery)])
    nearest = gallery[np.argmin(dist[np.ix_(probes, gallery)], axis=1)]
    return float((labels[nearest] == labels[probes]).mean()), len(probes), len(gallery)


def main():
    t_start = time.time()
    d = dict(np.load(CACHE))
    files, eyes, strips, masks, segs = d["files"], d["eyes"], d["strips"], d["masks"], d["segs"]
    all_files = sorted(glob.glob(f"{DATASET}/*/*/*.jpg"))
    subj = np.array([int(e[:3]) for e in eyes])
    dev, test = np.where(subj <= DEV_MAX_SUBJECT)[0], np.where(subj > DEV_MAX_SUBJECT)[0]
    res = {"dataset": {
        "total_images": len(all_files), "segmented": int(len(files)), "failed_to_acquire": int(len(all_files) - len(files)),
        "fta_rate": (len(all_files) - len(files)) / len(all_files),
        "subjects": int(len(set(e[:3] for e in eyes))), "eyes": int(len(set(eyes))),
        "dev_images": int(len(dev)), "dev_eyes": int(len(set(eyes[dev]))),
        "test_images": int(len(test)), "test_eyes": int(len(set(eyes[test]))),
        "dev_subjects": f"001-{DEV_MAX_SUBJECT:03d}", "test_subjects": f"{DEV_MAX_SUBJECT + 1:03d}-249",
        "image_size": [280, 320],
    }}

    # ---- encode everything with the final parameters ------------------------------------------
    codes, cmasks = encode_all(strips, masks)

    # ---- threshold selected on DEV only -------------------------------------------------------
    Dd = distance_matrix(codes[dev], cmasks[dev])
    gen_d, imp_d = mt.split_pairs(Dd, eyes[dev])
    eer_dev, thr_dev = mt.eer(gen_d, imp_d)
    res["dev"] = mt.summarize(gen_d, imp_d, matching.MATCH_THRESHOLD)
    res["dev"]["eer_threshold_dev"] = thr_dev
    print(f"dev: EER {eer_dev:.4f} at t={thr_dev:.3f}; deployed threshold {matching.MATCH_THRESHOLD}")
    op_thr = {"Balanced (dev EER point)": thr_dev}
    for far_target, label in ((1e-3, "High security (dev FAR = 0.1%)"), (1e-4, "Maximum security (dev FAR = 0.01%)")):
        op_thr[label] = mt.frr_at_far(gen_d, imp_d, far_target)[1]

    # ---- final test results -------------------------------------------------------------------
    Dt = distance_matrix(codes[test], cmasks[test])
    lab_t = eyes[test]
    gen, imp = mt.split_pairs(Dt, lab_t)
    res["test"] = mt.summarize(gen, imp, matching.MATCH_THRESHOLD)
    r1, n_probe, n_gal = rank1(Dt, lab_t)
    res["test"].update({"rank1_identification": r1, "id_probes": n_probe, "id_gallery": n_gal})
    print("test:", {k: (round(v, 5) if isinstance(v, float) else v) for k, v in res["test"].items()})

    res["operating_points"] = {}
    for label, thr in op_thr.items():
        far_o, frr_o = mt.far_frr(gen, imp, thr)
        res["operating_points"][label] = {"threshold": float(thr), "far": far_o, "frr": frr_o}
        print("operating point", label, round(thr, 4), "test FAR %.5f FRR %.4f" % (far_o, frr_o))

    # ---- ablations (test set) -----------------------------------------------------------------
    abl = {}

    def add_ablation(name, g, i, extra=None):
        s = mt.summarize(g, i, matching.MATCH_THRESHOLD)
        abl[name] = {"eer": s["eer"], "d_prime": s["d_prime"], "frr_at_far_0.001": s["frr_at_far_0.001"],
                     "genuine_mean": s["genuine_mean"], "impostor_mean": s["impostor_mean"], **(extra or {})}
        print("ablation", name, {k: round(v, 4) for k, v in abl[name].items()})

    add_ablation("Full system", gen, imp)
    g, i = mt.split_pairs(distance_matrix(codes[test], cmasks[test], shifts=(0,)), lab_t)
    add_ablation("No rotation compensation (shift = 0 only)", g, i)
    c2, m2 = encode_all(strips[test], masks[test], all_valid=True)
    g, i = mt.split_pairs(distance_matrix(c2, m2), lab_t)
    add_ablation("No noise mask (all pixels treated as valid)", g, i)
    c3, m3 = encode_all(strips[test], masks[test], band=(0, 16))
    g, i = mt.split_pairs(distance_matrix(c3, m3), lab_t)
    add_ablation("All radial bands (no inner-band restriction)", g, i)
    # No iris-radius prior: re-segment the test images without the acquisition prior.
    old_prior = sg.IRIS_RADIUS_RANGE
    sg.IRIS_RADIUS_RANGE = (0.0, 1e9)
    cs, ms, lab_np, fail_np = [], [], [], 0
    for f in files[test]:
        im = load_gray(str(f))
        try:
            seg = sg.segment(im)
            st, mk = nm.normalize(im, seg)
            c, m = enc.encode(st, mk)
        except sg.SegmentationError:
            fail_np += 1
            continue
        cs.append(c)
        ms.append(m)
        lab_np.append(eye_id(str(f)))
    sg.IRIS_RADIUS_RANGE = old_prior
    g, i = mt.split_pairs(distance_matrix(np.array(cs), np.array(ms)), np.array(lab_np))
    add_ablation("No iris-radius prior in segmentation", g, i, {"failures": fail_np})
    res["ablation"] = abl

    # ---- robustness (probe degradation vs clean gallery) -------------------------------------
    first_idx = {}
    for k in test:
        first_idx.setdefault(eyes[k], k)
    multi = [e for e in first_idx if (lab_t == e).sum() >= 2]
    probe_idx = np.array([first_idx[e] for e in multi])
    probe_labels = np.array(multi)
    pos_in_test = {k: n for n, k in enumerate(test)}

    def rotate(a):
        def f(im):
            M = cv2.getRotationMatrix2D((im.shape[1] / 2, im.shape[0] / 2), a, 1.0)
            return cv2.warpAffine(im, M, (im.shape[1], im.shape[0]), borderMode=cv2.BORDER_REPLICATE)
        return f

    rng = np.random.default_rng(0)

    def noise(sigma):
        return lambda im: np.clip(im.astype(np.float32) + rng.normal(0, sigma, im.shape), 0, 255).astype(np.uint8)

    degr = {
        "None (clean probe)": lambda im: im,
        "Gaussian noise, sigma = 10": noise(10),
        "Gaussian noise, sigma = 25": noise(25),
        "Gaussian blur, sigma = 2 px": lambda im: cv2.GaussianBlur(im, (0, 0), 2.0),
        "Gaussian blur, sigma = 4 px": lambda im: cv2.GaussianBlur(im, (0, 0), 4.0),
        "Brightness x 0.6": lambda im: np.clip(im.astype(np.float32) * 0.6, 0, 255).astype(np.uint8),
        "Brightness x 1.4": lambda im: np.clip(im.astype(np.float32) * 1.4, 0, 255).astype(np.uint8),
        "Rotation 5 deg": rotate(5), "Rotation 10 deg": rotate(10), "Rotation 20 deg": rotate(20),
    }
    rob = {}
    for name, fn in degr.items():
        pc, pm, keep = [], [], []
        fta = 0
        for n, k in enumerate(probe_idx):
            im = fn(load_gray(str(files[k])))
            try:
                st = process_image(im)
            except sg.SegmentationError:
                fta += 1
                continue
            pc.append(st.code)
            pm.append(st.code_mask)
            keep.append(n)
        D = distance_matrix(np.array(pc), np.array(pm), codes[test], cmasks[test])
        gen_l, imp_l = [], []
        for row, n in enumerate(keep):
            own = pos_in_test[probe_idx[n]]
            same = (lab_t == probe_labels[n])
            same[own] = False
            gen_l.append(D[row][same])
            imp_l.append(D[row][lab_t != probe_labels[n]])
        g_, i_ = np.concatenate(gen_l), np.concatenate(imp_l)
        far_t, frr_t = mt.far_frr(g_, i_, matching.MATCH_THRESHOLD)
        rob[name] = {"probes": len(probe_idx), "fta": fta, "frr": frr_t, "far": far_t,
                     "genuine_mean": float(g_.mean()), "impostor_mean": float(i_.mean()),
                     "eer": mt.eer(g_, i_)[0]}
        print("robustness", name, {k: round(v, 4) for k, v in rob[name].items()})
    res["robustness"] = rob

    # ---- timing --------------------------------------------------------------------------------
    random.seed(0)
    sample = random.sample([str(f) for f in files[test]], 100)
    imgs = [load_gray(f) for f in sample]
    t = {"segmentation": 0.0, "normalization": 0.0, "encoding": 0.0}
    done = []
    for im in imgs:
        a = time.perf_counter(); seg = sg.segment(im)
        b = time.perf_counter(); st, mk = nm.normalize(im, seg)
        c = time.perf_counter(); cd, cm = enc.encode(st, mk)
        e = time.perf_counter()
        t["segmentation"] += b - a; t["normalization"] += c - b; t["encoding"] += e - c
        done.append((cd, cm))
    res["timing_ms"] = {k: v / len(imgs) * 1000 for k, v in t.items()}
    a = time.perf_counter()
    reps = 0
    for x in range(len(done)):
        for y in range(x + 1, min(x + 21, len(done))):
            matching.hamming_distance(done[x][0], done[x][1], done[y][0], done[y][1])
            reps += 1
    res["timing_ms"]["matching_per_comparison"] = (time.perf_counter() - a) / reps * 1000
    res["timing_ms"]["enroll_total"] = sum(res["timing_ms"][k] for k in ("segmentation", "normalization", "encoding"))
    res["timing_ms"]["verify_total"] = res["timing_ms"]["enroll_total"] + res["timing_ms"]["matching_per_comparison"]
    print("timing", {k: round(v, 3) for k, v in res["timing_ms"].items()})

    # ---- system parameters (Table I) ------------------------------------------------------------
    res["parameters"] = {
        "image_size": "320 x 280 (NIR, 8-bit)", "normalized_strip": f"{nm.RADIAL_RES} x {nm.ANGULAR_RES}",
        "log_gabor_wavelengths": list(enc.WAVELENGTHS), "log_gabor_sigma_on_f": enc.SIGMA_ON_F,
        "code_shape": list(enc.CODE_SHAPE), "code_bits": enc.CODE_LENGTH, "row_band": list(enc.ROW_BAND),
        "iris_radius_prior_px": list(sg.IRIS_RADIUS_RANGE), "rotation_shifts": [min(matching.SHIFTS), max(matching.SHIFTS)],
        "shift_degrees": 360.0 / enc.CODE_COLS, "min_overlap": matching.MIN_OVERLAP,
        "match_threshold": matching.MATCH_THRESHOLD,
    }

    # ---- hardest genuine images (for the segmentation figure) -----------------------------------
    Dg = np.where(((lab_t[:, None] == lab_t[None, :]) & ~np.eye(len(test), dtype=bool)), Dt, np.nan)
    with np.errstate(all="ignore"):
        med = np.nanmedian(Dg, axis=1)
    hardest = test[np.argsort(-np.nan_to_num(med, nan=0))[:4]]

    # ---- figures ---------------------------------------------------------------------------------
    # Fig: score distributions
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    bins = np.linspace(0.0, 0.55, 111)
    ax.hist(imp, bins=bins, density=True, color=C_IMP, alpha=0.65, label="Impostor")
    ax.hist(gen, bins=bins, density=True, color=C_GEN, alpha=0.75, label="Genuine")
    ax.axvline(matching.MATCH_THRESHOLD, color="k", ls="--", lw=1, label=f"Balanced threshold = {matching.MATCH_THRESHOLD:.2f}")
    ax.set_xlabel("Hamming distance"); ax.set_ylabel("Density"); ax.legend(loc="upper left")
    save(fig, "fig_score_distribution")

    # Fig: ROC + DET
    ts, far, frr = mt.roc_points(gen, imp, 4000)
    e, te = res["test"]["eer"], res["test"]["eer_threshold"]
    fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.5))
    m = far > 0
    axs[0].semilogx(far[m], 1 - frr[m], color=C_GEN, lw=1.3)
    axs[0].plot(e, 1 - e, "o", color=C_IMP, ms=4, label=f"EER = {100 * e:.2f}%")
    axs[0].set_xlabel("False Accept Rate (FAR)"); axs[0].set_ylabel("Genuine Accept Rate (1 - FRR)")
    axs[0].set_xlim(1e-6, 1); axs[0].set_ylim(0.85, 1.002); axs[0].legend(loc="lower right")
    both = (far > 0) & (frr > 0)
    axs[1].loglog(far[both], frr[both], color=C_GEN, lw=1.3)
    axs[1].plot([1e-6, 1], [1e-6, 1], color=C_ACC, ls=":", lw=0.8)
    axs[1].plot(e, e, "o", color=C_IMP, ms=4, label=f"EER = {100 * e:.2f}%")
    axs[1].set_xlabel("False Accept Rate (FAR)"); axs[1].set_ylabel("False Reject Rate (FRR)")
    axs[1].set_xlim(1e-6, 1); axs[1].set_ylim(1e-3, 1); axs[1].legend(loc="upper right")
    axs[0].text(0.5, -0.38, "(a) ROC curve", transform=axs[0].transAxes, ha="center")
    axs[1].text(0.5, -0.38, "(b) DET curve", transform=axs[1].transAxes, ha="center")
    save(fig, "fig_roc_det")

    # Fig: FAR/FRR vs threshold
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    ax.plot(ts, 100 * far, color=C_IMP, label="FAR"); ax.plot(ts, 100 * frr, color=C_GEN, label="FRR")
    ax.axvline(matching.MATCH_THRESHOLD, color="k", ls="--", lw=1, label="Balanced threshold")
    ax.set_xlim(0.25, 0.5); ax.set_ylim(0, 25)
    ax.set_xlabel("Decision threshold (Hamming distance)"); ax.set_ylabel("Error rate (%)"); ax.legend()
    save(fig, "fig_far_frr")

    # Fig: pipeline stages on one test image
    ex = str(files[test[np.argmin(np.abs(np.arange(len(test)) - len(test) // 3))]])
    st = process_image(load_gray(ex))
    fig = plt.figure(figsize=(7.1, 3.6))
    gs = fig.add_gridspec(3, 2, height_ratios=[3.0, 1.0, 1.0], hspace=0.45, wspace=0.08)
    a0, a1 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    a2, a3 = fig.add_subplot(gs[1, :]), fig.add_subplot(gs[2, :])
    a0.imshow(st.image, cmap="gray"); a0.set_title("(a) NIR eye image")
    a1.imshow(sg.draw_overlay(st.image, st.segmentation)[:, :, ::-1]); a1.set_title("(b) Pupil and iris boundaries")
    vis = np.dstack([st.strip] * 3).astype(float) / 255
    vis[~st.strip_mask] = vis[~st.strip_mask] * 0.3 + np.array([0.7, 0.0, 0.0]) * 0.7
    a2.imshow(vis, aspect="auto"); a2.set_title("(c) Normalized iris strip (red = masked: lashes, glare, edges)")
    a3.imshow(enc.code_bitmap(st.code, st.code_mask), cmap="gray", aspect="auto", vmin=0, vmax=255)
    a3.set_title(f"(d) Iris code: {enc.CODE_LENGTH} bits ({'x'.join(map(str, enc.CODE_SHAPE))}; grey = masked)")
    for a in (a0, a1, a2, a3):
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
    save(fig, "fig_pipeline")

    # Fig: segmentation examples
    rng2 = np.random.default_rng(5)
    good = rng2.choice(np.setdiff1d(test, hardest), 4, replace=False)
    fig, axs = plt.subplots(2, 4, figsize=(7.1, 3.9))
    for ax, k in zip(axs[0], good):
        s = segs[k]
        ax.imshow(sg.draw_overlay(load_gray(str(files[k])), ((s[0], s[1]), s[2], (s[3], s[4]), s[5]))[:, :, ::-1])
    for ax, k in zip(axs[1], hardest):
        s = segs[k]
        ax.imshow(sg.draw_overlay(load_gray(str(files[k])), ((s[0], s[1]), s[2], (s[3], s[4]), s[5]))[:, :, ::-1])
    for ax in axs.ravel():
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    axs[0, 0].set_ylabel("Typical"); axs[1, 0].set_ylabel("Hardest")
    save(fig, "fig_segmentation")

    res["meta"] = {"elapsed_s": time.time() - t_start, "example_image": ex.replace("\\", "/"),
                   "hardest_images": [str(files[k]).replace("\\", "/") for k in hardest]}
    with open(os.path.join(OUT_RES, "metrics.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"done in {time.time() - t_start:.0f}s -> {OUT_RES}/metrics.json, {OUT_FIG}/")


if __name__ == "__main__":
    main()
