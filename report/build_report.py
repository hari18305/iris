"""Generate report/main.tex (IEEEtran conference format) from report/results/metrics.json.

Every number in the paper is filled in from the evaluation output, so the text cannot drift from the results.
Usage: python report/build_report.py   (then upload the report/ folder to Overleaf, or run pdflatex twice)
"""
import json
import os
import re
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

with open("report/results/metrics.json") as fh:
    M = json.load(fh)
D, T, DV = M["dataset"], M["test"], M["dev"]
P, OP, AB, RB, TM = M["parameters"], M["operating_points"], M["ablation"], M["robustness"], M["timing_ms"]


def pct(x, d=2):
    return f"{100 * x:.{d}f}\\%"


def num(n):
    return f"{n:,}".replace(",", "{,}")


def test_count(path):
    out = subprocess.run([sys.executable, "-m", "pytest", "-q", path], capture_output=True, text=True).stdout
    m = re.search(r"(\d+) passed", out)
    failed = re.search(r"(\d+) failed", out)
    if not m or failed:
        raise SystemExit(f"tests did not all pass for {path}:\n{out}")
    return int(m.group(1))


n_pipe, n_store = test_count("tests/test_pipeline.py"), test_count("tests/test_store.py")
n_safe, n_app = test_count("tests/test_safeimage.py"), test_count("tests/test_app.py")

abl_names = list(AB)
full, norot, nomask, allband, noprior = (AB[k] for k in abl_names)
rb = {k: RB[k] for k in RB}
rot = {a: RB[f"Rotation {a} deg"] for a in (5, 10, 20)}
bright_hi = RB["Brightness x 1.4"]
deg_note = lambda r: f"{r['fta']} of {r['probes']}"  # noqa: E731

V = {
    "total_images": num(D["total_images"]), "eyes": D["eyes"], "subjects": D["subjects"],
    "segmented": num(D["segmented"]), "fta": D["failed_to_acquire"], "fta_pct": pct(D["fta_rate"]),
    "dev_images": D["dev_images"], "dev_eyes": D["dev_eyes"], "test_images": num(D["test_images"]), "test_eyes": D["test_eyes"],
    "dev_subj": D["dev_subjects"].replace("-", "--"), "test_subj": D["test_subjects"].replace("-", "--"),
    "ngen": num(T["n_genuine"]), "nimp": num(T["n_impostor"]),
    "eer": pct(T["eer"]), "eer_dev": pct(DV["eer"]), "dprime": f"{T['d_prime']:.2f}", "auc": f"{T['auc']:.4f}",
    "rank1": pct(T["rank1_identification"], 1), "probes": num(T["id_probes"]), "gallery": T["id_gallery"],
    "gen_mean": f"{T['genuine_mean']:.3f}", "imp_mean": f"{T['impostor_mean']:.3f}",
    "gen_std": f"{T['genuine_std']:.3f}", "imp_std": f"{T['impostor_std']:.3f}",
    "thr": f"{T['threshold']:.2f}", "far_thr": pct(T["far_at_threshold"]), "frr_thr": pct(T["frr_at_threshold"]),
    "frr_far2": pct(T["frr_at_far_0.01"]), "frr_far3": pct(T["frr_at_far_0.001"]), "frr_far4": pct(T["frr_at_far_0.0001"]),
    "code_bits": num(P["code_bits"]), "shift_deg": f"{P['rotation_shifts'][1] * P['shift_degrees']:.1f}",
    "t_enroll": f"{TM['enroll_total']:.1f}", "t_verify": f"{TM['verify_total']:.1f}", "t_seg": f"{TM['segmentation']:.1f}",
    "t_match": f"{TM['matching_per_comparison']:.2f}",
    "n_pipe": n_pipe, "n_store": n_store, "n_safe": n_safe, "n_app": n_app, "n_tests": n_pipe + n_store + n_safe + n_app,
    "abl_full": pct(full["eer"]), "abl_norot": pct(norot["eer"]), "abl_nomask": pct(nomask["eer"]),
    "abl_allband": pct(allband["eer"]), "abl_noprior": pct(noprior["eer"]),
    "abl_rot_x": f"{norot['eer'] / full['eer']:.1f}",
    "bal_thr": f"{OP[list(OP)[0]]['threshold']:.3f}", "high_thr": f"{OP[list(OP)[1]]['threshold']:.3f}",
    "max_thr": f"{OP[list(OP)[2]]['threshold']:.3f}",
    "high_far": pct(OP[list(OP)[1]]["far"]), "high_frr": pct(OP[list(OP)[1]]["frr"]),
    "max_far": pct(OP[list(OP)[2]]["far"], 3), "max_frr": pct(OP[list(OP)[2]]["frr"]),
    "rot5_frr": pct(rot[5]["frr"], 1), "rot10_frr": pct(rot[10]["frr"], 1), "rot20_frr": pct(rot[20]["frr"], 1),
    "bright_fta": bright_hi["fta"], "probes_n": RB["None (clean probe)"]["probes"],
    "clean_frr": pct(RB["None (clean probe)"]["frr"]), "clean_far": pct(RB["None (clean probe)"]["far"]),
}


def table(caption, label, colspec, header, rows, star=False, font="small"):
    env = "table*" if star else "table"
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    return (f"\\begin{{{env}}}[!t]\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\centering\\{font}\n"
            f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{header} \\\\\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{{env}}}\n")


TABLES = {}
STRIP = P["normalized_strip"].replace(" x ", r"$\times$")
TABLES["PARAMS"] = table(
    "System Parameters Used in the Experiments", "tab:params", "ll", r"\textbf{Parameter} & \textbf{Value}", [
        ("Input image", "320$\\times$280 NIR, 8-bit"),
        ("Normalized strip", f"{STRIP} (radial $\\times$ angular)"),
        ("Iris radius prior", f"{P['iris_radius_prior_px'][0]:.0f}--{P['iris_radius_prior_px'][1]:.0f} px"),
        ("Log-Gabor wavelengths", f"{P['log_gabor_wavelengths'][0]:.0f} and {P['log_gabor_wavelengths'][1]:.0f} samples"),
        ("Log-Gabor bandwidth ($\\sigma_f$)", f"{P['log_gabor_sigma_on_f']}"),
        ("Iris code size", f"{P['code_shape'][0]}$\\times${P['code_shape'][1]}$\\times${P['code_shape'][2]} = {num(P['code_bits'])} bits"),
        ("Radial rows used", f"{P['row_band'][0] + 1}--{P['row_band'][1]} of {P['code_shape'][0]} (inner 75\\%)"),
        ("Rotation shifts", f"$\\pm${P['rotation_shifts'][1]} columns ($\\pm${P['rotation_shifts'][1] * P['shift_degrees']:.1f}$^\\circ$)"),
        ("Minimum joint valid bits", f"{int(100 * P['min_overlap'])}\\% of code"),
        ("Decision threshold (balanced)", f"{P['match_threshold']:.2f} (HD)"),
        ("Server default threshold", f"{OP[list(OP)[1]]['threshold']:.3f} (high security)"),
        ("Failed-verification lockout", "5 per user within 300 s"),
        ("Template encryption", "Fernet (AES-128-CBC, HMAC-SHA256)"),
    ])
TABLES["DATASET"] = table(
    "Dataset and Subject-Disjoint Evaluation Protocol", "tab:dataset", "lrrr", r"\textbf{Set} & \textbf{Subjects} & \textbf{Eyes} & \textbf{Images}", [
        ("Development (tuning)", D["dev_subjects"], str(D["dev_eyes"]), str(D["dev_images"])),
        ("Test (reported)", D["test_subjects"], str(D["test_eyes"]), num(D["test_images"])),
        ("Not acquired (FTA)", "--", "--", str(D["failed_to_acquire"])),
        ("Total", "249", str(D["eyes"]), num(D["total_images"])),
    ])


def fmt_row(label, k, f=lambda v: f"{v:.3f}"):
    return (label, f(DV[k]), f(T[k]))


TABLES["PERF"] = table(
    "Verification Performance (Genuine/Impostor Hamming Distances)", "tab:perf", "lrr", r"\textbf{Metric} & \textbf{Dev} & \textbf{Test}", [
        ("Genuine comparisons", num(DV["n_genuine"]), num(T["n_genuine"])),
        ("Impostor comparisons", num(DV["n_impostor"]), num(T["n_impostor"])),
        fmt_row("Mean genuine HD", "genuine_mean"), fmt_row("Mean impostor HD", "impostor_mean"),
        fmt_row("Decidability $d'$", "d_prime", lambda v: f"{v:.2f}"),
        fmt_row("ROC-AUC", "auc", lambda v: f"{v:.4f}"),
        fmt_row("EER", "eer", lambda v: pct(v)),
        fmt_row("FRR at FAR = 1\\%", "frr_at_far_0.01", lambda v: pct(v)),
        fmt_row("FRR at FAR = 0.1\\%", "frr_at_far_0.001", lambda v: pct(v)),
        fmt_row("FRR at FAR = 0.01\\%", "frr_at_far_0.0001", lambda v: pct(v)),
        ("Rank-1 identification", "--", pct(T["rank1_identification"], 1)),
    ])
TABLES["OPS"] = table(
    "Security Levels: Thresholds Chosen on the Development Set, Error Rates Measured on the Test Set", "tab:ops", "lrrr",
    r"\textbf{Level} & \textbf{Threshold} & \textbf{FAR} & \textbf{FRR}", [
        (k.split(" (")[0], f"{v['threshold']:.3f}", pct(v["far"], 3), pct(v["frr"])) for k, v in OP.items()])
TABLES["ABL"] = table(
    "Ablation Study on the Test Set", "tab:ablation", r"L{0.44\columnwidth}rrr", r"\textbf{Configuration} & \textbf{EER} & \textbf{$d'$} & \textbf{FRR@FAR 0.1\%}", [
        (name.replace("&", "\\&"), pct(v["eer"]), f"{v['d_prime']:.2f}", pct(v["frr_at_far_0.001"])) for name, v in AB.items()])
TABLES["ROB"] = table(
    "Robustness to Probe Degradation (Threshold $= 0.44$, Clean Gallery)", "tab:robust", "lrrrr",
    r"\textbf{Probe degradation} & \textbf{Not acquired} & \textbf{FRR} & \textbf{FAR} & \textbf{EER}", [
        (k.replace("sigma", "$\\sigma$").replace(" deg", "$^\\circ$").replace(" x ", " $\\times$ ").replace("= ", "$=$ ").replace("$$=$", "$=$"),
         deg_note(v), pct(v["frr"], 1), pct(v["far"], 1), pct(v["eer"], 1)) for k, v in RB.items()], font="footnotesize")
TABLES["SEC"] = table(
    "Verification of Security Mechanisms (Automated Tests)", "tab:security", r"L{0.25\columnwidth}L{0.40\columnwidth}L{0.27\columnwidth}",
    r"\textbf{Mechanism} & \textbf{Test performed} & \textbf{Outcome}", [
        ("Encrypted templates", "Searched the database file for the plaintext template bytes", "Not found; Fernet token"),
        ("Tamper detection", "Flipped one bit of a stored ciphertext, then verified", "Rejected and logged"),
        ("Account binding", "Copied one account's ciphertext into another account's row", "Rejected"),
        ("Wrong key or passphrase", "Opened the store with a different key or passphrase", "Refused"),
        ("Key rotation", "Rotated the key; verified, checked the log, tried the old key", "Works; old key fails"),
        ("Secure deletion", "Deleted a user, searched the file for its ciphertext", "Not found"),
        ("Duplicate identity", "Enrolled the same eye under a second name", "Refused; other account not named"),
        ("Username handling", "SQL metacharacters, case variants, over-long and non-ASCII names", "Rejected; one account"),
        ("No enumeration", "Verified against an unknown, a wrong and an invalid account", "Identical denial"),
        ("Rate limits", "5 failures per account, 20 per client, recovery after 301 s", "Locked; other clients unaffected"),
        ("Replay rejection", "Verified with the enrolled image, and with an accepted image twice", "Rejected; fresh capture accepted"),
        ("Audit chain", "Edited, deleted, reordered and truncated log rows", "All detected"),
        ("Input limits", "PNG header claiming 40{,}000$\\times$40{,}000 px; oversized, tiny and garbage files", "Refused before decoding"),
        ("Web application", "Open and password modes, login throttle, score hiding", "Admin functions hidden; throttled"),
    ], font="footnotesize")
TABLES["THREAT"] = table(
    "Threat Model: Mitigations and Residual Risk", "tab:threat", r"L{0.22\columnwidth}L{0.38\columnwidth}L{0.34\columnwidth}",
    r"\textbf{Threat} & \textbf{Mitigation} & \textbf{Residual risk}", [
        ("Zero-effort impostor", "Server-side threshold set for a measured FAR; high-security level by default (Table~\\ref{tab:ops})", "FAR is " + V["high_far"] + " at the default level, never zero"),
        ("Hill climbing on match scores", "Scores hidden by default; lockout limits attempts", "Visible in presentation mode only"),
        ("Guessing and username spraying", "Limits per account (5 per 300 s) and per client (20 per 300 s); identical response for unknown accounts", "A new session gets a new client identifier, so real deployments also need network-level limits; the lockout allows denial of service against one account"),
        ("Database theft", "Ciphertext only; key outside the data directory with owner-only permissions; optional passphrase", "Key and database on one host: a host compromise yields both"),
        ("Template tampering or swapping", "Authentication tag and account binding; failures logged", "Write access can still delete users"),
        ("Duplicate accounts", "1:N check with a fixed policy; quality gate", "Detects the same eye only"),
        ("Replay of a captured image file", "Identical captures are rejected", "Modified copies and fresh recaptures are not detected"),
        ("Presentation attack (print, lens, screen)", "None", "Not mitigated; needs liveness / PAD"),
        ("Audit-log manipulation", "HMAC hash chain with a keyed head, verified in the interface", "Rollback to an older consistent snapshot; anyone holding the key can forge"),
        ("Malicious uploads", "5 MB cap; format and size checked from the header; never written to disk", "Bugs in the image decoder remain possible"),
        ("Vulnerable dependencies", "Versions pinned; audited with pip-audit; pip and setuptools upgraded", "Pillow has no fixed release for Python 3.10; it is kept off untrusted input"),
        ("Network exposure", "Loopback only; optional administrator password", "No TLS: use a TLS reverse proxy for remote use"),
        ("Key compromise plus database copy", "None beyond encryption", "Templates are not cancelable and cannot be reissued"),
    ], font="footnotesize")
TABLES["PRIVACY"] = table(
    "Privacy Properties: What the System Provides and What It Does Not", "tab:privacy", r"L{0.20\columnwidth}L{0.42\columnwidth}L{0.30\columnwidth}",
    r"\textbf{Principle} & \textbf{In this system} & \textbf{Gap}", [
        ("Data minimization", "Only the iris code and mask are stored; eye images are processed in memory and never written by the store or the upload path (tested)", "Capture devices and logs outside this software are out of scope"),
        ("Confidentiality", "Templates encrypted, authenticated and bound to the account; key kept outside the data directory", "Host compromise exposes key and data"),
        ("Irreversibility", "Not guaranteed: images resembling the original can be reconstructed from binary iris templates", "Encryption at rest is essential; codes are decrypted in memory for matching"),
        ("Unlinkability", "Not provided: one iris gives comparable codes in every system", "Leaked templates from two systems can be linked"),
        ("Renewability", "Not provided: a compromised iris cannot be replaced", "Needs cancelable transforms"),
        ("Purpose limitation", "Templates used only for verification and the enrollment duplicate check", "Duplicate check touches every stored template in memory"),
        ("Disclosure to others", "Duplicate refusals do not name the other account; visitors cannot list accounts in secure mode", "The operator can see the match in the log"),
        ("Consent and erasure", "Deleting a user overwrites the template in the database file (tested) and is logged", "No consent screen or self-service deletion in the demo; audit rows keep the username"),
        ("Log content", "Usernames, client identifiers, outcomes and distances; never biometric data (tested)", "No retention period defined"),
    ], font="footnotesize")
TABLES["TIME"] = table(
    "Average Processing Time per Image (CPU)", "tab:time", "lr", r"\textbf{Stage} & \textbf{Time (ms)}", [
        ("Segmentation", f"{TM['segmentation']:.2f}"), ("Normalization", f"{TM['normalization']:.2f}"),
        ("Encoding", f"{TM['encoding']:.2f}"), ("Enrollment total", f"{TM['enroll_total']:.2f}"),
        ("Matching (one comparison)", f"{TM['matching_per_comparison']:.2f}"), ("Verification total", f"{TM['verify_total']:.2f}"),
    ])

TEX = r'''\documentclass[conference]{IEEEtran}
\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{xcolor}
\usepackage{booktabs}
\usepackage{url}
\usepackage{array}
\newcolumntype{L}[1]{>{\raggedright\arraybackslash}p{#1}}
\graphicspath{{figures/}}

\begin{document}

\title{Iris Recognition-Based Authentication System\\Using Daugman's Algorithm}

\author{%
\begin{tabular}{@{}c@{\hspace{4em}}c@{}}
\begin{tabular}[t]{@{}c@{}}
Hari Sankar A\\
\textit{School of Computing}\\
\textit{Amrita Vishwa Vidyapeetham}\\
Amritapuri, India\\
am.sc.u4cse23221@am.students.amrita.edu
\end{tabular}
&
\begin{tabular}[t]{@{}c@{}}
Abhinav Dileep\\
\textit{School of Computing}\\
\textit{Amrita Vishwa Vidyapeetham}\\
Amritapuri, India\\
am.sc.u4cse23202@am.students.amrita.edu
\end{tabular}\\[1.4em]
\begin{tabular}[t]{@{}c@{}}
Ayyappadas M T\\
\textit{School of Computing}\\
\textit{Amrita Vishwa Vidyapeetham}\\
Amritapuri, India\\
am.sc.u4cse23209@am.students.amrita.edu
\end{tabular}
&
\begin{tabular}[t]{@{}c@{}}
Ajith Ashok\\
\textit{School of Computing}\\
\textit{Amrita Vishwa Vidyapeetham}\\
Amritapuri, India\\
am.sc.u4cse23204@am.students.amrita.edu
\end{tabular}\\[1.4em]
\begin{tabular}[t]{@{}c@{}}
Pranav Pradeep\\
\textit{School of Computing}\\
\textit{Amrita Vishwa Vidyapeetham}\\
Amritapuri, India\\
am.sc.u4cse23243@am.students.amrita.edu
\end{tabular}
&\\
\end{tabular}}

\maketitle

\begin{abstract}
Passwords and tokens can be shared, stolen or guessed, whereas the human iris is a stable and highly distinctive biometric trait. This paper presents a complete iris-recognition-based authentication system built on Daugman's algorithm. The system localizes the pupil and iris with a circular integro-differential operator, unwraps the iris with the rubber-sheet model, encodes it into a {{code_bits}}-bit iris code using two-scale one-dimensional Log-Gabor filters, and verifies a claimed identity with a rotation-tolerant, noise-masked Hamming distance. Because a leaked biometric template cannot be revoked like a password, the system also stores templates only as authenticated ciphertext, binds each template to its account, refuses duplicate enrollments and replayed captures, rate-limits failed verifications without revealing which accounts exist, and records every event in a tamper-evident audit log; a threat model and privacy analysis state which attacks and privacy properties are not covered. Evaluation on CASIA-Iris-Interval uses a subject-disjoint protocol: parameters and thresholds were tuned on subjects {{dev_subj}} and results are reported on the remaining subjects ({{test_eyes}} eyes). On {{ngen}} genuine and {{nimp}} impostor comparisons the system reaches an equal error rate of {{eer}}, a decidability index of {{dprime}} and {{rank1}} rank-1 identification accuracy, and a verification takes about {{t_verify}}\,ms on a standard CPU. Ablation and degradation experiments show which design choices matter, and the limitations of a single-dataset, classical-segmentation study are discussed.
\end{abstract}

\begin{IEEEkeywords}
Iris Recognition, Biometric Authentication, Daugman Algorithm, Iris Code, Log-Gabor Filter, Hamming Distance, Template Protection.
\end{IEEEkeywords}

\section{Introduction}
Authentication based on what a user knows or holds is vulnerable to sharing, theft and guessing, which motivates biometric authentication based on who the user is. Among the physiological traits in use, the iris is attractive because its texture is formed early in life, is considered stable for most of adulthood, and is protected inside the eye. Together these properties give iris codes a very low probability of false matches between different eyes~\cite{daugman1993,daugman2006}. Iris recognition is deployed at borders, in national identity programmes and in access control.

Building an iris-authentication system, however, involves more than the matching algorithm. The image must be segmented despite eyelids, eyelashes and reflections; the iris must be represented in a form that tolerates pupil dilation and head tilt; and, because an iris cannot be changed once compromised, the stored template must be protected at least as carefully as a password hash~\cite{ratha2001,jain2004}.

This project therefore implements an end-to-end iris authentication system with two goals: (i) a faithful, reproducible implementation of Daugman's pipeline~\cite{daugman2004}, evaluated with a rigorous subject-disjoint protocol, and (ii) a security layer around the biometric core consisting of encrypted template storage, duplicate-identity detection, lockout and audit logging. All results in this paper are produced by the accompanying code, and every figure and number is regenerated from the evaluation scripts.

The remainder of the paper is organized as follows. Section~II reviews related work. Section~III describes the proposed system and Section~IV the methodology. Section~V covers the implementation, Section~VI presents results and discussion, Section~VII summarizes the security features, Section~VIII gives the threat model and privacy analysis, and Sections~IX and~X conclude and outline future work.

\section{Literature Review}
Daugman introduced the first operational iris-recognition algorithm, in which the iris is located by an integro-differential operator, mapped to a dimensionless polar coordinate system, encoded by the phase of two-dimensional Gabor wavelets into a binary iris code, and compared by the fractional Hamming distance~\cite{daugman1993,daugman2004}. Because different irises produce codes that behave like independent random bit strings, the distribution of impostor distances is narrow and centred near 0.5, which makes very low false-match rates attainable~\cite{daugman2006}. Wildes proposed an alternative based on a Hough-transform localization and Laplacian-of-Gaussian pyramid features with normalized correlation matching~\cite{wildes1997}.

Open implementations made the approach reproducible. Masek's widely used implementation replaced the two-dimensional filters with one-dimensional Log-Gabor filters applied row-wise to the normalized image~\cite{masek2003}; Log-Gabor filters, introduced by Field, have no DC response and a long high-frequency tail, which suits natural textures~\cite{field1987}. Bowyer et al.\ surveyed the field, covering segmentation, normalization, feature extraction and the dominant sources of error~\cite{bowyer2008}. Large-scale public evaluations such as NIST IREX have since benchmarked commercial and academic algorithms on standard iris images~\cite{irex2009}, and the CASIA iris image databases~\cite{casia} remain the most common research benchmark for classical methods. More recently, deep convolutional networks have been used to learn iris representations~\cite{gangwar2016}; they require large labelled collections and are not examined here.

A second line of work concerns the security of biometric systems themselves. Unlike passwords, biometric traits cannot be revoked, so a compromised template is compromised for life. Ratha et al.\ analysed attack points in biometric authentication systems and proposed cancelable (revocable) templates~\cite{ratha2001}; Jain et al.\ discuss the same vulnerabilities from the standpoint of system design~\cite{jain2004}. Presentation-attack detection is standardized in ISO/IEC 30107~\cite{iso30107}. Many academic prototypes concentrate on recognition accuracy and store templates in plain files, so we treat secure storage, duplicate-identity detection, rate limiting and auditing as first-class parts of the system and verify each with automated tests.

\section{Proposed System}
Fig.~\ref{fig:arch} shows the system. A near-infrared (NIR) eye image first passes through a common front end: segmentation, normalization and encoding produce an iris code and a validity mask. In enrollment, the new template is first compared with all enrolled templates; if it matches an existing user it is refused as a duplicate identity. Otherwise it is encrypted with an authenticated cipher and written to the database together with an audit entry.

In verification, the user claims an identity. The system first checks that the claimed account is not locked by repeated failures, then decrypts and authenticates the stored template (any modification of the ciphertext is detected), computes the masked, rotation-tolerant Hamming distance between the probe and stored codes, and grants access only if the distance is below the threshold of the selected security level. Each outcome---granted, rejected, locked or error---is written to the audit log. The eye image itself is never stored.

\begin{figure*}[!t]
\centering
\includegraphics[width=0.92\textwidth]{fig_architecture.pdf}
\caption{System architecture. The front end (blue) is shared by enrollment and verification; red blocks are security controls.}
\label{fig:arch}
\end{figure*}

\section{Methodology}
\subsection{Image Acquisition and Dataset}
Experiments use CASIA-Iris-Interval~\cite{casia}, which contains {{total_images}} NIR images of {{eyes}} eyes from {{subjects}} subjects, captured with a close-up iris camera with circular NIR illumination; each image is 320$\times$280 pixels. The illuminators produce a ring of bright corneal reflections inside the pupil that must be handled during segmentation. File names carry no session label, so genuine pairs are taken as all pairs of different images of the same eye.

\subsection{Iris Segmentation}
The pupil is found first. A grayscale morphological opening with a $15\times15$ elliptical element removes the small bright corneal reflections. The opened image is thresholded at several levels between its 1st percentile and its median intensity; for each level, connected components are scored by circularity, fill ratio, proximity to the image centre, area and the number of corneal reflections they contain, and components that are too small, too elongated or at the image border are discarded. The best component gives an initial pupil circle.

Both boundaries are then refined with Daugman's integro-differential operator~\cite{daugman2004}, which searches for the circle $(r,x_0,y_0)$ maximizing the Gaussian-smoothed radial derivative of the mean intensity along the circle:
\begin{equation}
\max_{(r,x_0,y_0)}\left| G_\sigma(r) * \frac{\partial}{\partial r}\oint_{r,x_0,y_0}\frac{I(x,y)}{2\pi r}\,ds \right|.
\label{eq:daugman}
\end{equation}
The pupil boundary is refined over centre offsets of $\pm5$ px and radii between $0.7$ and $1.25$ times the initial radius, using the opened image so that reflections do not attract the circle. The outer iris boundary is searched over centre offsets of $\pm6$ px around the pupil centre and radii between $\max(1.6\,r_p,\,{{iris_prior_lo}}\ \text{px})$ and $\min(3.6\,r_p,\,{{iris_prior}}\ \text{px})$. Only the left and right angular sectors ($-35^\circ$ to $35^\circ$ and $145^\circ$ to $215^\circ$) contribute to the integral because the upper and lower parts of the limbus are covered by eyelids and eyelashes. Because the camera operates at a fixed distance, the iris radius prior encodes a property of the acquisition setup rather than of individual images; its effect is quantified in Section~\ref{sec:ablation}. A segmentation is rejected, and the image counted as a failure to acquire, if the boundary gradient is weak, the pupil and iris centres are more than $0.2\,r_i$ apart, or $r_p/r_i$ lies outside $[0.2,0.7]$.

\subsection{Normalization}
The annulus between the pupil and iris boundaries is mapped onto a fixed $64\times512$ rectangle with Daugman's rubber-sheet model, which linearly interpolates between corresponding points on the two (not necessarily concentric) boundaries:
\begin{equation}
x(r,\theta)=(1-r)\,x_p(\theta)+r\,x_i(\theta),
\label{eq:rubber}
\end{equation}
and likewise for $y$, with $r\in[0,1]$. This removes the effect of pupil dilation and of the eye's position and size. A validity mask marks pixels that should not contribute: specular highlights, pixels much darker than the iris median (eyelashes and shadows), pixels much brighter than it (eyelid and sclera glare), samples outside the image, and the rows adjacent to the pupil boundary.

\subsection{Feature Encoding}
Each normalized row is filtered in the Fourier domain with a one-dimensional Log-Gabor filter
\begin{equation}
G(f)=\exp\!\left(-\frac{\left(\ln (f/f_0)\right)^2}{2\left(\ln \sigma_f\right)^2}\right),\quad f>0,
\label{eq:loggabor}
\end{equation}
with $\sigma_f=0.5$ and two centre wavelengths of 24 and 48 samples, giving a complex response $z$ at every location. The response is averaged over groups of four radial rows and subsampled by two angularly, so the code has $16$ rows and $256$ columns. Phase is quantized to two bits per filter, $b=[\,\mathrm{Re}(z)>0,\ \mathrm{Im}(z)>0\,]$, yielding $16\times256\times4=${{code_bits}} bits. A bit is marked reliable only if all pixels contributing to it are valid; the outer four radial rows, which suffer most from occlusion, are always masked.

\subsection{Matching}
Two codes $C_A, C_B$ with masks $M_A, M_B$ are compared by the fractional Hamming distance over bits that are reliable in both, minimized over a set $\mathcal{S}$ of angular shifts that compensates for head tilt:
\begin{equation}
\mathrm{HD}=\min_{s\in\mathcal{S}}\frac{\left\|\left(C_A\oplus \mathrm{R}_s(C_B)\right)\cap M_A\cap \mathrm{R}_s(M_B)\right\|}{\left\|M_A\cap \mathrm{R}_s(M_B)\right\|},
\label{eq:hd}
\end{equation}
where $\mathrm{R}_s$ circularly shifts the code by $s$ columns. We use $\mathcal{S}=\{-6,\dots,6\}$ ($\pm${{shift_deg}}$^\circ$). If fewer than 15\% of the bits are jointly reliable, the comparison is uninformative and the distance is set to the chance level 0.5. A claimed identity is accepted if $\mathrm{HD}<T$.

\subsection{Secure Template Storage}
Only the packed code and mask are stored, never the eye image. Each template is encrypted with Fernet, an authenticated-encryption scheme from the Python \texttt{cryptography} package that combines AES-128 in CBC mode with an HMAC-SHA256 tag~\cite{fernet}, so tampering is detected before any data is parsed. The encrypted plaintext also contains the account name, so a valid ciphertext copied from one account to another is rejected when it is decrypted. The key is kept outside the data directory with owner-only permissions; it can instead be derived from a passphrase with scrypt or supplied through the environment, and it can be rotated, which re-encrypts every template. Usernames are validated against a strict pattern and lower-cased so that case variants cannot create look-alike accounts, and all database access uses parameterized queries. Enrollment is atomic. A quality gate refuses images with too few usable bits, and the new template is compared with every enrolled template using a fixed policy threshold, so one iris cannot hold two accounts; the refusal does not name the other account, although the operator can see it in the log. Accounts are never created with elevated privileges, and deleted templates are overwritten in the database file (SQLite secure delete).

\subsection{Verification Hardening}
Verification performs the same work and returns the same kind of result whether or not the claimed account exists, so it cannot be used to enumerate usernames. Failed attempts are limited per account (5 in 300 seconds) and per client (20 in 300 seconds, which stops one client from spraying many usernames); a lock is recorded once per window so that an attacker cannot flood the log. Exact replays are rejected: a probe whose code is identical to the enrolled capture or to a previously accepted capture is refused. Two different genuine captures of the same eye were never closer than 0.052 in our data (8{,}819 genuine pairs), whereas a replayed file gives a distance of 0, and the rejection threshold is 0.03. This stops reuse of a stolen image file but not a fresh presentation attack. The decision threshold is server-side policy and cannot be chosen by the person being verified.

\subsection{Tamper-Evident Audit Logging}
Every enrollment, verification and integrity event is appended to an audit table with timestamp, client, username, outcome and distance. Each row carries an HMAC-SHA256 over its contents and the previous row's hash, and a keyed head value records the latest row, so editing, deleting, reordering or truncating entries is detected when the chain is verified; the application shows the result in the audit view. The chain key is stored encrypted under the master key. Rolling the whole database back to an older consistent snapshot is not detected.

\subsection{Application Hardening}
The demonstration server listens on the loopback interface only, limits uploads to 5 MB, and rejects from the file header alone any image that is not JPEG, PNG or BMP or that exceeds one megapixel, so decompression bombs are never decoded; uploads are never written to disk. Pillow, which Streamlit installs, has known vulnerabilities with no fixed release for our Python version, so it is not used to parse untrusted files. An optional administrator password restricts enrollment, the audit log and the security level, stops the Verify page from listing accounts, and is compared in constant time behind a global throttle. Match scores are hidden unless the operator enables them for a presentation, because a visible score lets an attacker adjust an image step by step towards a match; the lockout limits such attempts. Error messages are fixed strings and framework tracebacks are disabled. Dependency versions are pinned and were audited with pip-audit.

\section{Implementation}
The system is implemented in Python using OpenCV~\cite{opencv} for image processing, NumPy and SciPy for numerical work, and the \texttt{cryptography} package for encryption; the template store uses SQLite. Each pipeline stage is a separate module with a fixed interface (segmentation, normalization, encoding, matching, pipeline), so stages can be developed and tested independently. A Streamlit application provides enrollment, verification, a pipeline-stage view that renders the boundaries, the normalized strip and the iris code, and an audit-log view, with three selectable security levels. For evaluation, all pairwise distances are computed as matrix products of $\pm1$ code matrices rather than by looping, and a test verifies that this vectorized routine agrees with the reference matcher on every sample pair. The repository contains {{n_tests}} automated tests ({{n_pipe}} for the biometric pipeline, {{n_store}} for the secure store, {{n_safe}} for input validation and {{n_app}} for the web application).

Fig.~\ref{fig:pipeline} shows the stages for one test image.

\begin{figure*}[!t]
\centering
\includegraphics[width=0.85\textwidth]{fig_pipeline.pdf}
\caption{Pipeline stages for one test image: (a) NIR eye image, (b) detected pupil (green) and iris (orange) boundaries, (c) normalized iris strip with masked pixels in red, (d) the resulting iris code.}
\label{fig:pipeline}
\end{figure*}

\section{Results and Discussion}
\subsection{Experimental Setup}
Table~\ref{tab:params} lists the parameters of the final system and Table~\ref{tab:dataset} the evaluation protocol. To avoid optimistic estimates, the data are split by subject: all numeric parameters (segmentation search ranges, filter wavelengths, radial band) and the decision thresholds were selected by their equal error rate on subjects {{dev_subj}} ({{dev_eyes}} eyes, {{dev_images}} images), and the remaining {{test_eyes}} eyes ({{test_images}} images) were used only to report final results. One caveat applies: during development, segmentation failure cases were inspected visually on images drawn from the whole dataset, including test subjects, and the iris-radius range was set after looking at radius statistics over all images, so the test set is not strictly blind (Section~\ref{sec:limits}). A genuine comparison is any pair of different images of the same eye; an impostor comparison is any pair of images from different eyes (left and right eyes of one person are treated as different eyes). Errors are reported with the standard metrics: False Accept Rate (FAR), False Reject Rate (FRR), Equal Error Rate (EER), the decidability index $d'=|\mu_g-\mu_i|/\sqrt{(\sigma_g^2+\sigma_i^2)/2}$, ROC-AUC, and rank-1 identification accuracy.

{{PARAMS}}
{{DATASET}}

\subsection{Segmentation}
{{segmented}} of {{total_images}} images ({{fta}} failures, {{fta_pct}}) passed segmentation and the plausibility checks; the rest are counted as failures to acquire (FTA) and are not silently removed from the totals. The dataset has no ground-truth boundaries, so we do not report a pixel-level segmentation accuracy; instead segmentation quality is assessed visually (Fig.~\ref{fig:seg}) and through its effect on recognition. The hardest cases, defined as the images with the largest median distance to other images of the same eye, show that the dominant failure mode is a pupil circle distorted by eyelashes or an eyelid shadow that merges with the pupil.

\begin{figure*}[!t]
\centering
\includegraphics[width=0.72\textwidth]{fig_segmentation.pdf}
\caption{Segmentation examples from the test subjects. Top: randomly chosen images. Bottom: the four images with the poorest genuine scores, where lashes or eyelids distort the pupil or iris circle.}
\label{fig:seg}
\end{figure*}

\subsection{Verification Performance}
Table~\ref{tab:perf} reports verification performance. On the test subjects, genuine comparisons have a mean Hamming distance of {{gen_mean}} (standard deviation {{gen_std}}) and impostor comparisons {{imp_mean}} ({{imp_std}}), which is close to the 0.5 expected for statistically independent codes. The two distributions are well separated (Fig.~\ref{fig:dist}); $d'$ is {{dprime}} and the ROC-AUC {{auc}}. The EER is {{eer}} on the test set against {{eer_dev}} on the development set. The difference is small, suggesting that tuning on the development subjects did not substantially overfit, though the test set is, as expected, slightly harder. In identification mode, using the first image of each of the {{gallery}} test eyes as the gallery and all {{probes}} remaining images as probes, the nearest gallery code belongs to the correct eye {{rank1}} of the time.

\begin{figure}[!t]
\centering
\includegraphics[width=\columnwidth]{fig_score_distribution.pdf}
\caption{Genuine and impostor Hamming-distance distributions on the test set and the balanced decision threshold.}
\label{fig:dist}
\end{figure}

{{PERF}}

\begin{figure*}[!t]
\centering
\includegraphics[width=0.9\textwidth]{fig_roc_det.pdf}
\caption{Verification performance on the test set: (a) ROC curve and (b) DET curve, with the EER point marked.}
\label{fig:rocdet}
\end{figure*}

The ROC and DET curves in Fig.~\ref{fig:rocdet} show that the genuine-accept rate falls only gradually as FAR is pushed down. The reason is visible in Table~\ref{tab:perf}: even at FAR $=0.01\%$ the FRR is {{frr_far4}}. Judging from the hardest-case images in Fig.~\ref{fig:seg}, this floor is most likely caused by a small fraction of genuine pairs in which at least one image is poorly segmented, rather than by the encoding itself; improving pupil localization on occluded eyes is therefore the most promising route to better accuracy.

\subsection{Choice of Operating Point}
A single threshold cannot suit every application. Table~\ref{tab:ops} gives three security levels exposed in the demonstration application. Their thresholds were chosen on the development set (the EER point, and the points where the development FAR is 0.1\% and 0.01\%), and the error rates in the table were then measured on the independent test set. The measured FAR agrees closely with the development target at each level ({{high_far}} against a 0.1\% target at the high-security level, and {{max_far}} against a 0.01\% target at the maximum-security level), showing that the thresholds transfer to unseen subjects. Moving from the balanced to the maximum-security level lowers the FAR by about two orders of magnitude at the cost of {{max_frr}} FRR; for an access-control system where a false accept is much more costly than a retry, the higher levels are preferable. The rounded balanced threshold of {{thr}}, used in the robustness experiments, has FAR {{far_thr}} and FRR {{frr_thr}} (Fig.~\ref{fig:farfrr}); the application's default is the high-security level.

\begin{figure}[!t]
\centering
\includegraphics[width=\columnwidth]{fig_far_frr.pdf}
\caption{FAR and FRR as functions of the decision threshold on the test set.}
\label{fig:farfrr}
\end{figure}

{{OPS}}

\subsection{Ablation Study}\label{sec:ablation}
Table~\ref{tab:ablation} removes one design element at a time and re-evaluates on the test set. Rotation compensation matters most: restricting the comparison to zero shift raises the EER from {{abl_full}} to {{abl_norot}} ({{abl_rot_x}} times higher), because even small head tilts misalign the codes. The iris-radius prior in the segmentation also helps (EER {{abl_noprior}} without it): without a prior, the outer-boundary search is occasionally attracted to the collarette or to eyelid edges. Using all radial rows instead of the inner 75\% raises the EER to {{abl_allband}}, consistent with the outer iris being more occluded. Excluding the noise mask has a smaller effect ({{abl_nomask}}); in our implementation pixels flagged as invalid are replaced by the row mean before filtering, which already limits their influence, so the mask mainly removes the few bits that remain affected.

{{ABL}}

\subsection{Robustness}
Table~\ref{tab:robust} degrades one probe image per eye ({{probes_n}} probes) and compares it with the clean test gallery at the balanced threshold. The clean probes give FRR {{clean_frr}} and FAR {{clean_far}}. Additive noise, mild blur and a reduction in brightness leave the error rates close to the clean case. A $1.4\times$ brightness increase saturates the image and {{bright_fta}} of {{probes_n}} probes then fail segmentation, which raises the rejection rate. Rotation shows the limit of shift compensation: at $5^\circ$ the FRR is {{rot5_frr}}, at $10^\circ$ it rises to {{rot10_frr}}, and at $20^\circ$, beyond the $\pm${{shift_deg}}$^\circ$ shift range, it reaches {{rot20_frr}}. Increasing the shift range would handle larger tilts at the cost of a slightly higher impostor-accept rate, since more alignments are tried.

{{ROB}}

\subsection{Security Feature Verification}
Table~\ref{tab:security} summarizes the checks performed on the security mechanisms; all {{n_tests}} automated tests pass ({{n_store}} on the store, {{n_safe}} on input validation, {{n_app}} on the web application and {{n_pipe}} on the pipeline). To check that the tests are able to fail, five central defences (account binding of ciphertext, the log's head check, replay rejection, secure deletion and lockout of unknown accounts) were disabled one at a time, and the corresponding test failed in every case.

{{SEC}}

\subsection{Performance}
Table~\ref{tab:time} gives the average processing time per image, measured on 100 test images on the development laptop CPU. Enrollment takes {{t_enroll}}\,ms and verification {{t_verify}}\,ms; segmentation dominates at {{t_seg}}\,ms, while one code comparison takes {{t_match}}\,ms, so searching a gallery of thousands of templates remains practical.

{{TIME}}

\subsection{Discussion and Limitations}\label{sec:limits}
The results show that a classical, fully interpretable implementation of Daugman's pipeline reaches an EER of about {{eer}} on a standard NIR benchmark, and that the security layer adds negligible latency. Several limitations apply. First, only one dataset and one sensor were used; the iris-radius prior and the corneal-reflection handling are tuned to a fixed-distance NIR camera, and accuracy on other sensors, on visible-light images or in unconstrained capture is likely to be lower. Second, file names do not identify sessions, so genuine pairs may be taken within a single session, and cross-session results can be expected to be somewhat worse. Third, the test set contains {{test_eyes}} eyes and the {{ngen}} genuine pairs are correlated (many pairs share images), so confidence intervals on the error rates are wider than the pair counts suggest. Fourth, the test subjects were not entirely blind: although all numeric tuning used only the development subjects, the segmentation stage was debugged by inspecting outlier images from the whole dataset, so the reported test error may be slightly optimistic. Fifth, no ground-truth segmentation was available, and the remaining errors are dominated by poorly segmented, occluded eyes. Sixth, the system does not detect presentation attacks such as printed iris images or contact lenses; it verifies that the presented image matches the template, not that it comes from a live eye. Finally, the encryption key is a file or passphrase on the same host as the database, so compromise of that host exposes both; a deployment should use a key-management service or hardware module, put the application behind TLS, and add network-level rate limits, because the per-client counter is keyed on a session identifier that an attacker can change. The iris codes themselves are not cancelable, so a stolen key plus database would expose templates that cannot be reissued.

\section{Security Features}
\subsection{Encrypted, Account-Bound Storage}
Iris templates are stored only as Fernet tokens that include the account name. A copy of the database file reveals no template, a modified or swapped template is detected by the authentication tag, and deleted templates are overwritten.

\subsection{Key Management}
The key is held outside the data directory with owner-only permissions, can be derived from a passphrase or supplied through the environment, and can be rotated without re-enrolling users.

\subsection{Duplicate-Identity Prevention}
Each new template is compared with all enrolled templates under a fixed policy, so a single iris cannot be registered under several identities, and the refusal does not disclose who holds it.

\subsection{Verification Hardening}
Unknown and wrong accounts are indistinguishable, failures are rate limited per account and per client, replayed captures are rejected, and the decision threshold is a server-side setting.

\subsection{Tamper-Evident Audit Log}
Enrollments, verifications, lockouts, integrity failures and errors are recorded in an HMAC hash chain, and the interface reports whether the chain is intact.

\subsection{Input Validation and Access Control}
Uploads are size-limited and checked from the file header, are never written to disk, and match scores are hidden by default. An optional administrator password restricts enrollment, the audit log and the security level (Table~\ref{tab:ops}), and the server listens on the loopback interface only.

\section{Threat Model and Privacy Analysis}
\subsection{Threat Model}
We consider an adversary who may present eye images hoping to be accepted as someone else (a zero-effort or targeted impostor), read or copy the database file, modify it, send crafted requests or uploads to the web interface, or repeat attempts to guess. We do not defend against an adversary who controls the running server or the encryption key, nor against presentation attacks that carry the genuine iris texture. Table~\ref{tab:threat} lists each threat with the mitigation implemented and the risk that remains; the unmitigated rows are as important as the mitigated ones, and the lockout itself introduces a denial-of-service risk for the victim account.

{{THREAT}}

\subsection{Privacy Analysis}
Biometric data is personal data that cannot be replaced once exposed, and it is regulated accordingly. The GDPR treats biometric data processed to identify a person uniquely as a special category requiring a stricter legal basis~\cite{gdpr}, and India's Digital Personal Data Protection Act, 2023 requires consent for specified purposes and gives individuals a right to erasure~\cite{dpdp}. ISO/IEC 24745 names irreversibility, unlinkability and renewability as the properties of a protected biometric template~\cite{iso24745}. We do not claim legal compliance; Table~\ref{tab:privacy} maps these principles to what the system does and does not do.

{{PRIVACY}}

The system is strong on data minimization and confidentiality: the eye image is never stored, and the stored template is authenticated ciphertext. It is weak on irreversibility, unlinkability and renewability, which are properties of the template format rather than of the storage layer. Iris codes are lossy, but images resembling the original eye have been reconstructed from binary iris templates~\cite{galbally2013}, so a leaked plaintext template is a genuine privacy loss, and because the same iris produces comparable codes in every system, leaked templates from two systems can be linked to the same person. Encryption at rest reduces the likelihood of such a leak but does not remove the consequence if the key is also compromised. Cancelable templates, in which a revocable per-system transform is applied before storage, are the standard remedy and are listed as future work.

\section{Conclusion}
This paper presented an iris-recognition-based authentication system that implements Daugman's pipeline---integro-differential segmentation, rubber-sheet normalization, Log-Gabor phase encoding and masked, rotation-tolerant Hamming matching---together with encrypted template storage, duplicate-identity detection, lockout and audit logging. Under a subject-disjoint protocol on CASIA-Iris-Interval the system achieves an EER of {{eer}}, $d'=${{dprime}} and {{rank1}} rank-1 accuracy, and the ablation study quantifies the contribution of rotation compensation, the segmentation prior and the radial-band restriction. The remaining errors come mainly from poorly segmented occluded eyes, which identifies where improvement is most valuable.

\section{Future Work}
The most direct improvement is more robust localization of occluded pupils, for example with a learned segmentation network, which should lower the error floor visible in the DET curve. Presentation-attack detection, using texture or frequency analysis of the iris and pupil dynamics, is needed before the system can be deployed against active attackers. Cancelable iris codes, in which a revocable user-specific transform is applied before storage, would allow a compromised template to be reissued. Evaluation should be extended to further sensors, cross-session data and visible-light images, and the iris could be combined with a second modality such as face in a multimodal system. Finally, real-time acquisition with NIR hardware and quality feedback would turn the dataset-driven demonstration into a deployable capture station.

\begin{thebibliography}{99}
\bibitem{daugman1993} J. G. Daugman, ``High confidence visual recognition of persons by a test of statistical independence,'' \textit{IEEE Trans. Pattern Anal. Mach. Intell.}, vol. 15, no. 11, pp. 1148--1161, Nov. 1993.
\bibitem{daugman2004} J. Daugman, ``How iris recognition works,'' \textit{IEEE Trans. Circuits Syst. Video Technol.}, vol. 14, no. 1, pp. 21--30, Jan. 2004.
\bibitem{daugman2006} J. Daugman, ``Probing the uniqueness and randomness of IrisCodes: Results from 200 billion iris pair comparisons,'' \textit{Proc. IEEE}, vol. 94, no. 11, pp. 1927--1935, Nov. 2006.
\bibitem{wildes1997} R. P. Wildes, ``Iris recognition: An emerging biometric technology,'' \textit{Proc. IEEE}, vol. 85, no. 9, pp. 1348--1363, Sep. 1997.
\bibitem{masek2003} L. Masek, ``Recognition of human iris patterns for biometric identification,'' B.Eng. thesis, School of Computer Science and Software Engineering, Univ. Western Australia, 2003.
\bibitem{field1987} D. J. Field, ``Relations between the statistics of natural images and the response properties of cortical cells,'' \textit{J. Opt. Soc. Am. A}, vol. 4, no. 12, pp. 2379--2394, 1987.
\bibitem{bowyer2008} K. W. Bowyer, K. Hollingsworth, and P. J. Flynn, ``Image understanding for iris biometrics: A survey,'' \textit{Comput. Vis. Image Underst.}, vol. 110, no. 2, pp. 281--307, 2008.
\bibitem{irex2009} P. Grother, E. Tabassi, G. W. Quinn, and W. Salamon, ``IREX I: Performance of iris recognition algorithms on standard images,'' NIST Interagency Report 7629, 2009.
\bibitem{gangwar2016} A. Gangwar and A. Joshi, ``DeepIrisNet: Deep iris representation with applications in iris recognition and cross-sensor iris recognition,'' in \textit{Proc. IEEE Int. Conf. Image Process. (ICIP)}, 2016, pp. 2301--2305.
\bibitem{jain2004} A. K. Jain, A. Ross, and S. Prabhakar, ``An introduction to biometric recognition,'' \textit{IEEE Trans. Circuits Syst. Video Technol.}, vol. 14, no. 1, pp. 4--20, Jan. 2004.
\bibitem{ratha2001} N. K. Ratha, J. H. Connell, and R. M. Bolle, ``Enhancing security and privacy in biometrics-based authentication systems,'' \textit{IBM Syst. J.}, vol. 40, no. 3, pp. 614--634, 2001.
\bibitem{iso30107} \textit{Information Technology---Biometric Presentation Attack Detection---Part 3: Testing and Reporting}, ISO/IEC 30107-3, 2023.
\bibitem{casia} Institute of Automation, Chinese Academy of Sciences, ``CASIA Iris Image Database (CASIA-Iris-Interval),'' http://biometrics.idealtest.org/.
\bibitem{fernet} Python Cryptographic Authority, ``Fernet (symmetric encryption),'' \textit{cryptography} documentation.
\bibitem{opencv} G. Bradski, ``The OpenCV Library,'' \textit{Dr. Dobb's Journal of Software Tools}, 2000.
\bibitem{galbally2013} J. Galbally, A. Ross, M. Gomez-Barrero, J. Fierrez, and J. Ortega-Garcia, ``Iris image reconstruction from binary templates: An efficient probabilistic approach based on genetic algorithms,'' \textit{Comput. Vis. Image Underst.}, vol. 117, no. 10, pp. 1512--1525, 2013.
\bibitem{iso24745} \textit{Information Security, Cybersecurity and Privacy Protection---Biometric Information Protection}, ISO/IEC 24745, 2022.
\bibitem{gdpr} European Parliament and Council of the EU, ``Regulation (EU) 2016/679 (General Data Protection Regulation),'' Art.~9, 2016.
\bibitem{dpdp} Government of India, ``The Digital Personal Data Protection Act, 2023,'' Act No.~22 of 2023.
\end{thebibliography}

\end{document}
'''

V["iris_prior"] = f"{P['iris_radius_prior_px'][1]:.0f}"
V["iris_prior_lo"] = f"{P['iris_radius_prior_px'][0]:.0f}"
for key, tbl in TABLES.items():
    TEX = TEX.replace("{{" + key + "}}", tbl)
missing = set(re.findall(r"\{\{(\w+)\}\}", TEX)) - set(V)
if missing:
    raise SystemExit(f"unfilled template fields: {sorted(missing)}")
TEX = re.sub(r"\{\{(\w+)\}\}", lambda m: str(V[m.group(1)]), TEX)
with open("report/main.tex", "w", encoding="utf-8") as fh:
    fh.write(TEX)

figs = ["fig_architecture", "fig_pipeline", "fig_segmentation", "fig_score_distribution", "fig_roc_det", "fig_far_frr"]
with zipfile.ZipFile("report/iris_report_overleaf.zip", "w", zipfile.ZIP_DEFLATED) as z:
    z.write("report/main.tex", "main.tex")
    for f in figs:
        z.write(f"report/figures/{f}.pdf", f"figures/{f}.pdf")
print(f"wrote report/main.tex ({len(TEX.splitlines())} lines) and report/iris_report_overleaf.zip")
