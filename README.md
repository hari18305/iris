# Iris Recognition-Based Authentication

An iris authentication system built on Daugman's algorithm, wrapped in a hardened template store and evaluated on CASIA-Iris-Interval with a subject-disjoint protocol (parameters tuned on subjects 001-060, results reported on 061-249).

| Held-out result | Value |
|---|---|
| Equal error rate | 1.94% |
| Decidability d' | 4.24 |
| Rank-1 identification | 96.5% |
| Verification time | about 15 ms |

## What is here

| Folder | Contents |
|---|---|
| `src/` | Pipeline (segmentation, normalization, encoding, matching), input validation, and the encrypted template store |
| `demo/` | Streamlit app: enroll, verify, pipeline view, audit log |
| `eval/` | Metrics and the full evaluation (ablations, robustness, timing, figures) |
| `tests/` | 79 automated tests for the pipeline, store, input validation and web app |
| `report/` | IEEE-format paper source, figures, results and the presentation |
| `data/` | See `data/README.md` |

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run the demo

Run from the project root. The server listens on `127.0.0.1` only.

```
$env:IRIS_DEMO_SCORES="1"        # show Hamming distances (hidden by default)
streamlit run demo/app.py
```

| Variable | Effect |
|---|---|
| `IRIS_DEMO_SCORES=1` | Show distances. Off by default because visible scores enable hill-climbing attacks. |
| `IRIS_ADMIN_PASSWORD` | Restrict enrollment, the audit log and the security level to an administrator. |
| `IRIS_AUTH_KEY` / `IRIS_AUTH_PASSPHRASE` | Template-encryption key source (default: a key file under `~/.iris_auth/`). |
| `IRIS_DEMO_DB` | Database path (default `data/store/demo.db`). |

## Tests

```
pytest
```

The tests need ten CASIA sample images in `data/samples/` (see `data/README.md` for the file names). The images are not redistributed here because of the dataset licence.

## Reproduce the results

Needs the CASIA-Iris-Interval dataset under `data/raw/` (see `data/README.md`).

```
python eval/build_cache.py         # segment and normalize every image once
python eval/run_evaluation.py      # dev/test protocol, ablations, robustness, timing, figures
python eval/make_architecture.py   # architecture figure
python report/build_report.py      # report/main.tex and the Overleaf zip, filled from the results
```

## Security properties and limits

Encrypted, account-bound templates; a tamper-evident (hash-chained) audit log; rate limiting without account enumeration; replay rejection; validated uploads that are never written to disk. Each has a test.

Not covered: presentation-attack (liveness) detection, a compromised host, and revocable templates. The paper lists these as open risks.

## Dataset

CASIA-Iris-Interval (Institute of Automation, Chinese Academy of Sciences) must be obtained separately under its own licence.
