# Release validation

Validation date: 2026-11-17.

| Check | Outcome |
|---|---|
| CPU unit/integration tests | 18 passed |
| CLI workflow on a small synthetic model | Passed, including CSV + JSON output |
| Paper Tables 1–2 | All 144 SRC/deviation entries and 72 repeated radius entries checked against the PDF text |
| Original supplementary script, original results CSV, and both PDFs | Byte-identical to supplied files |
| Local Markdown links and Python syntax | Checked |
| README layout | Inspected through a static HTML/PDF rendering; banner and method figure inspected |
| Learned DBCNN inference | Not run: trained FS-IQA heads were not supplied |
| Reproduction of paper comparisons, attacks, and A100 timing | Not run: full experiment code, trained models, and original splits were not supplied |

Tests ran on CPU using Python 3.12.14, PyTorch 2.5.1+cpu,
torchvision 0.20.1+cpu, NumPy 2.5.4, SciPy 1.17.0, pandas 2.2.3,
OpenCV 4.11.0, Pillow 13.0.0, and tqdm 4.70.1.

The tests use synthetic inputs and a tiny integration-test model. No random
model is presented as a trained FS-IQA checkpoint, and no test result is
presented as a reproduction of a paper result.
