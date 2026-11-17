# Model checkpoints

The supplementary archive did not include model weights. Inference requires all
three files below; analyzing the bundled results CSV does not require them.

| File | Purpose | Availability |
|---|---|---|
| `DBCNN_KonIQ10k-2de81c0a.pth` | Full pretrained DBCNN, including the SCNN branch | Upstream IQA-PyTorch weights |
| `dbcnn_prevscorer_FS_01.pth` | Trained DBCNN FTN for σf = 0.1 | Author checkpoint required; absent from supplied archive |
| `dbcnn_scorer_FS_01.pth` | Trained scorer for σf = 0.1 | Author checkpoint required; absent from supplied archive |

The upstream backbone URL referenced in the supplied code is:

<https://huggingface.co/chaofengc/IQA-PyTorch-Weights/resolve/main/DBCNN_KonIQ10k-2de81c0a.pth>

The upstream [IQA-PyTorch repository](https://github.com/chaofengc/IQA-PyTorch)
provides the backbone implementation and weight distribution. Downloading the
backbone does **not** provide the trained FS-IQA heads. No substitute random
heads are used by this repository.

Place the three files in this directory, or set `--backbone-checkpoint`,
`--ftn-checkpoint`, and `--scorer-checkpoint` to their actual paths.
Use heads trained for the selected noise level; setting `--sigma 1.0` does not
turn σf = 0.1 checkpoints into the strong-setting model from the paper.

The cleaned loader accepts raw state dictionaries and dictionaries wrapped under
`params`, `state_dict`, or `model_state_dict`; a `module.` prefix is removed.
All keys and tensor shapes must match. Files are loaded with `weights_only=True`.

Unlike the historical script, the cleaned loader does not need a separate SCNN
file or a VGG download: the **full** DBCNN checkpoint supplies both feature
branches. The historical script additionally expects `DBCNN_scnn-7ea73d75.pth`.

## Memory

The provided DBCNN FTN has layers 65,536 → 32,768 → 16,384 → 512, roughly
2.69 billion parameters (about 10.03 GiB of float32 weights). This architecture
is retained for checkpoint compatibility. The exact 512-output Jacobian on a
512 × 384 RGB image alone occupies about 1.13 GiB, before activations and
linear-algebra workspace. A high-memory GPU and adequate host RAM are needed;
the paper's experiments used NVIDIA A100 GPUs. CPU execution is supported by
the interface but full DBCNN inference and IVE can be very slow.
