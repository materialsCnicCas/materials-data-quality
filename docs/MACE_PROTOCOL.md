# MACE subset protocol

The MACE experiment used Python 3.10.20, PyTorch 2.11.0 with CUDA 12.8, MACE 0.3.15, ASE 3.29.0, and two NVIDIA RTX 5090 GPUs. The full model files are not included in this repository.

`experiments/mace_inputs.tar.gz` contains the frozen 4,000-record group-split inputs. `experiments/mace_evidence/` contains the recorded training commands, per-run metrics, predictions, retention reports, environment metadata, and collection manifest.

The local workflow is:

1. Install Python 3.10 and the packages in `requirements-mace.txt`.
2. Export or restore the MACE inputs from `experiments/mace_inputs.tar.gz`.
3. Run `experiments/scripts/remote_mace.py` on a CUDA host.
4. Run `experiments/scripts/remote_collect.py` to aggregate the run logs and predictions.

Training uses a common ten-epoch budget for the reported subset experiments. Models are selected from a separate validation split; the common clean test crystals are not used for training or threshold selection. See `experiments/results/mace_protocol_checks.json` for the recorded checks.
