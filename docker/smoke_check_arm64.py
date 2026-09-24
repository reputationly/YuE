"""Import-level validation for the ARM64 / A100 YuE2 container.

Runs at image build time (no GPU there), so it checks what a GPU-less build can
check, for both interpreters the engine uses:

- the main process (this interpreter, /opt/venv): pinned transformers 4.x, a
  CUDA-built torch, the model/VAE/pipeline/scheduler modules and the serving
  entrypoint itself;
- the vLLM worker (/usr/bin/python3 via YUE2_VLLM_PYTHON): vLLM importable next
  to yue2.fast, which is what the worker process runs.
"""
import importlib.metadata
import os
import subprocess

import torch

EXPECTED = {"transformers": "4.57.6", "huggingface-hub": "0.36.2", "tiktoken": "0.12.0"}


def main() -> None:
    versions = {name: importlib.metadata.version(name) for name in EXPECTED}
    for name, want in EXPECTED.items():
        if versions[name].split("+")[0] != want:
            raise RuntimeError(f"{name}: expected {want}, got {versions[name]}")
    if not torch.version.cuda:
        raise RuntimeError(
            f"torch {torch.__version__} is not a CUDA build (probably replaced by PyPI's CPU aarch64 wheel)"
        )

    import yue2.modeling_vae  # noqa: F401
    import yue2.modeling_yue2  # noqa: F401
    import yue2.nar  # noqa: F401
    import yue2.pipeline  # noqa: F401
    import yue2.service  # noqa: F401
    # SheetSage2's third-party imports (its model code lives with the weights).
    import mido  # noqa: F401
    import mir_eval.chord  # noqa: F401
    import pretty_midi  # noqa: F401

    # The serving entrypoint itself, not just the model modules: server.py builds
    # the FastAPI app at import time but starts the worker only in its lifespan,
    # so importing it here is side-effect free. (Breeze's first ARM64 image passed
    # a model-only check and then died at container start on an import only
    # server.py pulled in.)
    import server  # noqa: F401

    worker_python = os.environ["YUE2_VLLM_PYTHON"]
    probe = subprocess.run(
        [worker_python, "-c",
         "import vllm, transformers, yue2.fast; print(vllm.__version__, transformers.__version__)"],
        capture_output=True, text=True,
    )
    if probe.returncode != 0:
        raise RuntimeError(f"vLLM worker interpreter {worker_python} cannot import vllm + yue2.fast:\n{probe.stderr}")
    vllm_version, worker_transformers = probe.stdout.split()

    print("YuE2 container smoke check passed:", versions, "torch", torch.__version__, "cuda", torch.version.cuda,
          "| vLLM worker:", worker_python, "vllm", vllm_version, "transformers", worker_transformers)


if __name__ == "__main__":
    main()
