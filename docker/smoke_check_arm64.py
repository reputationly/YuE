"""Import-level validation for the ARM64 / A100 YuE2 container.

Runs at image build time (no GPU there), so it checks what a GPU-less build can
check: pinned versions, a CUDA-built torch, and that every module on the
serving path imports under this exact dependency set — the model and VAE code
too, since they are the ones written against transformers 4.x APIs.
"""
import importlib.metadata

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

    # SheetSage2's third-party imports (its model code lives with the weights).
    import mido  # noqa: F401
    import mir_eval.chord  # noqa: F401
    import pretty_midi  # noqa: F401
    import torchaudio.transforms  # noqa: F401

    import yue2.modeling_vae  # noqa: F401
    import yue2.modeling_yue2  # noqa: F401
    import yue2.nar  # noqa: F401
    import yue2.pipeline  # noqa: F401
    from yue2 import YuE2Pipeline  # noqa: F401

    # The serving entrypoint itself, not just the model modules: server.py
    # builds the FastAPI app and the queue at import time but loads weights only
    # in its lifespan hook, so importing it here is side-effect free. (Breeze's
    # first ARM64 image passed a model-only check and then died at container
    # start on an import only server.py pulled in.)
    import server  # noqa: F401

    print("YuE2 container smoke check passed:", versions, "torch", torch.__version__, "cuda", torch.version.cuda)


if __name__ == "__main__":
    main()
