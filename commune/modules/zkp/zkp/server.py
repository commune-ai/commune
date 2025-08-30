#!/usr/bin/env python3
# zkpy_playground.py
# A tiny playground to prove a (restricted) Python function with ezkl
# using a PyTorch->ONNX export. Includes a mock mode for quick iteration.

import os, json, subprocess, sys, dataclasses
from typing import Callable, List, Dict, Any, Optional

# --- Optional Torch imports (only needed for real ONNX export / mock tensor eval) ---
try:
    import torch
    import torch.onnx as to
except Exception:
    torch = None
    to = None


# ================================
#  Torch wrapper for your function
# ================================
class TorchWrapperModel(torch.nn.Module if torch else object):
    """
    Wrap a python function that operates on a torch.Tensor and returns a torch.Tensor.
    """
    def __init__(self, fn: Callable):
        if torch is None:
            raise RuntimeError("torch not available; install torch to export ONNX")
        super().__init__()
        self._fn = fn

    def forward(self, x):
        return self._fn(x)


# ================================
#  ezkl Orchestrator
# ================================
@dataclasses.dataclass
class ZKPyFunctionEZKL:
    pyfunc: Callable                 # function: torch.Tensor -> torch.Tensor
    input_shape: List[int]           # e.g. [1, 3]
    workdir: str = "zk_artifacts"
    model_path: str = "model.onnx"
    settings_path: str = "settings.json"
    compiled_path: str = "circuit.ezkl"
    srs_path: str = "kzg.srs"
    input_path: str = "input.json"
    witness_path: str = "witness.json"
    proof_path: str = "proof"
    vk_path: str = "vk.key"

    # ---------- Helpers ----------
    def _abs(self, rel: str) -> str:
        p = os.path.join(self.workdir, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p

    def _check_ezkl(self) -> None:
        try:
            subprocess.check_output(["ezkl", "--help"])
        except Exception:
            raise RuntimeError(
                "The `ezkl` CLI is not available on PATH.\n"
                "Install it (see ezkl docs) or use --mock to iterate first."
            )

    def _sh(self, *args: str):
        print("$", " ".join(args))
        subprocess.check_call(list(args), cwd=self.workdir)

    # ---------- Mock (no ezkl needed) ----------
    def mock_prove(self, x_flat: List[float]) -> Dict[str, Any]:
        """
        Quickly test your function locally. Returns {'ok': True, 'output': ...}
        """
        if torch is None:
            # Very light fallback: only supports scalar functions, not real tensors
            try:
                y = [self.pyfunc(v) for v in x_flat]  # if fn supports scalars
                return {"ok": True, "output": y, "note": "mock (no torch installed)"}
            except Exception as e:
                return {"ok": False, "error": f"mock failed: {e}"}
        else:
            with torch.no_grad():
                t = torch.tensor(x_flat, dtype=torch.float32).reshape(self.input_shape)
                model = TorchWrapperModel(self.pyfunc).eval()
                y = model(t).detach().cpu().numpy().tolist()
                return {"ok": True, "output": y, "note": "mock (local tensor eval)"}

    # ---------- Real pipeline (ezkl) ----------
    def export_onnx(self, opset: int = 17) -> str:
        if torch is None or to is None:
            raise RuntimeError("Need torch to export ONNX: `pip install torch`")
        model = TorchWrapperModel(self.pyfunc).eval()
        dummy = torch.zeros(self.input_shape, dtype=torch.float32)
        out = self._abs(self.model_path)
        to.export(
            model, (dummy,), out,
            opset_version=opset, input_names=["x"], output_names=["y"],
            dynamic_axes=None
        )
        return out

    def write_settings(self, scale: int = 16, logrows: int = 17) -> str:
        """
        Basic ezkl settings: adjust scale (fixed-point precision) and logrows (circuit size).
        """
        settings = {
            "run_args": {"input_visibility": "private", "output_visibility": "public"},
            "model_output_scales": [scale],
            "visibility": {"input": "private", "params": "private", "output": "public"},
            "logrows": logrows
        }
        out = self._abs(self.settings_path)
        with open(out, "w") as f:
            json.dump(settings, f, indent=2)
        return out

    def gen_input(self, x_flat: List[float]) -> str:
        """
        Save input.json for ezkl. Expect a flat list matching product(input_shape).
        """
        data = {"input_data": [x_flat]}
        out = self._abs(self.input_path)
        with open(out, "w") as f:
            json.dump(data, f)
        return out

    def compile(self):
        self._check_ezkl()
        self._sh("ezkl", "compile",
                 "-M", self.model_path,
                 "-S", self.settings_path,
                 "-O", self.compiled_path)

    def setup(self):
        self._check_ezkl()
        self._sh("ezkl", "setup",
                 "-M", self.model_path,
                 "-S", self.settings_path,
                 "--compiled-circuit", self.compiled_path,
                 "-D", self.srs_path,
                 "--vk-path", self.vk_path)

    def gen_witness(self):
        self._check_ezkl()
        self._sh("ezkl", "gen-witness",
                 "-M", self.model_path,
                 "-S", self.settings_path,
                 "-I", self.input_path,
                 "-O", self.witness_path)

    def prove(self, proof_type: str = "single"):
        self._check_ezkl()
        self._sh("ezkl", "prove",
                 "-M", self.model_path,
                 "--witness", self.witness_path,
                 "--compiled-circuit", self.compiled_path,
                 "--vk-path", self.vk_path,
                 "--proof-path", self.proof_path,
                 "--proof-type", proof_type,
                 "-S", self.settings_path)

    def verify(self) -> bool:
        self._check_ezkl()
        try:
            self._sh("ezkl", "verify",
                     "--proof-path", self.proof_path,
                     "--vk-path", self.vk_path)
            return True
        except subprocess.CalledProcessError:
            return False


    # ================================
    #  Example function (edit me)
    # ================================
    @staticmethod
    def example_poly(x):
        """
        Elementwise polynomial:
        y = x^2 + 3x + 5
        Must be Torch ops when used with real pipeline.
        """
        # Works both for torch.Tensor and Python scalars (for mock fallback)
        return x * x + 3 * x + 5


    # ================================
    #  CLI
    # ================================
    @staticmethod
    def main():
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("--mock", action="store_true", help="Run mock eval (no ezkl required)")
        p.add_argument("--shape", type=str, default="1,3", help="Input shape, e.g. 1,3")
        p.add_argument("--input", type=str, default="1.0,2.0,3.5", help="Flat input values")
        p.add_argument("--workdir", type=str, default="zk_artifacts", help="Artifacts dir")
        args = p.parse_args()

        input_shape = [int(s) for s in args.shape.split(",") if s]
        x_flat = [float(s) for s in args.input.split(",") if s]

        zk = ZKPyFunctionEZKL(example_poly, input_shape=input_shape, workdir=args.workdir)
        os.makedirs(zk.workdir, exist_ok=True)

        if args.mock:
            print(zk.mock_prove(x_flat))
            return

        # Real pipeline
        zk.export_onnx()
        zk.write_settings(scale=16, logrows=17)
        zk.gen_input(x_flat)
        zk.compile()
        zk.setup()
        zk.gen_witness()
        zk.prove(proof_type="single")
        ok = zk.verify()
        print({"verified": ok})


if __name__ == "__main__":
   ZKPyFunctionEZKL.main()
