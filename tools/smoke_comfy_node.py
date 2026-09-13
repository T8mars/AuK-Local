from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def load_node_module(comfy_root: Path):
    comfy_root_string = str(comfy_root)
    if comfy_root_string not in sys.path:
        sys.path.insert(0, comfy_root_string)
    node_init = comfy_root / "custom_nodes" / "ComfyUI-AuK-Local" / "__init__.py"
    module_name = "comfyui_auk_local_smoke"
    spec = importlib.util.spec_from_file_location(
        module_name,
        node_init,
        submodule_search_locations=[str(node_init.parent)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载节点：{node_init}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.nodes


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AuK Local through its installed ComfyUI node")
    parser.add_argument("--comfy-root", type=Path, required=True)
    parser.add_argument("--service-url", default="http://127.0.0.1:7860")
    args = parser.parse_args()
    nodes = load_node_module(args.comfy_root.resolve())
    connection = nodes.AuKLocalConnection.execute(
        args.service_url,
        "",
        "flash",
        cpu_offload=True,
        keep_loaded=False,
    )[0]
    result = nodes.AuKLocalGenerateEdit.execute(
        connection,
        "描述生成语音",
        "AuK ComfyUI bridge is ready.",
        "A calm and clear young woman",
        1.0,
        20260913,
    )
    audio, instruction, metadata_json = result.result
    waveform = audio["waveform"]
    summary = {
        "shape": list(waveform.shape),
        "sample_rate": audio["sample_rate"],
        "finite": bool(waveform.isfinite().all()),
        "peak": float(waveform.abs().max()),
        "instruction": instruction,
        "metadata": json.loads(metadata_json),
    }
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
