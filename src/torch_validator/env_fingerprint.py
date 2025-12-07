"""
Environment fingerprint utility for capturing system configuration.

Used to identify differences between hosts that may cause deterministic
divergence in distributed training.
"""

import warnings

# Suppress import order warning when running as module (-m torch_validator.env_fingerprint)
warnings.filterwarnings("ignore", message="found in sys.modules")

import json
import os
import platform
import subprocess
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    """Information about a single GPU."""
    index: int
    uuid: str
    name: str
    vbios_version: str
    driver_version: str
    memory_total: str
    pcie_bus_id: str
    serial: str = ""
    board_part_number: str = ""
    inforom_version: str = ""


@dataclass
class CUDAInfo:
    """CUDA-related information."""
    cuda_version: str
    cudnn_version: str
    nccl_version: str
    cuda_home: str
    cuda_visible_devices: str


@dataclass
class PythonInfo:
    """Python environment information."""
    version: str
    torch_version: str
    torch_cuda_version: str
    torch_cudnn_version: str
    torch_nccl_version: str
    torch_compile_backend: str
    triton_version: str


@dataclass
class SystemInfo:
    """System-level information."""
    hostname: str
    kernel_version: str
    os_release: str
    cpu_model: str
    cpu_count: int
    memory_total_gb: float
    numa_nodes: int
    instance_id: str = ""  # AWS EC2 instance ID from board_asset_tag


@dataclass
class NetworkInfo:
    """Network/interconnect information."""
    nccl_net: str
    nccl_socket_family: str
    efa_provider: str
    infiniband_devices: List[str]
    efa_devices: List[str]


@dataclass
class EnvFingerprint:
    """Complete environment fingerprint."""
    timestamp: str
    fingerprint_hash: str = ""
    system: Optional[SystemInfo] = None
    gpus: List[GPUInfo] = field(default_factory=list)
    cuda: Optional[CUDAInfo] = None
    python: Optional[PythonInfo] = None
    network: Optional[NetworkInfo] = None
    env_vars: Dict[str, str] = field(default_factory=dict)

    def compute_hash(self) -> str:
        """Compute a hash of critical configuration elements."""
        critical = {
            "gpu_vbios": [g.vbios_version for g in self.gpus],
            "gpu_driver": self.gpus[0].driver_version if self.gpus else "",
            "cuda_version": self.cuda.cuda_version if self.cuda else "",
            "torch_version": self.python.torch_version if self.python else "",
            "nccl_version": self.cuda.nccl_version if self.cuda else "",
        }
        content = json.dumps(critical, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]


def _run_cmd(cmd: List[str], default: str = "") -> str:
    """Run a command and return stdout, or default on failure."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=10
        )
        return result.stdout.strip() if result.returncode == 0 else default
    except Exception:
        return default


def _get_gpu_info() -> List[GPUInfo]:
    """Get information about all GPUs using nvidia-smi."""
    gpus = []

    # Query nvidia-smi for GPU info
    query = "index,uuid,name,vbios_version,driver_version,memory.total,pci.bus_id"
    output = _run_cmd([
        "nvidia-smi",
        f"--query-gpu={query}",
        "--format=csv,noheader,nounits"
    ])

    if not output:
        return gpus

    for line in output.strip().split("\n"):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 7:
            gpus.append(GPUInfo(
                index=int(parts[0]),
                uuid=parts[1],
                name=parts[2],
                vbios_version=parts[3],
                driver_version=parts[4],
                memory_total=parts[5],
                pcie_bus_id=parts[6],
            ))

    # Try to get additional info from nvidia-smi -q
    for gpu in gpus:
        xml_output = _run_cmd([
            "nvidia-smi", "-i", str(gpu.index), "-q"
        ])
        for line in xml_output.split("\n"):
            if "Serial Number" in line and ":" in line:
                gpu.serial = line.split(":")[-1].strip()
            elif "Board Part Number" in line and ":" in line:
                gpu.board_part_number = line.split(":")[-1].strip()
            elif "Image Version" in line and ":" in line:
                gpu.inforom_version = line.split(":")[-1].strip()

    return gpus


def _get_cuda_info() -> CUDAInfo:
    """Get CUDA-related information."""
    # CUDA version from nvidia-smi
    cuda_version = _run_cmd(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    nvcc_version = _run_cmd(["nvcc", "--version"])
    if "release" in nvcc_version.lower():
        for line in nvcc_version.split("\n"):
            if "release" in line.lower():
                cuda_version = line.split("release")[-1].split(",")[0].strip()
                break

    # cuDNN version
    cudnn_version = ""
    try:
        import torch
        cudnn_version = str(torch.backends.cudnn.version())
    except Exception:
        pass

    # NCCL version
    nccl_version = ""
    try:
        import torch.distributed as dist
        if hasattr(dist, 'get_nccl_version'):
            nccl_version = str(dist.get_nccl_version())
    except Exception:
        pass

    # Also try from nccl.h or environment
    if not nccl_version:
        nccl_version = os.environ.get("NCCL_VERSION", "")

    return CUDAInfo(
        cuda_version=cuda_version,
        cudnn_version=cudnn_version,
        nccl_version=nccl_version,
        cuda_home=os.environ.get("CUDA_HOME", ""),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    )


def _get_python_info() -> PythonInfo:
    """Get Python environment information."""
    torch_version = ""
    torch_cuda = ""
    torch_cudnn = ""
    torch_nccl = ""
    torch_compile_backend = ""
    triton_version = ""

    try:
        import torch
        torch_version = torch.__version__
        torch_cuda = torch.version.cuda or ""
        torch_cudnn = str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else ""

        # NCCL version
        try:
            import torch.distributed as dist
            if hasattr(dist, 'get_nccl_version'):
                torch_nccl = str(dist.get_nccl_version())
        except Exception:
            pass

        # torch.compile backend
        torch_compile_backend = os.environ.get("TORCH_COMPILE_BACKEND", "inductor")

    except ImportError:
        pass

    try:
        import triton
        triton_version = triton.__version__
    except ImportError:
        pass

    return PythonInfo(
        version=platform.python_version(),
        torch_version=torch_version,
        torch_cuda_version=torch_cuda,
        torch_cudnn_version=torch_cudnn,
        torch_nccl_version=torch_nccl,
        torch_compile_backend=torch_compile_backend,
        triton_version=triton_version,
    )


def _get_system_info() -> SystemInfo:
    """Get system-level information."""
    # CPU info
    cpu_model = ""
    cpu_info = _run_cmd(["cat", "/proc/cpuinfo"])
    for line in cpu_info.split("\n"):
        if "model name" in line:
            cpu_model = line.split(":")[-1].strip()
            break

    # Memory
    mem_total_gb = 0.0
    mem_info = _run_cmd(["cat", "/proc/meminfo"])
    for line in mem_info.split("\n"):
        if "MemTotal" in line:
            kb = int(line.split()[1])
            mem_total_gb = kb / 1024 / 1024
            break

    # NUMA nodes
    numa_nodes = 0
    numa_output = _run_cmd(["ls", "/sys/devices/system/node/"])
    numa_nodes = len([d for d in numa_output.split() if d.startswith("node")])

    # AWS EC2 instance ID (from board_asset_tag)
    instance_id = ""
    asset_tag_path = Path("/sys/devices/virtual/dmi/id/board_asset_tag")
    if asset_tag_path.exists():
        try:
            instance_id = asset_tag_path.read_text().strip()
        except Exception:
            pass

    return SystemInfo(
        hostname=platform.node(),
        kernel_version=platform.release(),
        os_release=_run_cmd(["cat", "/etc/os-release"]).split("\n")[0] if Path("/etc/os-release").exists() else "",
        cpu_model=cpu_model,
        cpu_count=os.cpu_count() or 0,
        memory_total_gb=round(mem_total_gb, 2),
        numa_nodes=numa_nodes,
        instance_id=instance_id,
    )


def _get_network_info() -> NetworkInfo:
    """Get network/interconnect information."""
    # InfiniBand devices
    ib_devices = []
    ib_output = _run_cmd(["ls", "/sys/class/infiniband/"])
    if ib_output:
        ib_devices = ib_output.split()

    # EFA devices
    efa_devices = []
    efa_output = _run_cmd(["ls", "/sys/class/infiniband_verbs/"])
    if efa_output:
        efa_devices = [d for d in efa_output.split() if "efa" in d.lower()]

    return NetworkInfo(
        nccl_net=os.environ.get("NCCL_NET", ""),
        nccl_socket_family=os.environ.get("NCCL_SOCKET_FAMILY", ""),
        efa_provider=os.environ.get("FI_PROVIDER", ""),
        infiniband_devices=ib_devices,
        efa_devices=efa_devices,
    )


def _get_relevant_env_vars() -> Dict[str, str]:
    """Get environment variables relevant to training determinism."""
    prefixes = [
        "NCCL_", "CUDA_", "TORCH_", "PYTORCH_", "TRITON_",
        "FI_", "GLOO_", "LOCAL_RANK", "RANK", "WORLD_SIZE",
        "MASTER_", "NVIDIA_", "OMP_", "MKL_", "OPENBLAS_",
    ]

    relevant = {}
    for key, value in os.environ.items():
        if any(key.startswith(p) for p in prefixes):
            relevant[key] = value

    return dict(sorted(relevant.items()))


def capture_fingerprint() -> EnvFingerprint:
    """Capture complete environment fingerprint."""
    from datetime import datetime

    fp = EnvFingerprint(
        timestamp=datetime.now().isoformat(),
        system=_get_system_info(),
        gpus=_get_gpu_info(),
        cuda=_get_cuda_info(),
        python=_get_python_info(),
        network=_get_network_info(),
        env_vars=_get_relevant_env_vars(),
    )
    fp.fingerprint_hash = fp.compute_hash()

    return fp


def fingerprint_to_dict(fp: EnvFingerprint) -> Dict[str, Any]:
    """Convert fingerprint to dictionary for JSON serialization."""
    return asdict(fp)


def save_fingerprint(fp: EnvFingerprint, path: str) -> None:
    """Save fingerprint to JSON file."""
    with open(path, "w") as f:
        json.dump(fingerprint_to_dict(fp), f, indent=2)
    logger.info(f"Saved environment fingerprint to {path}")


def load_fingerprint(path: str) -> Dict[str, Any]:
    """Load fingerprint from JSON file."""
    with open(path) as f:
        return json.load(f)


def compare_fingerprints(fp1: Dict[str, Any], fp2: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Compare two fingerprints and return differences.

    Returns dict with keys: 'critical', 'warning', 'info'
    """
    diffs = {"critical": [], "warning": [], "info": []}

    # Critical: GPU VBIOS versions
    vbios1 = [g.get("vbios_version", "") for g in fp1.get("gpus", [])]
    vbios2 = [g.get("vbios_version", "") for g in fp2.get("gpus", [])]
    if vbios1 != vbios2:
        diffs["critical"].append(f"GPU VBIOS versions differ: {vbios1} vs {vbios2}")

    # Critical: Driver version
    drv1 = fp1.get("gpus", [{}])[0].get("driver_version", "")
    drv2 = fp2.get("gpus", [{}])[0].get("driver_version", "")
    if drv1 != drv2:
        diffs["critical"].append(f"Driver versions differ: {drv1} vs {drv2}")

    # Critical: CUDA version
    cuda1 = fp1.get("cuda", {}).get("cuda_version", "")
    cuda2 = fp2.get("cuda", {}).get("cuda_version", "")
    if cuda1 != cuda2:
        diffs["critical"].append(f"CUDA versions differ: {cuda1} vs {cuda2}")

    # Critical: PyTorch version
    torch1 = fp1.get("python", {}).get("torch_version", "")
    torch2 = fp2.get("python", {}).get("torch_version", "")
    if torch1 != torch2:
        diffs["critical"].append(f"PyTorch versions differ: {torch1} vs {torch2}")

    # Warning: NCCL version
    nccl1 = fp1.get("cuda", {}).get("nccl_version", "")
    nccl2 = fp2.get("cuda", {}).get("nccl_version", "")
    if nccl1 != nccl2:
        diffs["warning"].append(f"NCCL versions differ: {nccl1} vs {nccl2}")

    # Warning: cuDNN version
    cudnn1 = fp1.get("cuda", {}).get("cudnn_version", "")
    cudnn2 = fp2.get("cuda", {}).get("cudnn_version", "")
    if cudnn1 != cudnn2:
        diffs["warning"].append(f"cuDNN versions differ: {cudnn1} vs {cudnn2}")

    # Warning: Triton version
    triton1 = fp1.get("python", {}).get("triton_version", "")
    triton2 = fp2.get("python", {}).get("triton_version", "")
    if triton1 != triton2:
        diffs["warning"].append(f"Triton versions differ: {triton1} vs {triton2}")

    # Info: GPU count
    ngpu1 = len(fp1.get("gpus", []))
    ngpu2 = len(fp2.get("gpus", []))
    if ngpu1 != ngpu2:
        diffs["info"].append(f"GPU count differs: {ngpu1} vs {ngpu2}")

    # Info: Hostname
    host1 = fp1.get("system", {}).get("hostname", "")
    host2 = fp2.get("system", {}).get("hostname", "")
    if host1 != host2:
        diffs["info"].append(f"Hostnames differ: {host1} vs {host2}")

    # Compare relevant env vars
    env1 = fp1.get("env_vars", {})
    env2 = fp2.get("env_vars", {})
    critical_env = ["NCCL_ALGO", "NCCL_PROTO", "CUDA_VISIBLE_DEVICES"]
    for key in critical_env:
        if env1.get(key) != env2.get(key):
            diffs["warning"].append(f"Env {key} differs: {env1.get(key)} vs {env2.get(key)}")

    return diffs


def print_fingerprint_summary(fp: EnvFingerprint) -> None:
    """Print a human-readable summary of the fingerprint."""
    print(f"=== Environment Fingerprint ===")
    print(f"Hash: {fp.fingerprint_hash}")
    print(f"Timestamp: {fp.timestamp}")
    print()

    if fp.system:
        print(f"System:")
        print(f"  Hostname: {fp.system.hostname}")
        if fp.system.instance_id:
            print(f"  Instance ID: {fp.system.instance_id}")
        print(f"  Kernel: {fp.system.kernel_version}")
        print(f"  CPU: {fp.system.cpu_model} ({fp.system.cpu_count} cores)")
        print(f"  Memory: {fp.system.memory_total_gb} GB")
        print(f"  NUMA nodes: {fp.system.numa_nodes}")
        print()

    if fp.gpus:
        print(f"GPUs ({len(fp.gpus)}):")
        for gpu in fp.gpus:
            print(f"  [{gpu.index}] {gpu.name}")
            print(f"      UUID: {gpu.uuid}")
            print(f"      VBIOS: {gpu.vbios_version}")
            print(f"      Driver: {gpu.driver_version}")
            print(f"      Memory: {gpu.memory_total}")
        print()

    if fp.cuda:
        print(f"CUDA:")
        print(f"  CUDA: {fp.cuda.cuda_version}")
        print(f"  cuDNN: {fp.cuda.cudnn_version}")
        print(f"  NCCL: {fp.cuda.nccl_version}")
        print()

    if fp.python:
        print(f"Python:")
        print(f"  Version: {fp.python.version}")
        print(f"  PyTorch: {fp.python.torch_version}")
        print(f"  Triton: {fp.python.triton_version}")
        print()

    if fp.network:
        print(f"Network:")
        print(f"  NCCL_NET: {fp.network.nccl_net}")
        print(f"  FI_PROVIDER: {fp.network.efa_provider}")
        print(f"  IB devices: {fp.network.infiniband_devices}")
        print()


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Capture environment fingerprint")
    parser.add_argument("-o", "--output", help="Output JSON file")
    parser.add_argument("--compare", nargs=2, metavar=("FILE1", "FILE2"),
                       help="Compare two fingerprint files")
    parser.add_argument("-q", "--quiet", action="store_true",
                       help="Only output JSON, no summary")
    args = parser.parse_args()

    if args.compare:
        fp1 = load_fingerprint(args.compare[0])
        fp2 = load_fingerprint(args.compare[1])
        diffs = compare_fingerprints(fp1, fp2)

        print(f"Comparing: {args.compare[0]} vs {args.compare[1]}")
        print()

        if diffs["critical"]:
            print("[CRITICAL] Differences that likely cause divergence:")
            for d in diffs["critical"]:
                print(f"  - {d}")
            print()

        if diffs["warning"]:
            print("[WARNING] Differences that may cause divergence:")
            for d in diffs["warning"]:
                print(f"  - {d}")
            print()

        if diffs["info"]:
            print("[INFO] Other differences:")
            for d in diffs["info"]:
                print(f"  - {d}")
            print()

        if not any(diffs.values()):
            print("[OK] No significant differences found")

        return

    # Capture fingerprint
    fp = capture_fingerprint()

    if not args.quiet:
        print_fingerprint_summary(fp)

    if args.output:
        save_fingerprint(fp, args.output)
        print(f"Saved to: {args.output}")
    else:
        print(json.dumps(fingerprint_to_dict(fp), indent=2))


if __name__ == "__main__":
    main()
