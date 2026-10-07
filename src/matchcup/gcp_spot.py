"""gcloud argument builder for a Spot A100 training VM.

The L4 fold pipeline in ``scripts/gcpctl.py`` caps every job at
``--max-run-duration=8h`` on on-demand ``g2-standard-8``. The full-Silver
pretrain needs ~19-25h on a single A100 and must ride Spot capacity — the only
A100 quota this project was granted — so it gets its own launch flags: Spot
provisioning, no run-duration cap, and STOP-on-preemption so a supervisor can
restart the same VM. Durability comes from GCS-mirrored checkpoints (see
``checkpoint_gcs``), so losing the local disk on preemption only costs the work
since the last 20-minute checkpoint.

The flags are built by a pure function so the properties that matter — Spot, no
8-hour cap, an A100 machine type — are unit-tested without invoking gcloud.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

# 1x A100 40GB + 12 vCPU: matches the granted Spot A100 and 12 preemptible CPUs.
A100_MACHINE_TYPE = "a2-highgpu-1g"


def spot_instance_create_args(
    *,
    name: str,
    project: str,
    zone: str,
    image_family: str,
    image_project: str,
    startup_script: str | Path,
    metadata: Sequence[str],
    machine_type: str = A100_MACHINE_TYPE,
    boot_disk_gb: int = 200,
) -> list[str]:
    """Build the ``gcloud compute instances create`` command for a Spot A100 job.

    Deliberately omits ``--max-run-duration``: a Spot VM already ends on
    preemption, and the whole point of this launch is to outlive the 8-hour cap
    the on-demand pipeline imposes. ``--instance-termination-action=STOP`` keeps
    the (empty, disposable) disk so the supervisor can ``instances start`` the
    same VM; the checkpoint it resumes from is pulled from GCS regardless.
    """
    if not metadata:
        raise ValueError("Spot VM needs metadata (bucket, stage, checkpoint prefix)")
    return [
        "gcloud",
        "compute",
        "instances",
        "create",
        name,
        f"--project={project}",
        f"--zone={zone}",
        f"--machine-type={machine_type}",
        f"--image-family={image_family}",
        f"--image-project={image_project}",
        f"--boot-disk-size={boot_disk_gb}GB",
        "--boot-disk-type=pd-balanced",
        "--maintenance-policy=TERMINATE",
        "--provisioning-model=SPOT",
        "--instance-termination-action=STOP",
        "--scopes=cloud-platform",
        f"--metadata={','.join(metadata)}",
        f"--metadata-from-file=startup-script={startup_script}",
    ]
