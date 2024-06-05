"""General constants and helper classes to run the main experiments on htcondor.
"""
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import ClassVar

import classad
import htcondor

from folktexts._utils import hash_dict

# Cluster settings
DEFAULT_JOB_BID = 25            # htcondor bid (min. is 15 apparently...)
DEFAULT_JOB_CPUS = 4            # number of CPUs per experiment (per cluster job)
DEFAULT_JOB_MEMORY_GB = 62      # GBs of memory
DEFAULT_GPU_MEMORY_GB = 30      # GBs of GPU memory

MAX_RUNNING_PRICE = 1500        # Max price for running a job


@dataclass
class Experiment:
    """A generic experiment to run on the cluster."""
    executable_path: str
    kwargs: dict = field(default_factory=dict)

    job_cpus: int = DEFAULT_JOB_CPUS
    job_gpus: int = 0
    job_memory_gb: int = DEFAULT_JOB_MEMORY_GB
    job_gpu_memory_gb: int = DEFAULT_GPU_MEMORY_GB
    job_bid: int = DEFAULT_JOB_BID

    _all_experiments: ClassVar[list["Experiment"]] = []

    def __post_init__(self):
        # Add experiment to the class-level list
        self._all_experiments.append(self)

    @classmethod
    def get_all_experiments(cls):
        return cls._all_experiments

    def __getattr__(self, name: str):
        if name in self.kwargs:
            return self.kwargs[name]
        else:
            raise AttributeError(f"Attribute '{name}' not found in Experiment.")

    def hash(self) -> str:
        """Generate a hexadecimal hash that uniquely identifies the experiment's arguments.
        """
        # Get hash of the experiment's arguments
        kwargs_for_hash = dict(
            executable_path=self.executable_path,
            **self.kwargs,
        )

        # These kwargs shouldn't be used to generate a unique hash
        kwargs_for_hash.pop("results_dir", None)
        kwargs_for_hash.pop("hash", None)

        return hash_dict(kwargs_for_hash)

    def to_dict(self) -> dict:
        return asdict(self)


def launch_experiment_job(exp: Experiment):

    # Name/prefix for cluster logs related to this job
    cluster_job_log_name = (Path(exp.results_dir) / "log.$(Cluster).$(Process)").as_posix()

    # Construct executable cmd-line arguments
    cmd_line_args = " ".join(f"--{key.replace('_', '-')} {value}" for key, value in exp.kwargs.items())

    # Construct job description
    job_description = htcondor.Submit({
        "executable": f"{sys.executable}",  # correct env for the python executable
        "arguments": f"{exp.executable_path} {cmd_line_args}",
        "output": f"{cluster_job_log_name}.out",
        "error": f"{cluster_job_log_name}.err",
        "log": f"{cluster_job_log_name}.log",
        "request_cpus": f"{exp.job_cpus}",
        "request_gpus": f"{exp.job_gpus}",
        "request_memory": f"{exp.job_memory_gb}GB",
        "request_disk": "10GB",
        "jobprio": f"{exp.job_bid - 1000}",
        "notify_user": "andre.cruz@tuebingen.mpg.de",
        "notification": "error",

        # GPU requirements
        "requirements": (
            f"(TARGET.CUDAGlobalMemoryMb > {exp.job_gpu_memory_gb * 1_000})"
        ) if exp.job_gpus > 0 else "",

        # Concurrency limits:
        # > each job uses this amount of resources out of a pool of 10k
        "concurrency_limits": "user.folktexts:100",     # 100 jobs in parallel

        "+MaxRunningPrice": MAX_RUNNING_PRICE,
        "+RunningPriceExceededAction": classad.quote("restart"),
    })

    # Submit job to the htcondor scheduler
    schedd = htcondor.Schedd()
    submit_result = schedd.submit(job_description)

    logging.info(
        f"Launched {submit_result.num_procs()} processe(s) with "
        f"cluster-ID={submit_result.cluster()}\n")

    return submit_result
