import importlib.util
import subprocess
import sys


def install_nvdiffrast():
    if importlib.util.find_spec("nvdiffrast") is not None:
        return

    subprocess.run(
        [
            sys.executable, "-m", "pip", "install",
            "--no-deps", "--no-build-isolation",
            "git+https://github.com/NVlabs/nvdiffrast",
        ],
        check=True,
    )
