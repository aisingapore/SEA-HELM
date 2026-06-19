import os
import subprocess

from src.base_logger import get_logger
from src.sandbox.base_sandbox import BaseSandbox

logger = get_logger(__name__)


class EnrootSandbox(BaseSandbox):
    def __init__(self, image_location: str = "./enroot_images"):
        super().__init__(server_name="Enroot Sandbox Server")
        self.image_location = image_location

    def start_process(self):
        # Download docker image
        os.makedirs(self.image_location, exist_ok=True)
        subprocess.run(
            f"enroot import -o {self.image_location}/sandbox-fusion.sqsh docker://volcengine/sandbox-fusion:server-20250609",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
        )
        subprocess.run(
            f"enroot create -n sandbox-fusion {self.image_location}/sandbox-fusion.sqsh",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
        )
        self.process = subprocess.Popen(
            f"enroot start --env _BYTEFAAS_RUNTIME_PORT={self.port} sandbox-fusion",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
        )


if __name__ == "__main__":
    sandbox = EnrootSandbox()
    sandbox.start_sandbox()
    output = sandbox.run_code('print("Hello, world!")')
    print(f"Output from sandbox: {output}")
