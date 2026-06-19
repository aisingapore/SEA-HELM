import subprocess

from src.base_logger import get_logger
from src.sandbox.base_sandbox import BaseSandbox

logger = get_logger(__name__)


class PodmanSandbox(BaseSandbox):
    def __init__(self):
        super().__init__(server_name="Podman Sandbox Server")

    def start_process(self):
        self.process = subprocess.Popen(
            f"podman run --rm --env _BYTEFAAS_RUNTIME_PORT={self.port} docker://volcengine/sandbox-fusion:server-20250609",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
        )


if __name__ == "__main__":
    sandbox = PodmanSandbox()
    sandbox.start_sandbox()
    output = sandbox.run_code('print("Hello, world!")')
    print(f"Output from sandbox: {output}")
