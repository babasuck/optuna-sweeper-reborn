import logging
import subprocess
from typing import Optional

log = logging.getLogger(__name__)


class DashboardManager:
    def __init__(self, storage: str, host: str = "localhost", port: int = 8080) -> None:
        self.storage = storage
        self.host = host
        self.port = port
        self._process: Optional[subprocess.Popen] = None

    def start(self) -> None:
        try:
            self._process = subprocess.Popen(
                [
                    "optuna-dashboard",
                    self.storage,
                    "--host",
                    self.host,
                    "--port",
                    str(self.port),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            log.info(
                f"Optuna Dashboard started at http://{self.host}:{self.port}/"
            )
        except FileNotFoundError:
            log.warning(
                "optuna-dashboard is not installed. "
                "Install it with: pip install optuna-dashboard"
            )

    def stop(self) -> None:
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            log.info("Optuna Dashboard stopped.")
            self._process = None
