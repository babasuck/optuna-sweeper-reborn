from unittest.mock import MagicMock, patch

from hydra_plugins.hydra_optuna_sweeper_reborn._dashboard import DashboardManager


class TestDashboardManager:
    def test_start_with_missing_binary(self):
        """When optuna-dashboard is not installed, should log warning."""
        manager = DashboardManager(
            storage="sqlite:///test.db", host="localhost", port=9999
        )
        # FileNotFoundError is caught internally
        with patch(
            "hydra_plugins.hydra_optuna_sweeper_reborn._dashboard.subprocess.Popen",
            side_effect=FileNotFoundError,
        ):
            manager.start()
        assert manager._process is None

    def test_start_success(self):
        manager = DashboardManager(
            storage="sqlite:///test.db", host="localhost", port=9999
        )
        mock_process = MagicMock()
        with patch(
            "hydra_plugins.hydra_optuna_sweeper_reborn._dashboard.subprocess.Popen",
            return_value=mock_process,
        ):
            manager.start()
        assert manager._process is mock_process

    def test_stop(self):
        manager = DashboardManager(storage="sqlite:///test.db")
        mock_process = MagicMock()
        manager._process = mock_process
        manager.stop()
        mock_process.terminate.assert_called_once()
        assert manager._process is None

    def test_stop_no_process(self):
        manager = DashboardManager(storage="sqlite:///test.db")
        # Should not raise
        manager.stop()
