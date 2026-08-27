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
        mock_process.poll.return_value = None  # still running
        with patch(
            "hydra_plugins.hydra_optuna_sweeper_reborn._dashboard.subprocess.Popen",
            return_value=mock_process,
        ):
            manager.start()
        assert manager._process is mock_process

    def test_start_detects_immediate_exit(self):
        """A busy port makes optuna-dashboard die right away; we must not report
        it as started."""
        manager = DashboardManager(
            storage="sqlite:///test.db", host="localhost", port=9999
        )
        mock_process = MagicMock()
        mock_process.poll.return_value = 1
        with patch(
            "hydra_plugins.hydra_optuna_sweeper_reborn._dashboard.subprocess.Popen",
            return_value=mock_process,
        ):
            manager.start()
        assert manager._process is None

    def test_start_does_not_pipe_output(self):
        """PIPEs are never drained here, and a full pipe would deadlock the
        dashboard partway through a long sweep."""
        import subprocess

        manager = DashboardManager(storage="sqlite:///test.db")
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        with patch(
            "hydra_plugins.hydra_optuna_sweeper_reborn._dashboard.subprocess.Popen",
            return_value=mock_process,
        ) as popen:
            manager.start()
        kwargs = popen.call_args.kwargs
        assert kwargs["stdout"] is subprocess.DEVNULL
        assert kwargs["stderr"] is subprocess.DEVNULL

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
