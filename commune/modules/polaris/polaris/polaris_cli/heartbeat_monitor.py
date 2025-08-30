# src/heartbeat_monitor.py

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import psutil
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from .start import get_project_root, read_pid

console = Console()
logger = logging.getLogger(__name__)

class HeartbeatMonitor:
    def __init__(self):
        self.pulse_chars = "▁▂▃▄▅▆▇█▇▆▅▄▃▂▁"
        self.pulse_index = 0
        self.last_beat = time.time()
        self.heartbeat_pid = None
        self.status = "Initializing"
        self.metrics = {}
        self.last_update = datetime.now()

    def get_heartbeat_status(self):
        """Get current status of heartbeat service"""
        try:
            if not self.heartbeat_pid:
                self.heartbeat_pid = read_pid('heartbeat')
            
            if not self.heartbeat_pid:
                return "Not Running"

            process = psutil.Process(self.heartbeat_pid)
            if process.is_running():
                return "Online"
            return "Offline"
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return "Error"

    def update_metrics(self):
        """Update system metrics"""
        try:
            polaris_pid = read_pid('polaris')
            if polaris_pid and psutil.pid_exists(polaris_pid):
                process = psutil.Process(polaris_pid)
                self.metrics.update({
                    'cpu_usage': f"{process.cpu_percent()}%",
                    'memory_usage': f"{process.memory_percent():.1f}%",
                    'connections': len(process.connections()),
                    'threads': process.num_threads()
                })

            # Read latest system info
            system_info_path = os.path.join(get_project_root(), 'system_info.json')
            if os.path.exists(system_info_path):
                with open(system_info_path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list) and data:
                        self.metrics['system_info'] = data[0]

        except Exception as e:
            logger.error(f"Error updating metrics: {e}")

    def generate_display(self) -> Layout:
        """Generate the display layout"""
        # Update status
        self.status = self.get_heartbeat_status()
        self.update_metrics()
        
        # Create layout
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body", size=3),
            Layout(name="footer")
        )

        # Header with status
        status_color = "green" if self.status == "Online" else "red"
        header = Panel(
            Text(f"Status: {self.status}", style=status_color),
            title="Miner Heartbeat Monitor",
            border_style=status_color
        )

        # Animated heartbeat display
        self.pulse_index = (self.pulse_index + 1) % len(self.pulse_chars)
        pulse = self.pulse_chars[self.pulse_index]
        current_time = datetime.now().strftime("%H:%M:%S")
        body = Panel(
            Text(f"♥ {pulse * 5}\nTime: {current_time}", justify="center", style="red"),
            title="Real-time Heartbeat",
            border_style="red"
        )

        # System metrics
        metrics_text = [
            f"CPU Usage: {self.metrics.get('cpu_usage', 'N/A')}",
            f"Memory Usage: {self.metrics.get('memory_usage', 'N/A')}",
            f"Active Connections: {self.metrics.get('connections', 'N/A')}",
            f"Running Threads: {self.metrics.get('threads', 'N/A')}"
        ]

        if 'system_info' in self.metrics:
            sys_info = self.metrics['system_info']
            metrics_text.extend([
                "",
                "System Information:",
                f"Hostname: {sys_info.get('hostname', 'N/A')}",
                f"Platform: {sys_info.get('platform', 'N/A')}"
            ])

        footer = Panel(
            Text("\n".join(metrics_text)),
            title="System Metrics",
            border_style="blue"
        )

        # Update layout
        layout["header"].update(header)
        layout["body"].update(body)
        layout["footer"].update(footer)

        return layout

def monitor_heartbeat():
    """Monitor heartbeat signals in real-time"""
    try:
        # Check if heartbeat service is running
        heartbeat_pid = read_pid('heartbeat')
        if not heartbeat_pid:
            console.print("[red]Heartbeat service is not running. Start Polaris first.[/red]")
            return

        monitor = HeartbeatMonitor()
        with Live(
            monitor.generate_display(),
            refresh_per_second=4,
            screen=True
        ) as live:
            try:
                while True:
                    live.update(monitor.generate_display())
                    time.sleep(0.25)  # 4 times per second
            except KeyboardInterrupt:
                console.print("\n[yellow]Heartbeat monitoring stopped.[/yellow]")

    except Exception as e:
        logger.error(f"Error in heartbeat monitor: {e}")
        console.print(f"[red]Error monitoring heartbeat: {e}[/red]")
        sys.exit(1)