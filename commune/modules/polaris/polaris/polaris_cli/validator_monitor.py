import msvcrt
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from queue import Queue
from threading import Event, Thread

import psutil
from rich import box
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

class BaseLogReader:
    def __init__(self, log_path, queue, log_type):
        self.log_path = log_path
        self.queue = queue
        self.log_type = log_type
        self.running = Event()
        self.thread = Thread(target=self._read_log, daemon=True)
        self.last_position = 0
        
    def _read_existing_content(self):
        try:
            with open(self.log_path, 'r', buffering=1) as f:
                content = f.read()
                self.last_position = f.tell()
                lines = content.splitlines()
                for line in lines:
                    if line:
                        self.queue.put((line, False, self.log_type))
        except Exception as e:
            self.queue.put((f"Error reading log: {e}", False, self.log_type))

    def start(self):
        self._read_existing_content()
        self.running.set()
        self.thread.start()

    def stop(self):
        self.running.clear()
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def _read_log(self):
        raise NotImplementedError("Subclasses must implement _read_log")

class UnixLogReader(BaseLogReader):
    def _read_log(self):
        try:
            import select
            with open(self.log_path, 'r') as f:
                fd = f.fileno()
                poll_obj = select.poll()
                poll_obj.register(fd, select.POLLIN)

                while self.running.is_set():
                    events = poll_obj.poll(100)
                    if events:
                        f.seek(self.last_position)
                        new_content = f.read()
                        self.last_position = f.tell()
                        
                        if new_content:
                            lines = new_content.splitlines()
                            for line in lines:
                                if line:
                                    self.queue.put((line, True, self.log_type))
        except Exception as e:
            self.queue.put((f"Error monitoring log: {e}", True, self.log_type))

class WindowsLogReader(BaseLogReader):
    def _read_log(self):
        try:
            with open(self.log_path, 'r') as f:
                while self.running.is_set():
                    f.seek(self.last_position)
                    new_content = f.read()
                    if new_content:
                        self.last_position = f.tell()
                        lines = new_content.splitlines()
                        for line in lines:
                            if line:
                                self.queue.put((line, True, self.log_type))
                    time.sleep(0.1)
        except Exception as e:
            self.queue.put((f"Error monitoring log: {e}", True, self.log_type))

class ValidatorStats:
    def __init__(self, pid):
        self.pid = pid
        self.process = psutil.Process(pid)
        self.start_time = datetime.now()
        self.last_cpu_time = None
        self.last_check_time = None
        
    def get_status(self):
        try:
            with self.process.oneshot():
                current_time = time.time()
                current_cpu_time = sum(self.process.cpu_times())
                
                if self.last_cpu_time is not None:
                    time_diff = current_time - self.last_check_time
                    cpu_percent = ((current_cpu_time - self.last_cpu_time) / time_diff * 100)
                else:
                    cpu_percent = 0.0
                
                self.last_cpu_time = current_cpu_time
                self.last_check_time = current_time
                
                memory_info = self.process.memory_info()
                uptime = datetime.now() - self.start_time
                uptime_str = str(uptime).split('.')[0]
                
                status_text = Text()
                status_text.append("✨ Validator Statistics\n\n", style="bold cyan")
                status_text.append("PID: ", style="bold blue")
                status_text.append(f"{self.pid}\n", style="white")
                status_text.append("Uptime: ", style="bold blue")
                status_text.append(f"{uptime_str}\n", style="white")
                status_text.append("CPU: ", style="bold blue")
                status_text.append(f"{cpu_percent:.1f}%\n", style="green")
                status_text.append("Memory: ", style="bold blue")
                status_text.append(f"{memory_info.rss / 1024 / 1024:.1f} MB\n", style="green")
                status_text.append("Status: ", style="bold blue")
                status_text.append(f"{self.process.status()}\n", style="white")
                
                return Panel(
                    status_text,
                    border_style="cyan",
                    box=box.ROUNDED,
                    title="[bold cyan]Validator Monitor[/bold cyan]",
                    subtitle="[dim]↑/↓ to scroll · Ctrl+C to exit[/dim]"
                )
        except psutil.NoSuchProcess:
            return Panel(
                "[red]Validator process terminated[/red]", 
                border_style="red",
                box=box.ROUNDED
            )
        except Exception as e:
            return Panel(
                f"[red]Monitor error: {e}[/red]",
                border_style="red",
                box=box.ROUNDED
            )

def parse_log_line(line, log_type):
    try:
        # Handle standard timestamp format "YYYY-MM-DD HH:MM:SS,mmm"
        if " | " in line:
            parts = line.split(" | ", 2)
            if len(parts) == 3:
                timestamp, level, message = parts
                return timestamp.strip(), level.strip(), message.strip()
        
        return "", log_type.upper(), line.strip()
    except Exception:
        return "", log_type.upper(), line.strip()

def format_logs(log_lines, max_lines=1000, visible_lines=20, manual_scroll=False):
    table = Table(
        show_header=True,
        header_style="bold cyan",
        box=box.ROUNDED,
        expand=True,
        title="[bold cyan]Validator Logs[/bold cyan]",
        caption=f"[dim]Showing {min(len(log_lines), max_lines)} logs (↑/↓ to scroll)[/dim]"
    )
    
    table.add_column("Timestamp", style="cyan", no_wrap=True)
    table.add_column("Level", style="bold yellow", width=12)
    table.add_column("Message", style="white", ratio=1)
    table.add_column("Stream", style="dim", width=8)
    table.add_column("", style="green dim", width=3)

    # Keep only the last max_lines
    log_lines = log_lines[-max_lines:]
    total_rows = len(log_lines)
    
    # Calculate start position - auto-scroll to bottom unless manual scroll is active
    if manual_scroll:
        start_idx = getattr(format_logs, 'scroll_position', max(0, total_rows - visible_lines))
        start_idx = min(start_idx, max(0, total_rows - visible_lines))
    else:
        start_idx = max(0, total_rows - visible_lines)
    
    # Store the current scroll position
    format_logs.scroll_position = start_idx
    
    # Get visible portion of logs
    visible_logs = log_lines[start_idx:start_idx + visible_lines]
    
    for line, is_new, log_type in visible_logs:
        timestamp, level, message = parse_log_line(line, log_type)
        
        # Determine message style based on content and stream
        if log_type == "stderr" or "ERROR" in level:
            msg_style = "red"
        elif "WARNING" in level:
            msg_style = "yellow"
        elif "INFO" in level:
            msg_style = "green"
        else:
            msg_style = "white"

        table.add_row(
            timestamp,
            level,
            Text(message, style=msg_style),
            Text(log_type, style="dim"),
            "●" if is_new else ""
        )

    return table

def get_validator_pid():
    pid_file = Path.home() / '.polaris' / 'validator' / 'pids' / 'validator.pid'
    try:
        if pid_file.exists():
            pid = int(pid_file.read_text().strip())
            if psutil.pid_exists(pid):
                return pid
    except:
        pass
    return None

def get_log_reader_class():
    system = platform.system().lower()
    if system == 'windows':
        return WindowsLogReader
    return UnixLogReader

def monitor_validator_logs():
    manual_scroll_active = False  # Track if user has activated manual scroll
    validator_pid = get_validator_pid()
    if not validator_pid:
        console.print("[yellow]No active validator process found.[/yellow]")
        return

    log_dir = Path.home() / '.polaris' / 'validator' / 'logs'
    stdout_path = log_dir / 'validator_stdout.log'
    stderr_path = log_dir / 'validator_stderr.log'

    if not stdout_path.exists() and not stderr_path.exists():
        console.print("[red]No validator log files found.[/red]")
        return

    try:
        log_lines = []
        log_queue = Queue()
        readers = []
        
        LogReader = get_log_reader_class()
        
        # Start stdout reader if file exists
        if stdout_path.exists():
            stdout_reader = LogReader(str(stdout_path), log_queue, "stdout")
            stdout_reader.start()
            readers.append(stdout_reader)
            
        # Start stderr reader if file exists
        if stderr_path.exists():
            stderr_reader = LogReader(str(stderr_path), log_queue, "stderr")
            stderr_reader.start()
            readers.append(stderr_reader)

        validator_stats = ValidatorStats(validator_pid)

        with Live(
            auto_refresh=True,
            refresh_per_second=4,
            vertical_overflow="visible",
            screen=True
        ) as live:
            try:
                while True:
                    # Check for keyboard input (arrow keys)
                    if msvcrt.kbhit():
                        key = msvcrt.getch()
                        if key in [b'H', b'P']:  # Up or Down arrow
                            manual_scroll_active = True
                            current_pos = getattr(format_logs, 'scroll_position', 0)
                            if key == b'H':  # Up arrow
                                format_logs.scroll_position = max(0, current_pos - 1)
                            else:  # Down arrow
                                format_logs.scroll_position = current_pos + 1

                    # Process new log entries
                    while not log_queue.empty():
                        line, is_new, log_type = log_queue.get_nowait()
                        log_lines.append((line, is_new, log_type))
                        # Reset manual scroll if we're already at the bottom
                        if manual_scroll_active and format_logs.scroll_position >= len(log_lines) - 20:
                            manual_scroll_active = False

                    # Create layout with stats and logs
                    layout = Layout()
                    layout.split_column(
                        Layout(validator_stats.get_status(), size=10),
                        Layout(format_logs(log_lines, manual_scroll=manual_scroll_active))
                    )
                    
                    live.update(layout)
                    time.sleep(0.25)

            except KeyboardInterrupt:
                console.print("\n[yellow]Monitoring stopped.[/yellow]")
            finally:
                for reader in readers:
                    reader.stop()

    except Exception as e:
        console.print(f"[red]Monitor failed: {str(e)}[/red]")

if __name__ == "__main__":
    monitor_validator_logs()