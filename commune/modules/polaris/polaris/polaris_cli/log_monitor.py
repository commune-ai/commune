import os
import platform
import sys
import tty
import termios
import select
import time
from datetime import datetime
from queue import Queue
from threading import Event, Thread

import psutil
from rich import box
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.style import Style
from rich.table import Table
from rich.text import Text

console = Console()

class KeyboardReader:
    def __init__(self):
        self.is_windows = platform.system().lower() == 'windows'
        if self.is_windows:
            import msvcrt
            self.msvcrt = msvcrt
        else:
            self.old_settings = None

    def start(self):
        if not self.is_windows:
            self.old_settings = termios.tcgetattr(sys.stdin)
            tty.setraw(sys.stdin.fileno())

    def stop(self):
        if not self.is_windows and self.old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)

    def kbhit(self):
        if self.is_windows:
            return self.msvcrt.kbhit()
        else:
            dr, dw, de = select.select([sys.stdin], [], [], 0)
            return dr != []

    def getch(self):
        if self.is_windows:
            return self.msvcrt.getch()
        else:
            return sys.stdin.read(1).encode()

class BaseLogReader:
    def __init__(self, log_path, log_type, queue):
        self.log_path = log_path
        self.log_type = log_type
        self.queue = queue
        self.running = Event()
        self.thread = Thread(target=self._read_log, daemon=True)
        self.last_position = 0
        
    def _read_existing_content(self):
        """Read existing content exactly as is"""
        try:
            with open(self.log_path, 'r', buffering=1) as f:
                content = f.read()
                self.last_position = f.tell()
                lines = content.splitlines()
                for line in lines:
                    if line:  # Only send non-empty lines
                        self.queue.put((self.log_type, line, False))
        except Exception as e:
            self.queue.put((self.log_type, f"Error reading log: {e}", False))

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
                                    self.queue.put((self.log_type, line, True))
        except Exception as e:
            self.queue.put((self.log_type, f"Error monitoring log: {e}", True))

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
                                self.queue.put((self.log_type, line, True))
                    time.sleep(0.1)
        except Exception as e:
            self.queue.put((self.log_type, f"Error monitoring log: {e}", True))

def get_log_reader_class():
    system = platform.system().lower()
    if system == 'windows':
        return WindowsLogReader
    else:
        return UnixLogReader

class ProcessStats:
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
                    cpu_percent = ((current_cpu_time - self.last_cpu_time) / time_diff * 100) if time_diff > 0 else 0.0
                else:
                    cpu_percent = 0.0
                
                self.last_cpu_time = current_cpu_time
                self.last_check_time = current_time
                
                memory_info = self.process.memory_info()
                uptime = datetime.now() - self.start_time
                uptime_str = str(uptime).split('.')[0]
                
                status_text = Text()
                status_text.append("✨ Process Statistics\n\n", style="bold cyan")
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
                    title="[bold cyan]Compute Subnet Monitor[/bold cyan]",
                    subtitle="[dim]↑/↓ to scroll · Ctrl+C to exit[/dim]"
                )
        except psutil.NoSuchProcess:
            return Panel(
                "[red]Process terminated[/red]", 
                border_style="red",
                box=box.ROUNDED
            )
        except Exception as e:
            return Panel(
                f"[red]Monitor error: {e}[/red]",
                border_style="red",
                box=box.ROUNDED
            )

def parse_log_line(line):
    """Parse log line exactly as it appears in the file"""
    try:
        # Handle Cryptography warnings with file paths
        if "CryptographyDeprecationWarning" in line:
            if ":" in line:
                file_path, rest = line.split(":", 1)
                if ".py" in file_path:
                    file_num, message = rest.split(":", 1)
                    return f"{file_path}:{file_num}", "Warning", message.strip()
            return "", "Warning", line.strip()

        # Handle JSON-like dictionary entries
        if line.strip().startswith('"') and ":" in line:
            return "", "", line.strip()

        # Handle standard timestamp format
        if line.strip().startswith("2025-"):
            try:
                date_time = line[:23]  # Get "2025-01-03 07:11:44,084"
                msg = line[23:].strip()
                return date_time, "", msg
            except:
                return "", "", line.strip()

        return "", "", line.strip()
    except Exception:
        return "", "", line.strip()

def format_logs(log_lines, max_lines=1000, visible_lines=20):
    table = Table(
        show_header=True,
        header_style="bold cyan",
        box=box.ROUNDED,
        expand=True,
        title="[bold cyan]Compute Subnet Logs[/bold cyan]",
        caption=f"[dim]Showing {min(len(log_lines), max_lines)} logs (↑/↓ to scroll)[/dim]",
        padding=(0, 1),
        collapse_padding=True
    )
    
    table.add_column("Source/Time", style="cyan", no_wrap=True)
    table.add_column("Level", style="bold yellow", width=10)
    table.add_column("Message", style="white", ratio=1)
    table.add_column("", style="green dim", width=3)

    # Keep only the last max_lines
    log_lines = log_lines[-max_lines:]
    rows = []
    
    for line, is_new in log_lines:
        source_time, level, message = parse_log_line(str(line))
        
        # Determine message style based on content
        if "class" in message or "cipher" in message:
            msg_style = Style(color="yellow", dim=True)
        elif " - INFO" in message:
            msg_style = Style(color="green")
        elif "WARNING" in message or "Warning" in level:
            msg_style = Style(color="yellow")
        elif "ERROR" in message:
            msg_style = Style(color="red")
        else:
            msg_style = Style(color="white")

        rows.append((
            Text(source_time, style="cyan"),
            Text(level, style="yellow bold") if level else Text(""),
            Text(message, style=msg_style),
            "●" if is_new else ""
        ))

    # Custom sort: Warnings first, then timestamps
    def sort_key(row):
        timestamp = row[0].plain
        if "Warning" in row[1].plain:
            return (0, timestamp)
        if timestamp.startswith("2025-"):
            return (1, timestamp)
        return (2, timestamp)
    
    rows.sort(key=sort_key)
    
    # Calculate visible range for scrolling
    total_rows = len(rows)
    scroll_position = getattr(format_logs, 'scroll_position', 0)
    
    # Ensure scroll position is within bounds
    max_scroll = max(0, total_rows - visible_lines)
    scroll_position = max(0, min(scroll_position, max_scroll))
    format_logs.scroll_position = scroll_position
    
    # Only add visible rows
    start_idx = scroll_position
    end_idx = min(start_idx + visible_lines, total_rows)
    
    visible_rows = rows[start_idx:end_idx]
    for row in visible_rows:
        table.add_row(*row)

    return table

def get_log_path():
    """Get path to the compute subnet stderr log file"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.path.join(project_root, 'logs')
    return os.path.join(log_dir, 'polarise.log')

def monitor_logs(process_pid=None):
    log_path = get_log_path()
    format_logs.scroll_position = 0  # Initialize scroll position
    
    # Ensure log directory exists
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    if not os.path.exists(log_path):
        open(log_path, 'a').close()

    try:
        log_lines = []
        log_queue = Queue()
        
        LogReader = get_log_reader_class()
        reader = LogReader(log_path, "stderr", log_queue)
        keyboard = KeyboardReader()
        
        process_stats = None
        if process_pid:
            try:
                process_stats = ProcessStats(process_pid)
            except psutil.NoSuchProcess:
                console.print("[red]Process not found.[/red]")
                return False

        reader.start()
        keyboard.start()

        with Live(
            auto_refresh=True,
            refresh_per_second=4,
            vertical_overflow="visible",
            screen=True,
            console=Console(force_terminal=True)
        ) as live:
            try:
                while True:
                    # Handle keyboard input for scrolling
                    if keyboard.kbhit():
                        key = keyboard.getch()
                        # Handle both Windows and Unix key codes
                        if key in (b'H', b'A'):  # Up arrow (Windows: H, Unix: A)
                            format_logs.scroll_position = max(0, format_logs.scroll_position - 1)
                        elif key in (b'P', b'B'):  # Down arrow (Windows: P, Unix: B)
                            format_logs.scroll_position = format_logs.scroll_position + 1

                    while not log_queue.empty():
                        _, line, is_new = log_queue.get_nowait()
                        log_lines.append((line, is_new))

                    layout = Layout()
                    
                    if process_stats:
                        layout.split_column(
                            Layout(process_stats.get_status(), size=10),
                            Layout(format_logs(log_lines))
                        )
                    else:
                        layout.update(format_logs(log_lines))
                    
                    live.update(layout)
                    time.sleep(0.1)  # Shorter sleep for more responsive scrolling

            except KeyboardInterrupt:
                console.print("\n[yellow]Monitoring stopped.[/yellow]")
            finally:
                reader.stop()
                keyboard.stop()

    except Exception as e:
        console.print(f"[red]Monitor failed: {e}[/red]")
        console.print(f"[red]Error details: {str(e)}[/red]")
        return False

def get_compute_subnet_pid():
    try:
        for proc in psutil.process_iter(['pid', 'cmdline']):
            try:
                cmdline = proc.cmdline()
                if len(cmdline) > 1 and 'compute_subnet' in ' '.join(cmdline):
                    return proc.pid
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return None
    except Exception as e:
        console.print(f"[red]Error finding process: {e}[/red]")
        return None

def check_main():
    try:
        console.clear()
        console.print("[cyan]🔍 Looking for compute subnet process...[/cyan]")
        pid = get_compute_subnet_pid()
        
        if pid:
            console.print(f"[green]✓ Found compute subnet process (PID: {pid})[/green]")
            console.print("[cyan]📊 Starting monitor...[/cyan]")
            monitor_logs(pid)
        else:
            console.print("[yellow]⚠ Compute subnet process not found. Monitoring logs only...[/yellow]")
            monitor_logs(None)
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user.[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        sys.exit(1)

if __name__ == "__main__":
    check_main()
