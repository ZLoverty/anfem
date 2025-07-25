# monitor_simulations.py
import os
import glob
import time
from pathlib import Path

STATUS_DIR = "~/Documents/RATSIM"
STATUS_DIR = Path(STATUS_DIR).expanduser().resolve()
REFRESH_INTERVAL = 5 # seconds (how often the display updates)
BAR_LENGTH = 30    # Length of the progress bar itself

def clear_terminal():
    """Clears the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def get_progress_bar(current, total, bar_length=BAR_LENGTH):
    """Generates an ASCII progress bar string."""
    if total <= 0:
        return "N/A" # Avoid division by zero
    
    percent = (current / total) * 100
    filled_len = int(bar_length * current // total)
    bar = '█' * filled_len + '-' * (bar_length - filled_len) # Use solid block for filled part
    return f"[{bar}] {percent:5.1f}%"

def display_simulation_status():
    """Reads all status files and prints the dashboard."""
    clear_terminal()
    print("--- Simulation Progress Dashboard ---")
    print("-" * (BAR_LENGTH + 40)) # Adjust header length

    status_files = glob.glob(str(STATUS_DIR / "**" / "anfem.log"), recursive=True)

    if not status_files:
        print("No simulation status files found yet.")
        print("Waiting for simulations to start...")
        print("-" * (BAR_LENGTH + 40))
        return

    # Sort files by SIM_ID for consistent display order
    status_files.sort()

    active_sims = 0
    for status_file in status_files:
        status_file = Path(status_file)
        sim_id = str(status_file.parent).replace(str(STATUS_DIR), "")
        
        try:
            # Read the entire content to get the last update, or just the first line if single line
            with open(status_file, 'r') as f:
                status_line = f.readlines()[-1].strip() # Read the whole (single) line
            
            current_step = 0
            total_steps = 0
            status_message = "Pending..."
            progress_bar_str = get_progress_bar(0, 100) # Default to empty bar

            try:
                status_message = f"{status_line[:20]}"
                parts = status_line.split(" - ")[2].strip().split(",")[0].split('/')
                current_step = int(parts[0])
                total_steps = int(parts[1])
                progress_bar_str = get_progress_bar(current_step, total_steps)
                print(f"{sim_id: <30} {progress_bar_str} {status_message}")
                active_sims += 1
                
            except (IndexError, ValueError):
                status_message = f"Invalid PROGRESS format: {status_line[:20]}..."

        except FileNotFoundError:
            print(f"{sim_id: <30} [ File missing ] May have just completed or crashed.")
        except Exception as e:
            print(f"{sim_id: <30} [ Read Error ] {e}")
    
    print("-" * (BAR_LENGTH + 40))
    print(f"Simulations Running: {active_sims}. Last updated: {time.strftime('%H:%M:%S')}. Press Ctrl+C to exit.")

if __name__ == "__main__":
    if not os.path.exists(STATUS_DIR):
        print(f"Status directory '{STATUS_DIR}' not found. Please ensure your simulations create it.")
        # Attempt to create it anyway, in case launcher hasn't run yet or for manual testing
        os.makedirs(STATUS_DIR, exist_ok=True)
        # sys.exit(1) # Don't exit immediately, allow it to wait for files

    try:
        while True:
            display_simulation_status()
            time.sleep(REFRESH_INTERVAL)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
    finally:
        # Optional: Clean up status files on exit
        # This might be undesirable if you want to inspect final status after closing monitor
        # for f in glob.glob(os.path.join(STATUS_DIR, "status_*.txt")):
        #     os.remove(f)
        # for f in glob.glob(os.path.join(STATUS_DIR, "status_*.tmp")):
        #     os.remove(f)
        pass