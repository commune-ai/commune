import atexit
import subprocess
import datetime
import os

def start(delay_days):
    # Get the absolute path of the current working directory
    current_directory = os.getcwd()

    file_path = "./neurons/Miner/kill_container"

    # Calculate the future time
    current_datetime = datetime.datetime.now()
    future_datetime = current_datetime + datetime.timedelta(days=delay_days)

    # A function to schedule the command using 'at'
    def schedule_command():
        # Get a list of all job numbers
        job_numbers = subprocess.check_output(["atq"], text=True).splitlines()

        formatted_time = future_datetime.strftime("%H:%M %m/%d/%Y")

        # Remove each job
        for job_number in job_numbers:
            # Extract the job number (the first column)
            job_number = job_number.split()[0]
            subprocess.run(["atrm", job_number], check=True)

        subprocess.run(['at', formatted_time], input=f'{file_path}\n', text=True, check=True)

    # Register the function to be executed when the script exits
    schedule_command()