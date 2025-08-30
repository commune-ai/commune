import subprocess

commands = {
    "black": "black .",
    "ruff": "ruff check .",
    "isort": "isort .",
    "mypy": "mypy .",
}

if __name__ == "__main__":
    results = []
    for command in commands.values():
        command = command.split(" ")
        result = subprocess.run(command, capture_output=True)
        results.append(result)

    for result in results:
        print(result.stdout.decode("utf-8"))
        print(result.stderr.decode("utf-8"))
