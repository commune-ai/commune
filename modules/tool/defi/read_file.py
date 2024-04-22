import commune as c
class ReadFile(c.Module):
    description = """"Read from a file"""
    def call(self, file_path):
        """Read from a file"""
        try:
            with open(file_path, 'r') as file:
                content = file.read()
            return content
        except FileNotFoundError:
            print(f"Error: File not found: {file_path}")
        except Exception as e:
            print(f"Error: An error occurred while reading the file: {e}")
