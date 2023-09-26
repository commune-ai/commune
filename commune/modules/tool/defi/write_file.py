import commune as c 
class WriteFile(c.Module):
    description ="""Write to a file"""
    
    def call(self, file_path, content):
        """Write to a file
        """
        try:
            with open(file_path, 'w') as file:
                file.write(content)
            print(f"Content successfully written to {file_path}")
        except Exception as e:
            print(f"Error: An error occurred while writing to the file: {e}")

