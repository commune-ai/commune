import requests
import json
import sys

def test_container_api():
    # API base URL
    base_url = "https://polaris-test-server.onrender.com"
    
    # Your Windows machine's SSH public key
    public_key = """ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQC3IZ+cVHfjx5/Dp3o+xMBoShF+Oy4ioTW8pGY3C72S21gp237QtQM+TOU1KONHWflSYxBE390q9AUg9oEfNEqJBuwVCtgKzLWMXeIXGUAlt9wvYDzS41LVnAJpm3arV5jD3LjEuf54PbG6WF267L13nyHayAqA6SL+IdCqsmqvjqne8t9GWryz+016qKcwKSXPbzhZH5ytgHFlW9+bA7LFq3OxWKsoYg81L2hZxsdHO03w5Nv6p1rMoR4YKW7LPY+rf/kybKD7s6MGmQ7Sas7eRIIRDoxLC9b/O+i6kz0EVVuYB8B0JaSz1ROHvcRw+JWHHD09swXsUmp6t6JXzHvAq7WjQSiCoHZW1aPtsV7CjaDkMCWRc2RSYvdzjpr6VKbRZo5n2Bp6vNUi4+3m05eQ5Zd6glor2oDU/5FFOwfR9WBvsTs0lVjttgSwMOFkG6jrD6byrhH35JKhiFe1TrMGN2DdcShSTmwQlzqBBsUbbpU9wP8tRWW5uhZ508CrOQ3UtJsLhnod3+FMFALEEWbvsxh3AVG2FQ2yeidymGyJ/lr0MNvIzFajzBOA/4SYw474WSjHro+fPojZ7DL6covY7FjXEut8M/OQAF6Sohb8MSw9Gz40vSnx0FljAZZnDd19GCtkW+F4zdCyY2XxB1sSy5v6dE3eskvHg2uYQScQHQ== hp@BANADDA"""

    try:
        # Create a container
        print("Creating container...")
        response = requests.post(
            f"{base_url}/containers",
            json={"public_key": public_key}
        )
        
        if response.status_code == 200:
            container_info = response.json()
            print("\nContainer created successfully!")
            print(f"Container ID: {container_info['container_id']}")
            print(f"SSH Port: {container_info['ssh_port']}")
            print(f"Username: {container_info['username']}")
            print(f"Host: {container_info['host']}")
            print("\nFrom your Windows machine, connect using:")
            print(f"ssh -i C:/Users/HP/.ssh/id_rsa -p {container_info['ssh_port']} {container_info['username']}@{container_info['host']}")
            
            # Ask if user wants to terminate the container
            input("\nPress Enter to terminate the container...")
            
            # Terminate container
            print(f"\nTerminating container {container_info['container_id']}...")
            response = requests.delete(
                f"{base_url}/containers/{container_info['container_id']}"
            )
            
            if response.status_code == 200:
                print("Container terminated successfully!")
            else:
                print(f"Error terminating container: {response.text}")
                
        else:
            print(f"Error creating container: {response.text}")
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_container_api()