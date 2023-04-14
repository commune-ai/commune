
import ray

ray.init()

@ray.remote
class MyActor:
    def __init__(self):
        pass
    
    def do_work(self):
        print("Doing some work...")
        
my_actor = MyActor.remote()

# Get a handle to the actor
actor_handle = ray.get_actor("MyActor")

# Retrieve the actor's logs
logs = ray.get_actor_logs(actor_handle)

# Print the logs
for log in logs:
    print(log)