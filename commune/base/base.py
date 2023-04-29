import commune

class Base(commune.Module):
    def __init__(self, **kwargs):
        config = self.set_config(config=kwargs)
        self.setup(config)
        
    def setup(self, config):
        self.add_users(config.add_users)
        

    def add_user(self, user: str, secret: str) -> None:
        self.config.users.append(user)
        return self.config.users[-1]
        
    def rm_user(self, user: str) -> None:
        for i in range(len(self.config.users)):
            if self.config.users[i] == user:
                del self.config.users[i]
                
                
    def save(self):
        self.put('config',self.config)
        
    def load(self):
        self.config = self.get('config')
        

    def users(self):
        return self.config.users
    
if __name__ == '__main__':
    Base.run()