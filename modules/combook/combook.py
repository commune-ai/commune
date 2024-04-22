import commune as c

class Combook(c.Module):
    def __init__(self, max_people=1000):
        self.set_config(kwargs=locals())

    

    def send(self, text, chatroom='lobby', password=None ) -> int:
        password_hash = password
        timestamp = c.time()
        path = f'{chatroom}/TS::{timestamp}_MODULE::{user_address}'
        chat_info = {
            'user': user_address,
            'timestamp', timestamp,
            'text': text,
            'password': password
        }

        self.put(path, chat_info, password=password)

