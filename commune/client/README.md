

To use the clisent


c serve moduel::fam

{'success': True, 'name': 'module::fam', 'address': '198.161.104.67:50236', 'kwargs': {}, 'module': 'module'}



To call the function you need to use the address and the name of the function



c call 198.161.104.67:50236/name
🔵🔵🔵 Calling module/call/{"args": ["198.161.104.67:50236/name"], "kwargs": {}} 🔵🔵🔵
🟢🟢🟢 Result (0.12) 🟢🟢🟢
module::fam

or

c.call("198.161.104.67:50236/name")

or 

c.call("module::test/name")

🟢🟢🟢 Result (0.12) 🟢🟢🟢
module::fam




