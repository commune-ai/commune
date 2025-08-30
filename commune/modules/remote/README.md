Remote Class for Commune
Welcome! Imagine you have a magic wand that lets you talk to computers far away as if they were right in front of you. The Remote class is like that magic wand for adults, letting them send messages (we call them "commands") to computers that are not next to them. It's like sending a letter to a friend who lives in another city, but much, much faster!

How It Works
Our magic wand, the Remote class, can do several fun things:

Send Commands: Tell a faraway computer to do something, like showing all the pictures it has.
Remember Computers: Keep a list of faraway computers so we don't forget how to reach them.
Add New Friends: If we meet a new computer, we can add it to our list so we can talk to it later.
How to Use the Magic Wand
1. Adding a New Computer Friend
If you meet a new computer and want to talk to it later, you need to remember its name, where it lives, and a secret password so it knows you're a friend. Adults write this down in a special way:

Remote.add_host_from_ssh_string("vali@0.0.0.0:940 -p 22 -pwd your_password", name="my_computer_friend")
Replit

This is like saying, "I have a friend named 'my_computer_friend'. They live at house number 0.0.0.0, on street 940, and their secret password is 'your_password'."

2. Sending a Command to a Computer
Imagine you want to ask your computer friend to show you all the toys they have. Adults do this by sending a command:

toys_list = Remote.ssh_cmd("ls", host="my_computer_friend", verbose=True)
Replit

This is like asking, "Can you list all your toys, please?"

3. Remembering All Your Computer Friends
If you have many computer friends and you want to see a list of all their names, you can do this:

all_my_friends = Remote.names()
print(all_my_friends)
Replit

This will show you the names of all the computers you can talk to.

Safety First!
Remember, just like you shouldn't tell strangers your home address or your secrets, you should keep your computer friends' information safe and not share their secret passwords with anyone you don't trust.

Conclusion
Using the Remote class is like having a magic wand that lets you talk to computers no matter where they are. You can ask them to do things for you, remember new computer friends, and make sure all your computer friends are safe and happy.

This README.md aims to simplify the explanation of the Remote class functionalities and usage, making it accessible to beginners or even to a younger audience. Adjust the content as necessary to better fit your project or audience.


APP

to run the app 

c app remote 
or 
c app r
