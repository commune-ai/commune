
import commune as c

docs = c.file2text("./docs")
c.print(c.ask(f"whats is commune? {docs}"))