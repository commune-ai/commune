import commune as c


class LiteLLM(c.Module):

    def __init__(self, bot):
        super().__init__(bot)
        self.name = "LiteLLM"
        self.description = "A lightweight LLM module"
        self.version = "0.0.1"

    async def on_message(self, message):
        if message.content.startswith("!llm"):
            await message.channel.send("LiteLLM is working!")