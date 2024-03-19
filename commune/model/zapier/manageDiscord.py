import discord
from discord.ext import tasks, commands
import asyncio
from .config import userEmail, password, imap_host, imap_port, mail_channel_id, token
# Discord bot token
TOKEN = 'your-bot-token-here'

# Discord client setup
client = commands.Bot(command_prefix="!")

# List of channel IDs to send scheduled messages
CHANNEL_IDS = mail_channel_id  # Replace with your channel IDs

# Scheduled message and interval (in seconds)
SCHEDULED_MESSAGE = 'This is a scheduled message!'
MESSAGE_INTERVAL = 3600  # 1 hour

@client.event
async def on_ready():
    print(f'Logged in as {client.user}')
    send_scheduled_message.start()  # Start the scheduled message task

@client.command(name='hello')
async def hello(ctx):
    await ctx.send('Hello there!')

@tasks.loop(seconds=MESSAGE_INTERVAL)
async def send_scheduled_message():
    for channel_id in CHANNEL_IDS:
        channel = client.get_channel(channel_id)
        if channel:
            await channel.send(SCHEDULED_MESSAGE)

@client.event
async def on_message(message):
    # Prevent bot from responding to its own messages
    if message.author == client.user:
        return

    # Respond to certain keywords
    if 'ping' in message.content.lower():
        await message.channel.send('Pong!')

    # Process commands
    await client.process_commands(message)

@client.event
async def on_member_join(member):
    await member.guild.system_channel.send(f'Welcome {member.name} to {member.guild.name}!')

@client.event
async def on_member_remove(member):
    await member.guild.system_channel.send(f'Goodbye {member.name}. We will miss you!')

@client.command(name='purge')
@commands.has_permissions(manage_messages=True)
async def purge(ctx, num: int):
    await ctx.channel.purge(limit=num)

@client.command(name='userinfo')
async def userinfo(ctx, member: discord.Member):
    embed = discord.Embed(title="User Information", color=0x3498db)
    embed.add_field(name="Username", value=member.name, inline=False)
    embed.add_field(name="ID", value=member.id, inline=False)
    await ctx.send(embed=embed)

@client.command(name='status')
async def status(ctx, *, text: str):
    await client.change_presence(activity=discord.Game(name=text))

@client.command(name='createchannel')
@commands.has_permissions(manage_channels=True)
async def createchannel(ctx, channel_name: str):
    guild = ctx.guild
    await guild.create_text_channel(channel_name)

@client.command(name='listchannels')
async def listchannels(ctx):
    channels = ctx.guild.channels
    channel_names = [channel.name for channel in channels if isinstance(channel, discord.TextChannel)]
    await ctx.send("Channels:\n" + "\n".join(channel_names))

@client.command(name='echo')
async def echo(ctx, *, message: str):
    await ctx.send(message)

@client.command(name='dm')
async def dm(ctx, member: discord.Member, *, message: str):
    await member.send(message)

@client.command(name='announce')
@commands.has_permissions(administrator=True)
async def announce(ctx, *, message: str):
    for channel in ctx.guild.text_channels:
        await channel.send(message)

client.run(TOKEN)