import commune as c
import discord
from discord.ext import commands
from web3 import Web3, Account
import os
import json
import asyncio
import random
from typing import Any
import time
import requests
import nest_asyncio



class OTC(c.Module):
    def __init__(self,
             key_path : str = "/root/.commune/key",
             usdt_decimals : int = 6,
             # addys
             ethereum_address : str = "", # specify
             abi =[ {"constant":True,"inputs":[],"name":"name","outputs":[{"name":"","type":"string"}],"payable":False,"stateMutability":"view","type":"function"},{"constant":False,"inputs":[{"name":"_upgradedAddress","type":"address"}],"name":"deprecate","outputs":[],"payable":False,"stateMutability":"nonpayable","type":"function"},{"constant":False,"inputs":[{"name":"_spender","type":"address"},{"name":"_value","type":"uint256"}],"name":"approve","outputs":[],"payable":False,"stateMutability":"nonpayable","type":"function"},{"constant":True,"inputs":[],"name":"deprecated","outputs":[{"name":"","type":"bool"}],"payable":False,"stateMutability":"view","type":"function"},{"constant":False,"inputs":[{"name":"_evilUser","type":"address"}],"name":"addBlackList","outputs":[],"payable":False,"stateMutability":"nonpayable","type":"function"},{"constant":True,"inputs":[],"name":"totalSupply","outputs":[{"name":"","type":"uint256"}],"payable":False,"stateMutability":"view","type":"function"},{"constant":False,"inputs":[{"name":"_from","type":"address"},{"name":"_to","type":"address"},{"name":"_value","type":"uint256"}],"name":"transferFrom","outputs":[],"payable":False,"stateMutability":"nonpayable","type":"function"},{"constant":True,"inputs":[],"name":"upgradedAddress","outputs":[{"name":"","type":"address"}],"payable":False,"stateMutability":"view","type":"function"},{"constant":True,"inputs":[{"name":"","type":"address"}],"name":"balances","outputs":[{"name":"","type":"uint256"}],"payable":False,"stateMutability":"view","type":"function"},{"constant":True,"inputs":[],"name":"decimals","outputs":[{"name":"","type":"uint256"}],"payable":False,"stateMutability":"view","type":"function"},{"constant":True,"inputs":[],"name":"maximumFee","outputs":[{"name":"","type":"uint256"}],"payable":False,"stateMutability":"view","type":"function"},{"constant":True,"inputs":[],"name":"_totalSupply","outputs":[{"name":"","type":"uint256"}],"payable":False,"stateMutability":"view","type":"function"},{"constant":False,"inputs":[],"name":"unpause","outputs":[],"payable":False,"stateMutability":"nonpayable","type":"function"},{"constant":True,"inputs":[{"name":"_maker","type":"address"}],"name":"getBlackListStatus","outputs":[{"name":"","type":"bool"}],"payable":False,"stateMutability":"view","type":"function"},{"constant":True,"inputs":[{"name":"","type":"address"},{"name":"","type":"address"}],"name":"allowed","outputs":[{"name":"","type":"uint256"}],"payable":False,"stateMutability":"view","type":"function"},{"constant":True,"inputs":[],"name":"paused","outputs":[{"name":"","type":"bool"}],"payable":False,"stateMutability":"view","type":"function"},{"constant":True,"inputs":[{"name":"who","type":"address"}],"name":"balanceOf","outputs":[{"name":"","type":"uint256"}],"payable":False,"stateMutability":"view","type":"function"},{"constant":False,"inputs":[],"name":"pause","outputs":[],"payable":False,"stateMutability":"nonpayable","type":"function"},{"constant":True,"inputs":[],"name":"getOwner","outputs":[{"name":"","type":"address"}],"payable":False,"stateMutability":"view","type":"function"},{"constant":True,"inputs":[],"name":"owner","outputs":[{"name":"","type":"address"}],"payable":False,"stateMutability":"view","type":"function"},{"constant":True,"inputs":[],"name":"symbol","outputs":[{"name":"","type":"string"}],"payable":False,"stateMutability":"view","type":"function"},{"constant":False,"inputs":[{"name":"_to","type":"address"},{"name":"_value","type":"uint256"}],"name":"transfer","outputs":[],"payable":False,"stateMutability":"nonpayable","type":"function"},{"constant":False,"inputs":[{"name":"newBasisPoints","type":"uint256"},{"name":"newMaxFee","type":"uint256"}],"name":"setParams","outputs":[],"payable":False,"stateMutability":"nonpayable","type":"function"},{"constant":False,"inputs":[{"name":"amount","type":"uint256"}],"name":"issue","outputs":[],"payable":False,"stateMutability":"nonpayable","type":"function"},{"constant":False,"inputs":[{"name":"amount","type":"uint256"}],"name":"redeem","outputs":[],"payable":False,"stateMutability":"nonpayable","type":"function"},{"constant":True,"inputs":[{"name":"_owner","type":"address"},{"name":"_spender","type":"address"}],"name":"allowance","outputs":[{"name":"remaining","type":"uint256"}],"payable":False,"stateMutability":"view","type":"function"},{"constant":True,"inputs":[],"name":"basisPointsRate","outputs":[{"name":"","type":"uint256"}],"payable":False,"stateMutability":"view","type":"function"},{"constant":True,"inputs":[{"name":"","type":"address"}],"name":"isBlackListed","outputs":[{"name":"","type":"bool"}],"payable":False,"stateMutability":"view","type":"function"},{"constant":False,"inputs":[{"name":"_clearedUser","type":"address"}],"name":"removeBlackList","outputs":[],"payable":False,"stateMutability":"nonpayable","type":"function"},{"constant":True,"inputs":[],"name":"MAX_UINT","outputs":[{"name":"","type":"uint256"}],"payable":False,"stateMutability":"view","type":"function"},{"constant":False,"inputs":[{"name":"newOwner","type":"address"}],"name":"transferOwnership","outputs":[],"payable":False,"stateMutability":"nonpayable","type":"function"},{"constant":False,"inputs":[{"name":"_blackListedUser","type":"address"}],"name":"destroyBlackFunds","outputs":[],"payable":False,"stateMutability":"nonpayable","type":"function"},{"inputs":[{"name":"_initialSupply","type":"uint256"},{"name":"_name","type":"string"},{"name":"_symbol","type":"string"},{"name":"_decimals","type":"uint256"}],"payable":False,"stateMutability":"nonpayable","type":"constructor"},{"anonymous":False,"inputs":[{"indexed":False,"name":"amount","type":"uint256"}],"name":"Issue","type":"event"},{"anonymous":False,"inputs":[{"indexed":False,"name":"amount","type":"uint256"}],"name":"Redeem","type":"event"},{"anonymous":False,"inputs":[{"indexed":False,"name":"newAddress","type":"address"}],"name":"Deprecate","type":"event"},{"anonymous":False,"inputs":[{"indexed":False,"name":"feeBasisPoints","type":"uint256"},{"indexed":False,"name":"maxFee","type":"uint256"}],"name":"Params","type":"event"},{"anonymous":False,"inputs":[{"indexed":False,"name":"_blackListedUser","type":"address"},{"indexed":False,"name":"_balance","type":"uint256"}],"name":"DestroyedBlackFunds","type":"event"},{"anonymous":False,"inputs":[{"indexed":False,"name":"_user","type":"address"}],"name":"AddedBlackList","type":"event"},{"anonymous":False,"inputs":[{"indexed":False,"name":"_user","type":"address"}],"name":"RemovedBlackList","type":"event"},{"anonymous":False,"inputs":[{"indexed":True,"name":"owner","type":"address"},{"indexed":True,"name":"spender","type":"address"},{"indexed":False,"name":"value","type":"uint256"}],"name":"Approval","type":"event"},{"anonymous":False,"inputs":[{"indexed":True,"name":"from","type":"address"},{"indexed":True,"name":"to","type":"address"},{"indexed":False,"name":"value","type":"uint256"}],"name":"Transfer","type":"event"},{"anonymous":False,"inputs":[],"name":"Pause","type":"event"},{"anonymous":False,"inputs":[],"name":"Unpause","type":"event"}],
             private_key = "",   # specify
             usdt_contract_address : str = "0xdAC17F958D2ee523a2206206994597C13D831ec7",
             max_channels = 10,  # Max open trades channels
             user_channels : dict = {},
             buyers_data : dict  = {},
             sellers_data : dict  = {},
             time_data : dict  = {},
             command_status : dict  = {},
             # discord
             my_wallet_address ='x', # specify
             bot_commands_channel_id : int = 0, # specify
             bot_otc_category_id : int = 0,# specify
             bot_token : str  = "x", # specify
             ) -> None : 
        self.key_path = key_path
        # USDT has 6 decimals
        self.USDT_DECIMALS = usdt_decimals 
        self.ethereum_address = ethereum_address
        self.abi = abi
        self.private_key= private_key
        self.max_channels = max_channels
        self.user_channels = user_channels
        self.buyers_data = buyers_data
        self.sellers_data = sellers_data
        self.time_data = time_data
        self.command_status= command_status
        self.USDT_CONTRACT_ADDRESS = usdt_contract_address
        # NEED Chanel ID for bot command server
        self.BOT_COMMANDS_CHANNEL_ID = bot_commands_channel_id 
        # NEED Catrgory_ID for OTC Chanels
        self.BOT_OTC_CATEGORY_ID = bot_otc_category_id
        self.my_wallet_address = my_wallet_address
        self.BOTTOKEN = bot_token
         # == settings
        self.intents = discord.Intents.default()
        self.intents.typing  = True
        self.intents.presences = True        
        self.bot = commands.Bot(command_prefix='!', intents=discord.Intents.all())
        self.permissions = discord.Permissions().all()



        
        
        def check_channel(channel_id):
            async def predicate(ctx):
                if ctx.message.channel.id == channel_id:
                    return True
                else:
                    await ctx.send("You aren't in the correct channel")
                    return False
            return commands.check(predicate)


        def check_category(category_id):
            async def predicate(ctx):
                category = ctx.guild.get_channel(category_id)
                if isinstance(ctx.channel, discord.TextChannel) and ctx.channel.category == category:
                    return True
                else:
                    await ctx.send("You aren't in the correct category")
                    return False
            return commands.check(predicate)
        
        # == commune
        async def balance(wallet: str) -> float:
            return c.balance(wallet)

        async def transfer(sender: str, receiver: str, amount: float) -> bool | None:
            return c.transfer(key=sender, amount=amount, dest=receiver)
            
            
        def add_key(new_tag):
            return c.add_key(new_tag)
            
        def tag() -> str:
            import string
            letters = string.ascii_lowercase
            return ''.join(random.choice(letters) for _ in range(10))

        # returns address of the key
        def router_key() -> dict:
            new_tag = tag()
            filenames = os.listdir(self.key_path)
            filenames = [filename.replace('.json', '') for filename in filenames]
            #print(filenames)
            while new_tag in filenames:
                new_tag = tag()
            add_key(new_tag)
            print ("created key")

            key_info = f"{self.key_path}/{new_tag}.json"
            
            with open(key_info) as f:
                #print (f"key info is {key_info}")
                data = json.load(f)
                #print(data)
                data_field = json.loads(data['data'])
                #print (data_field)

                address = str(data_field['ss58_address'])
                key_name= str(data_field['path'])
                dictionary={'address': address, 'key_name': key_name}
            return dictionary

        async def manage_transaction(expected_amount:float, address:str) -> bool:
            amount = await balance(address)
            lower_bound = expected_amount * 0.998
            if amount >= lower_bound:
                return True
            else: 
                return False
            

        def calculate_fee_refund_com(transaction_value):
            return transaction_value -4

        async def info_about_fee(ctx, usdt):
            if usdt <= 500:
                await ctx.send(f"Your transaction value is below 500 USDT, so the fee will be fixed at 5 USDT + {info_transaction_usdt_fee()} USDT for gas on etherum")
            elif usdt <= 1000:
                await ctx.send(f"Your transaction value is below 1000 USDT, so the fee will be at 1% + {info_transaction_usdt_fee()} USDT for gas on etherum")
            elif usdt <= 8000:
                await ctx.send(f"Your transaction value is below 8000 USDT, so the fee will be at 0.8% + {info_transaction_usdt_fee()} USDT for gas on etherum")
            elif usdt >= 8000:
                await ctx.send(f"Your transaction value is over 8000 USDT, so the fee will be at 0.5% + {info_transaction_usdt_fee()} USDT for gas on etherum")

        def info_transaction_usdt_fee():
            gas_price_in_wei = get_gas_price()
            if gas_price_in_wei <= 30.0:
                return 1
            elif gas_price_in_wei<=60:
                return 3
            elif gas_price_in_wei<= 90:
                return 7
            elif gas_price_in_wei>90:
                return 10
            
        def get_gas_price():
            api_key = "fill-in"
            base_url = f"https://eth-mainnet.g.alchemy.com/v2/{api_key}"
            
            payload = {
                "jsonrpc": "2.0",
                "method": "eth_gasPrice",
                "params": [],
                "id": 1
            }
            
            try:
                response = requests.post(base_url, json=payload)
                data = response.json()

                if response.status_code == 200:
                    gas_price = int(data['result'], 16)  # Convert hex to int
                    gas_price= gas_price/10**9
                    gas_price=round(gas_price,0)
                    float(gas_price)
                    return gas_price
                else:
                    print(f"Chyba při získávání dat: {data['message']}")
            except Exception as e:
                print(f"Chyba: {e}")

        def calculate_fee_refund(transaction_value):
            gwei=get_gas_price()
            if gwei <= 30.0:
                return transaction_value - 6
            elif gwei<=60:
                return transaction_value-8
            elif gwei<= 90:
                return transaction_value-12
            elif gwei>90:
                return transaction_value-15
            
        def calculate_fee(transaction_value,  amount):
            gwei=get_gas_price()
            if gwei <= 30.0:
                amount -= 1
            elif gwei<=60:
                amount-=3
            elif gwei<= 90:
                amount-=7
            elif gwei>90:
                amount-=10

            if transaction_value <= 500:
                trans_round = amount - 5.0 # fixed fee below 500 USDT is 5 USDT
            elif 500 < transaction_value <= 1000:
                trans_round= amount * 0.99  # 99% fee
            elif transaction_value >= 8000:
                trans_round = amount * 0.995 # 99.5% fee
            else:
                trans_round = amount * 0.992 # 99.2% fee
            return round(trans_round, 2)
                    
        def calculate_fee_com(transaction_value,  amount):
            
            if transaction_value <= 500:# fixed fee below 500 USDT is 5 USDT
                full_value = transaction_value
                transaction_value= transaction_value-5
                clean_value_procent = transaction_value / full_value
                trans_round = amount * clean_value_procent
                return round(trans_round, 2)
            elif 500 < transaction_value <= 1000:
                trans_round= amount * 0.99  # 99% fee
                return round(trans_round, 2)
            elif transaction_value >= 8000:
                trans_round = amount * 0.995 # 99.5% fee
                return round(trans_round, 2)
            else:
                trans_round = amount * 0.992 # 99.2% fee
                return round(trans_round, 2)
        
        def confirm_deposit_ethereum(address, expected_amount):
            web3 = Web3(Web3.HTTPProvider("https://eth-mainnet.g.alchemy.com/v2/fill-in"))
            contract_abi = abi
            contract_address = usdt_contract_address
            contract = web3.eth.contract(address=contract_address, abi=contract_abi) # type: ignore
            event_filter = contract.events.Transfer.create_filter(fromBlock='latest', argument_filters={'to': address})  
            for event in event_filter.get_all_entries():
                amount_wei = event['args']['value']
                amount_usdt = amount_wei / 10**6

                if expected_amount * 0.998 <= amount_usdt <= expected_amount * 1.02:
                    return True
                else:
                    return False

        async def end_transaction(ctx, buyer_data, seller_data, support):
            await ctx.send("Deposits are okay. Processing further actions.")
            await ctx.send("Deposits are ok, will be sent in 3, 2, 1...")
            
            usdt = calculate_fee(seller_data['usdt'], seller_data['usdt'])
            com_amount = calculate_fee_com(seller_data['usdt'], buyer_data['com_amount'])
            
            send_usdt_transaction(address= seller_data['address'], amount=usdt) # calculate fee on ETH and send
            
            await ctx.send(f"USDT balance was sent on {seller_data['address']}")

            #print(seller_data['key_name'], buyer_data['address'], com_amount)
            await transfer(sender =seller_data['key_name'], receiver = buyer_data['address'], amount = com_amount) # calculate fee on Com and send
            
            await ctx.send(f"COM balance was sent on {buyer_data['address']}")

            await support.send(f"OTC transfer in cahnnel: {ctx.channel.id} was success realized.")
            all_ballance = await balance(seller_data['commune_add'])-0.3
            await transfer(sender=seller_data['key_name'], receiver = self.my_wallet_address, amount= all_ballance)
            await ctx.send(f"""
            The trade was made {com_amount} COM for {usdt} USDT at a price {calculate_price(com_amount, usdt)}COM/USDT
            Channel going to destroy in next 3 minutes.
            """)
            await asyncio.sleep(180)
            del self.time_data[ctx.channel.id]
            del self.buyers_data[ctx.channel.id]
            del self.sellers_data[ctx.channel.id]
            await ctx.channel.delete()

        def send_usdt_transaction(address, amount):
            def optimalize():
                # Setting up connection to the Ethereum network
                ALCHEMY_API_URL = 'https://eth-mainnet.g.alchemy.com/v2/fill-in'
                web3 = Web3(Web3.HTTPProvider(ALCHEMY_API_URL))

                # Creating sender's address from private key
                sender_address = Account.from_key(private_key).address

                # USDT contract address and ABI

                USDT_CONTRACT_ABI = abi
                contract = web3.eth.contract(address=usdt_contract_address, abi=USDT_CONTRACT_ABI)

                # Defining the amount of USDT to send
                amount_in_base_units = int(amount *10**6)

                # Building and signing the transaction
                nonce = web3.eth.get_transaction_count(sender_address)
                gasPrice = web3.eth.gas_price
                transaction = contract.functions.transfer(address, amount_in_base_units).build_transaction({
                    'gas': 200_000,
                    'gasPrice': gasPrice,
                    'nonce': nonce,
                    'chainId': 1, # 1 represents the mainnet. For other networks, use the appropriate network ID.
                })
                signed_transaction = web3.eth.account.sign_transaction(transaction, private_key)

                # Sending the signed transaction
                tx_hash = web3.eth.send_raw_transaction(signed_transaction.rawTransaction)

                # Tracking the transaction status
                tx_receipt = web3.eth.wait_for_transaction_receipt(tx_hash, 600)  # Nastavte hodnotu timeout v sekundách
                #print(tx_hash)
                #print(tx_receipt)
            return optimalize()

        def calculate_price(comm_amount, usdt_amount): # return price but in float
            price = comm_amount / usdt_amount
            rounded_price = round(price, 3)
            return rounded_price


        # == discord
        @self.bot.event
        async def on_ready() -> Any:
            print(f'signed as {self.bot.user.name}')


        @self.bot.command(name='OTC', help='create privite room for OTC')
        @check_channel(self.BOT_COMMANDS_CHANNEL_ID)
        async def OTC(ctx, user: discord.User):
            # Check if the maximum number of channels is reached
            category = discord.utils.get(ctx.guild.categories, name="OTC Channels")
            # wait 1 second and then delete the user's message
            await ctx.message.delete()
            if category is not None and len(category.channels) >= max_channels:
                # Send a message from the bot
                await ctx.send("OTC services are currently fully occupied, please try again in a moment.")
                return
            # Check if the user already has an active room
            if ctx.author.id in user_channels:
                await ctx.send("You already have an active OTC room.")
                return
            
            # Create a new text channel named "OTC"
            channel_name = "OTC"
            guild = ctx.guild
            
            # Check if the category exists, if not, create it
            if category is None:
                category = await guild.create_category("OTC Channels")
            
            # Create a text channel in the "OTC Channels" category
            overwrites = {
                guild.default_role: discord.PermissionOverwrite(read_messages=False),
                ctx.author: discord.PermissionOverwrite(read_messages=True),
                user: discord.PermissionOverwrite(read_messages=True)
            }
            channel = await guild.create_text_channel(channel_name, category=category, overwrites=overwrites)
            # Send a message to the user about successful connection to the OTC channel
            await ctx.send("A text room has been SUCCESSFULLY created for you and your partner in the category OTC channels.")
            
            

            # Send a message about possible deletion of the channel within 5 minutes
            message = await channel.send(f"{user.mention} agrees to create an OTC textroom, if so, confirm this message with a reaction :white_check_mark:")
            
            # Add a reaction to the message
            await message.add_reaction("✅")

            def check_user(reaction, user_reaction):
                return user_reaction == user and str(reaction.emoji) == "✅"
            # Add information about the established room to the dictionary
            self.user_channels[ctx.author.id] = ctx.author.id
            try:
                # Waiting for a reaction from the user for 60 seconds
                reaction, user_reaction = await self.bot.wait_for("reaction_add", timeout=60, check=check_user)

                # Here find out if the user reacted as a Buyer or Seller
                check_message = await channel.send("React to the message according to your role in the trade Buyer 🟩 or Seller 🟥")
                await check_message.add_reaction("🟩")
                await check_message.add_reaction("🟥")

                def check_choice(reaction, user_reaction):
                    return str(reaction.emoji) in ["🟩", "🟥"]

                try:
                    # Waiting for a reaction from the user for 60 seconds
                    reaction, user_reaction = await self.bot.wait_for("reaction_add", timeout=60, check=check_choice)

                    if str(reaction.emoji) == "🟩":
                        self.buyers_data[channel.id] = {
                            'user': user_reaction,
                            'address': None,
                            'com_amount': None,
                            'usdt': None,
                            'refund_address': None,
                            'name': None,
                            'user_id': None,
                            'channel_id':None
                        }
                        await channel.send(f"{user_reaction.mention} has been assigned as Buyer. When you are ready, enter the command !buy")
                    elif str(reaction.emoji) == "🟥":
                        self.sellers_data[channel.id] = {
                            'user': user_reaction,
                            'address': None,
                            'com_amount': None,
                            'usdt': None,
                            'refund_address': None,
                            'commune_add': None,
                            'key_name': None,
                            'name': None,
                            'user_id':None
                        }
                        await channel.send(f"{user_reaction.mention} has been assigned as Seller. When you are ready, enter the command !sell")

                    # Waiting for a reaction from the author for 60 seconds
                    reaction, author_reaction = await self.bot.wait_for("reaction_add", timeout=60, check=check_choice)

                    if str(reaction.emoji) == "🟩":
                        self.buyers_data[channel.id] = {
                            'user': author_reaction,
                            'address': None,
                            'com_amount': None,
                            'usdt': None,
                            'refund_address': None,
                            'name': None,
                            'user_id': None,
                            'channel_id':None
                        }
                        await channel.send(f"{author_reaction.mention} has been assigned as Buyer. When you are ready, enter the command !buy")
                    elif str(reaction.emoji) == "🟥":
                        self.sellers_data[channel.id] = {
                            'user': author_reaction,
                            'address': None,
                            'com_amount': None,
                            'usdt': None,
                            'refund_address': None,
                            'commune_add': None,
                            'key_name': None,
                            'name': None,
                            'user_id':None
                        }
                        await channel.send(f"{author_reaction.mention} has been assigned as Seller. When you are ready, enter the command !sell")

                    users_commands = (
                    "0. If you make some fuckup use cancel_trade\n"
                    "1. You have 2 hours to complete the OTC deal; the room will be deleted afterward.\n"
                    "2. Trading instructions:\n"
                    "   - Agree on the trade amount and sides: Buyer (COM Buy) = !buy and Seller (COM Sell) = !sell.\n"
                    "   - First user: Call !buy or !sell, answer Bot's questions accurately.\n"
                    "   - Second user: Call opposite function, answer Bot's questions accurately.\n"
                    "   - After correct input, call !check_trade; for deposit amount you have 90.\n"
                    
                    )

                    await channel.send(users_commands)
                except asyncio.TimeoutError:
                    await channel.send("No one reacted to the message about choosing the Buyer or Seller role.")
                    await ctx.author.send(f"{user.mention} did not confirm the message in time. The operation was cancelled.")
                    del user_channels[ctx.author.id]
                    self.buyers_data.pop(channel.id, None)
                    self.sellers_data.pop(channel.id, None)
                    await channel.delete()
                    return False

            except asyncio.TimeoutError:
                # If the user does not confirm within 60 seconds, return False
                await ctx.author.send(f"{user.mention} did not confirm the message in time. The operation was cancelled.")
                await channel.delete()
                del user_channels[ctx.author.id]
                self.buyers_data.pop(channel.id, None)
                self.sellers_data.pop(channel.id, None)
                return False

            self.time_data[channel.id] = {
                'time_left': 5600, 
                'time_extension_counter': 0,
                'auto_cancel_channel_time': 7200, 
                'round_counter' : 0,
                'com_status' : False,
                'eth_status' : False,
                'end_transaction_counter': 0,
                'cancel_trade_status': False,
                'check_trade_running': False,
                }
            self.command_status[ctx.channel.id] = False
            
            # Wait 9000 seconds 126minutes and delete the channel
            time_a_sleep =self.time_data[channel.id]["auto_cancel_channel_time"]
            await asyncio.sleep(time_a_sleep)
            # Removing room information back from the dictionary
            
            del user_channels[ctx.author.id]
            del time_data[channel.id]
            self.buyers_data.pop(channel.id, None)
            self.sellers_data.pop(channel.id, None)
            await channel.delete()
            
    
        def usdt_agreement(buy_usdt, sell_usdt):
            if buy_usdt == sell_usdt:
                return True
            else:
                return False

        def com_agreement(buyer_com_amount, seller_com_amount):
            if buyer_com_amount == seller_com_amount:
                return True
            else:
                return False

        @self.bot.command(name='buy', help='Buy command')
        @check_category(self.BOT_OTC_CATEGORY_ID)
        async def buy(ctx):
            name = ctx.author.mention
            user_id = ctx.author.id
            channel_id =ctx.channel.id
            # Check if the author of the command is the buyer
            buyer_data = self.buyers_data.get(ctx.channel.id)
            
            if buyer_data is None:
                return await ctx.send('There is no buyer set for this channel.')

            if ctx.author.id != buyer_data['user'].id:
                await ctx.send("You are not authorized to use this command.")
                return
            # Respond to the "!buy" command from the user

            await ctx.send('Enter your COMM Wallet address:')
            # Save user nickname to the variable seller_name
            
            def check(msg):
                return msg.author == ctx.author and msg.channel == ctx.channel

            try:
                response = await self.bot.wait_for('message', check=check, timeout=90)
                # Save the user's response to the buyer_address variable
                address = response.content
                await ctx.send(f'COMM Wallet address saved: {address}')

                # Ask the user for the refund address
                await ctx.send('Refund address(Ethere address):')
                response = await self.bot.wait_for('message', check=check, timeout=90)
                # Save the user's response to the buyer_refund_address variable
                refund_address = response.content
                await ctx.send(f'Refund address saved:{refund_address}')

                # Ask the user for the number of agreed COMM tokens
                await ctx.send('COM amount (max. two decimals)?')
                response = await self.bot.wait_for('message', check=check, timeout=90)
                # Save the user's response to the buyer_com_amount variable
                com_amount = float(response.content)
                await ctx.send(f'COM amount: {com_amount}')

                # Ask the user for the number of agreed USDT tokens
                await ctx.send('USDT amount (max. two decimals)?')
                response = await self.bot.wait_for('message', check=check, timeout=90)
                # Save the user's response to the buy_usdt variable
                usdt = float(response.content)
                await ctx.send(f'USDT amount saved: {usdt}')
                await ctx.send(f"If the command !sell was called and filled out, continue with the command !check_trade")
            except TimeoutError:
                await ctx.send('Time has expired, no response was provided.')
            address = str(address)
            buyer_data['address'] = address
            buyer_data['com_amount'] = com_amount
            buyer_data['usdt'] = usdt
            refund_address=str(refund_address)
            buyer_data['refund_address'] = refund_address
            buyer_data['name']=name
            buyer_data['user_id']=user_id
            buyer_data['channel_id']= channel_id

            self.buyers_data[ctx.channel.id] = buyer_data

        @self.bot.command(name='sell', help='Sell command')
        @check_category(self.BOT_OTC_CATEGORY_ID)
        async def sell(ctx):
            name = ctx.author.mention
            user_id = ctx.author.id
            # Check if the author of the command is the buyer
            seller_data = self.sellers_data.get(ctx.channel.id)
            
            if seller_data is None:
                return await ctx.send('There is no seller set for this channel.')
            
            if ctx.author.id != seller_data['user'].id:
                await ctx.send("You are not authorized to use this command.")
                return
            # Respond to the "!trade" command from the user
            await ctx.send('Enter your USDT Wallet address:')
            # Save user nickname to the variable seller_name
            def check(msg):
                return msg.author == ctx.author and msg.channel == ctx.channel
            
            try:
                response = await self.bot.wait_for('message', check=check, timeout=90)
                # Save the user's response to the seller_address variable
                address = response.content
                await ctx.send(f'USDT Wallet address saved: {address}')

                # Ask the user for the refund address
                await ctx.send('Refund address(Commune address):')
                response = await self.bot.wait_for('message', check=check, timeout=90)
                # Save the user's response to the seller_refund_address variable
                refund_address = response.content
                await ctx.send(f'Refund address saved: {refund_address}')

                # Ask the user for the number of agreed COMM tokens
                await ctx.send('COM amount (two decimals)?')
                response = await self.bot.wait_for('message', check=check, timeout=90)
                # Save the user's response to the seller_com_amount variable
                com_amount = float(response.content)
                await ctx.send(f'COM amount saved: {com_amount}')

                # Ask the user for the number of agreed USDT tokens
                await ctx.send('USDT amount (two decimals)?')
                response = await self.bot.wait_for('message', check=check, timeout=90)
                # Save the user's response to the sell_usdt variable
                usdt = float(response.content)
                await ctx.send(f'USDT amount saved: {usdt}')
                await ctx.send(f"If the command !buy was called and filled out, continue with the command !check_trade")
                
            except TimeoutError:
                await ctx.send('Time has expired, no response was provided.')
            address = str(address)
            seller_data['address'] = address
            seller_data['com_amount'] = com_amount
            seller_data['usdt'] = usdt
            refund_address= str(refund_address)
            seller_data['refund_address'] = refund_address
            commune_key = router_key()
            seller_data['commune_add']=commune_key['address']
            seller_data['key_name']= commune_key['key_name']
            seller_data['name'] = name
            seller_data['user_id']= user_id


            self.sellers_data[ctx.channel.id] = seller_data
            

        @self.bot.command(name='check_trade', help='Check trade command')
        @check_category(self.BOT_OTC_CATEGORY_ID)
        async def check_trade(ctx):
            seller_data = self.sellers_data.get(ctx.channel.id)
            buyer_data = self.buyers_data.get(ctx.channel.id)
            
            if seller_data['usdt']== None:
                await ctx.send('There is no seller set for this channel.')
                return     

            if buyer_data['usdt'] ==None:
                await ctx.send('There is no buyer set for this channel.')
                return 
            
            if not com_agreement(float(buyer_data['com_amount']), float(seller_data['com_amount'])):
                await ctx.send('Unfortunately, you did not agree with the counterparty on the number of COMM tokens. Please try again.')
                return

            usdt = float(seller_data['usdt'])
            if not usdt_agreement(floatt(buyer_data['usdt']), usdt):
                await ctx.send('Unfortunately, you did not agree with the counterparty on the number of USDT tokens. Please try again.')
                return
            if usdt <= 20:
                await ctx.send("Your transaction value is below 20 USDT, so it cannot be executed.")
                return
            if ctx.channel.id not in command_status:
                command_status[ctx.channel.id] = False

            if command_status[ctx.channel.id]:
                return await ctx.send('This command is already running in this channel.')
            else:
                command_status[ctx.channel.id] = True
            
            await info_about_fee(ctx, usdt)
            await ctx.send(f"{buyer_data['user'].mention} send {usdt} USDT to the address: {ethereum_address}")
            await ctx.send(f"{seller_data['user'].mention} send {buyer_data['com_amount']} COM to the address: {seller_data['commune_add']}")
            

            support = await self.bot.fetch_user(1174784033528487986)#OTC_BOT_SUPPORT
            await support.send(f"""
            ORDER channel.id: {buyer_data['channel_id']}
            ORDER is realizing buyer {buyer_data['name']} and seller {seller_data['name']}
            ORDER user_id: {buyer_data['user_id']}, {seller_data['user_id']}
            ORDER amount COM: {buyer_data['com_amount']}
            ORDER amount USDT: {buyer_data['usdt']}

            BUYER
            name: {buyer_data['name']}
            address: {buyer_data['address']}
            refund_address: {buyer_data['refund_address']}

            SELLER
            name: {seller_data['name']}
            address: {seller_data['address']}
            refund_address: {seller_data['refund_address']}

            My MIDLEBOT
            address com: {seller_data['commune_add']}
            path com: {seller_data['key_name']}
            """)

            
            round_counter= self.time_data.get(ctx.channel.id, {}).get('round_counter', 0)
            end_transaction_counter= self.time_data.get(ctx.channel.id, {}).get('end_transaction_counter', 0)
            time_left_for_channel = self.time_data.get(ctx.channel.id, {}).get('time_left', 5600)
            com_status = self.time_data.get(ctx.channel.id, {}).get('com_status', False)
            eth_status = self.time_data.get(ctx.channel.id, {}).get('eth_status', False)
            time_data[ctx.channel.id]['check_trade_running'] = True
            
            
            start_time = time.time()

            while True:
                
                try:
                    elapsed_time = time.time() - start_time
                    
                    agreed_cancel_trade = self.time_data[ctx.channel.id]['cancel_trade_status']#time_data.get(ctx.channel.id, {}).get('cancel_trade_status')
                    
                    if elapsed_time >= time_left_for_channel or agreed_cancel_trade:
                        print("drop to refund part")
                        end_transaction_counter +=1
                        if end_transaction_counter<=1:
                            try:
                                print("start refunding")
                                usdt_amount=calculate_fee_refund(buyer_data['usdt'])
                                com_amount= calculate_fee_refund_com(seller_data['com_amount'])
                                await ctx.send("Deposits aren't OK.")
                                
                                if eth_status:
                                    send_usdt_transaction(buyer_data['refund_address'], usdt_amount)
                                    await ctx.send("Refund USDT")
                                    await ctx.send(f"Your USDT amount was send on: {buyer_data['refund_address']}")
                                    await support.send(f"""
                                    refund usdt tokens channel_id: {ctx.channel.id} 
                                    buyer user name: {buyer_data['name']}
                                    buyer user id: {buyer_data['user_id']}
                                    buyer refund address: {buyer_data['refund_address']}
                                    buyer refund amount usdt: {buyer_data['usdt']}
                                    """)
                                    await ctx.send("If you sent some COM balance to the address, contact me at 'OTC_BOT_SUPPORT'")
                                elif com_status:
                                    await ctx.send("Refund COM")
                                    await transfer(sender=seller_data['key_name'], receiver=str(seller_data['refund_address']), amount=com_amount)
                                    
                                    await ctx.send(f"COM balance was sent on {seller_data['refund_address']}")
                                    await support.send(f"""
                                    refund commune tokens channel_id: {ctx.channel.id}
                                    seller user name: {seller_data['name']}
                                    seller user id: {seller_data['user_id']}
                                    seller refund address: {seller_data['refund_address']}
                                    seller refund amount usdt: {seller_data['usdt']}
                                    """)
                                    all_ballance = await balance(seller_data['commune_add'])
                                    all_ballance= all_ballance - 0.2
                                    await ctx.send("trasaction was delivery")
                                    await transfer(sender=seller_data['key_name'], receiver=self.my_wallet_address, amount=all_ballance)
                                    await ctx.send("If you sent some USDT balance to the address, contact me at 'OTC_BOT_SUPPORT'")
                                else:
                                    await support.send(f"""
                                    refund commune tokens channel_id: {ctx.channel.id}
                                    seller user name: {seller_data['name']}
                                    seller user id: {seller_data['user_id']}
                                    buyer user name: {buyer_data['name']}
                                    buyer user id: {buyer_data['user_id']}
                                    neither side sent the funds there is no refund
                                    """)
                                    await ctx.send("Neither of the transactions completed successfully")
                                    await ctx.send("If you sent any balance to the address, contact me at 'OTC_BOT_SUPPORT'")

                                self.command_status[ctx.channel.id] = False
                                end_transaction_counter=0
                                await ctx.send("Channel going to destroy in next 3 minutes.")
                                await asyncio.sleep(180)
                                del self.time_data[ctx.channel.id]
                                del self.buyers_data[ctx.channel.id]
                                del self.sellers_data[ctx.channel.id]
                                await ctx.channel.delete()
                                break
                                    
                            except Exception as e:
                                await ctx.send(f"error {e}")
                        
                    if eth_status != True:
                        eth_status = confirm_deposit_ethereum(ethereum_address, usdt)
                        #print(eth_status)
                        if eth_status:
                            await ctx.send("USDT deposit was recived")

                    round_counter +=1
                    
                    if com_status != True:
                        start_times = time.time()
                        if round_counter%50==0:
                            transaction_worth = buyer_data['com_amount']
                            com_status: bool = await manage_transaction(expected_amount=transaction_worth, address=seller_data['commune_add'])
                            elapsed_times = time.time() - start_times
                            if elapsed_times > 10:
                                return False
                            #print(f"com status is {com_status}")
                            if com_status:
                                await ctx.send("Commune deposit was recived")

                    if round_counter%100 == 0:
                    
                        time_left_chat_minute = elapsed_time/60
                        time_left_for_channel_minute = time_left_for_channel/60
                        true_time_left = time_left_for_channel_minute - time_left_chat_minute
                        true_time_left = round(true_time_left)
                        await ctx.send(f"Time left for deposit: {true_time_left} minutes")

                    if com_status and eth_status:
                        round_counter +=1
                        end_transaction_counter +=1
                        if end_transaction_counter<=1:

                            await end_transaction(ctx, buyer_data, seller_data, support)
                            self.command_status[ctx.channel.id] = False
                            end_transaction_counter = 0
                            break
                    else:
                        await asyncio.sleep(6)
                except Exception as e:
                    await ctx.send(f"error {e}")

        @self.bot.command(name='cancel_trade', help='Cancel trade command')
        @check_category(self.BOT_OTC_CATEGORY_ID)
        async def cancel_trade(ctx):
            author_id = ctx.author.id

            # Get the other user in the channel
            other_user = next(member for member in ctx.channel.members if member != ctx.author)
            other_user_id = other_user.id  # Fix this line

            # Send a message to the other user asking for confirmation
            message = await ctx.send(f"{other_user.mention}, do you agree to delete this channel? React with ✅ to confirm.")
            
            # Add the reaction options
            await message.add_reaction("✅")

            def check_reaction(reaction, user):
                return user == other_user and str(reaction.emoji) == "✅"
            try:
                # Wait for the other user's reaction for 60 seconds
                reaction, user = await self.bot.wait_for("reaction_add", timeout=60, check=check_reaction)
                # Send a message to users with permission to write in this channel
                running_check_trade :bool= self.time_data[ctx.channel.id]['check_trade_running']
                if running_check_trade:
                    self.time_data[ctx.channel.id]['cancel_trade_status'] = True
                elif running_check_trade ==  False:
                    
                    del self.time_data[ctx.channel.id]
                    del self.buyers_data[ctx.channel.id]
                    del self.sellers_data[ctx.channel.id]
                    self.command_status[ctx.channel.id] = False
                    await ctx.channel.delete()
                else:
                    print("TRUMP 2024")
                    return
                # Delete user from the dictionary
                if other_user_id in user_channels:
                    del self.user_channels[other_user_id]
                elif author_id in user_channels:
                    del self.user_channels[author_id]
                else:
                    print("nobody is in self.user_channels dictionary")
                    return
            except asyncio.TimeoutError:
                await ctx.send(f"{other_user.mention} did not agree to delete the channel within 60 seconds.")
                return
                # Handle timeout error if the other user doesn't react in tim
        
        self.bot.remove_command('help')
        @self.bot.command(name='help', help='Displays this help message')
        async def help(ctx):
            commands_info = ""
            for command in self.bot.commands:
                commands_info += f"{command.name}: {command.help}\n"
            await ctx.send(commands_info)

if __name__ == "__main__":
    nest_asyncio.apply()  # Apply the patch before any async tasks are created.
    otc = OTC()
    # connect to discord bot token
    
    otc.bot.run(otc.BOTTOKEN)
