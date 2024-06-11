# OTC BOT TUTORIAL

## Roles
- **Buyer:** Wants to buy COM.
- **Seller:** Wants to sell COM.

## Steps to Initiate Trade
1. **Find a Trading Partner:** Look for another user to trade with.
2. **Start Trade:** Use the command `!OTC @another_user` in the command channel. This creates a new channel for trading commands.
3. **Accept Channel Creation:** The other user has 60 seconds to agree to the creation of the room. If they do not accept, the room is automatically deleted.
4. **Button Selection:**
   - If you are the buyer, click the **GREEN** button.
   - If you are the seller, click the **RED** button.
5. **Cancel Trade:** If an error occurs, use `!cancel_trade`. If both parties agree, the trading room will be destroyed. This command can be used anytime during the trade in the trade channel. If any balance has been sent, it will be refunded.
6. **Time Limit:** You have 2 hours to complete the OTC deal, after which the room will be deleted.

## Trading Process
1. **Agree on Trade Terms:** Decide on the trade amount and roles (Buyer = COM Buy, Seller = COM Sell).
2. **First User Action:** The first user should call `!buy` or `!sell` and accurately answer the Bot's questions.
3. **Second User Action:** The second user should call the opposite function (`!buy` or `!sell`) and accurately answer the Bot's questions.
4. **Final Confirmation:** After all inputs are correctly entered, use `!check_trade`. You then have 90 minutes to deposit the agreed amount.

## Fee Information

### Transaction Fees for USDT
- Under 30 gwei: 1 USDT
- Under 60 gwei: 3 USDT
- Under 90 gwei: 7 USDT
- Above 90 gwei: 10 USDT

### Fees for Realized Trade
- Under 500 USDT: 5 USDT fee, 5 COM fee
- Under 1,000 USDT: 1% USDT fee, 1% COM fee
- Under 8,000 USDT: 0.8% USDT fee, 0.8% COM fee
- Above 8,000 USDT: 0.5% USDT fee, 0.5% COM fee

### Fees for Refund Transaction
- COM: 4 fee
- USDT: 5 fee

## Adding the Bot to Your Server
To add this bot to your Discord server, use the following link: [Add OTC Bot to Server](https://discord.com/api/oauth2/authorize?client_id=1161374958589595728&permissions=8&scope=bot).
- ***Note:*** in order for the bot to work correctly you have to create: 
- caregory: `OTC channels`
- channel: `bot commands`

## Support
- **Contact Support:** Use OTC_BOT_SUPPORT.
- **Verify Support ID:** Double check user_id: `1174784033528487986`.
