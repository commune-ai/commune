# OTC Bot User Guide

## Role Definitions
- **Buyer:** A user who wishes to purchase COM.
- **Seller:** A user who wishes to offload COM.

## Initiation Process

1. **Find a Trading Partner:** Locate another user who you wish to transact with.
2. **Initiate Trade:** Enter `!OTC @another_user` in the command channel to establish a new private setting for the transaction.
3. **Confirm Channel Creation:** The trading partner must give consent to the creation of this channel within 60 seconds. If this doesn't happen, the room is eradicated.
4. **Button Selection:** 
   - Buyers should click on the **GREEN** button.
   - Sellers should click on the **RED** button.
5. **Trade cancellation:** In case a mistake is made, enter `!cancel_trade` to undo the process. If both parties are in agreement, the negotiation room is dissolved. This command is valid at any point within the trade channel. Any transfer that might have occurred will be refunded.
6. **Time Limit:** A two-hour countdown is initiated at the start of the OTC deal. Failure to finalize within the allocated period will cause the room to be automatically deleted.

## Transaction Procedure

1. **Agree on Terms:** Both parties need to settle on the trading sum and roles (Buyer = COM Buy, Seller = COM Sell).
2. **Action by First User:** The initial user should key in `!buy` or `!sell` and respond accurately to the Bot's inquiry.
3. **Action by Second User:** The other user should call the contrary function (`!buy` or `!sell`) and also provide accurate responses to the Bot's questions.
4. **Closing Confirmation:** Upon accurate input of all details, input `!check_trade`. After this point, there's a 90-minute grace period to deposit the agreed trade value.

## Transaction Cost Information

### USDT Transaction Fees
- Less than 30 gwei: 1 USDT
- Less than 60 gwei: 3 USDT
- Less than 90 gwei: 7 USDT
- Over 90 gwei: 10 USDT

### Completed Trade Fees
- Under 500 USDT: 5 USDT fee, 5 COM fee
- Less than 1,000 USDT: 1% USDT fee, 1% COM fee
- Less than 8,000 USDT: 0.8% USDT fee, 0.8% COM fee
- Over 8,000 USDT: 0.5% USDT fee, 0.5% COM fee

### Refund Transaction Fees
- COM: 4 fee
- USDT: 5 fee

## Integrating the Bot with Your Server
Use the following URL to add the OTC Bot to your Discord Server: [Add OTC Bot to Server](https://discord.com/api/oauth2/authorize?client_id=1161374958589595728&permissions=8&scope=bot).
- ***Note:*** For the bot to function properly, you are required to generate: 
- Category: `OTC channels`
- Channel: `bot commands`

## Support Provision
- **Reach Out to Support:** Utilize OTC_BOT_SUPPORT.
- **Confirm Support ID Verification:** Ensure the user_id is `1174784033528487986`.