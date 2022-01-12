# NBA Bet 🏀

NBA Bet is an experimental software which aims at predicting the outcome of NBA games by the utilization of ML (and non-ML) models. 
After a result has been predicted, NBA Bet reaches a bookmaker of choice (currently testing with *SkyBet*) to get the odds for the predicted match.

The software is suitable for running continuously (e.g., by running it on a Raspberry Pi) and features a [telegram integration module](https://github.com/davideganna/NBA_Bet/blob/435dd874b8ccd60744a2b51cdb09f1aa9bfe320e/NBABet/Telegram.py) which can be used to get notified if a particular match is profitable.

## Main structure
The diagram below shows the execution flow of NBA Bet:

![image](https://user-images.githubusercontent.com/52606991/127752880-cef2e6c3-4e72-406f-b16b-9723b6a289fd.png)

## Disclaimer

Gambling involves risk. The author does not encourage or promote gambling in any way, nor the author takes any responsibilities for losses associated with the utilization of the software. 
The utilization NBA Bet is only intended for legal age people. 

### Best Hyperparameters Selection (latest software version)
``` 
Random Forest (500 trees) + Elo

leave_out = '2020'
margin = 0
betting_limiter = True
betting_limit = 0.125
prob_threshold = 0.65
prob_2x_bet = 0.8
offset = 0.0 # Added probability
average_N = 3
skip_n = 0

Net return per €: 2.10

```

![Figure_1](https://user-images.githubusercontent.com/52606991/145692119-f67c513c-2d7c-48a1-bd85-de689b8f5a7b.png)
