# NBA Bet 🏀

NBA Bet is an experimental software which aims at predicting the outcome of NBA games by the utilization of ML (and non-ML) models. 
After a result has been predicted, NBA Bet reaches a bookmaker of choice (currently testing with *Bwin*) to get the odds for the predicted match.

The software is suitable for running continuously (e.g., by running it on a Raspberry Pi) and features a [telegram integration module](https://github.com/davideganna/NBA_Bet/blob/main/NBABet/telegram_integration.py) which can be used to get notified if a particular match is profitable.

## Main structure
The diagram below shows the execution flow of NBA Bet:

![image](https://user-images.githubusercontent.com/52606991/127752880-cef2e6c3-4e72-406f-b16b-9723b6a289fd.png)

#### A few notes regarding ```Setup.py```:
In its current version, running ```Setup.py``` is not needed if the project has been cloned from https://github.com/davideganna/NBA_Bet.git. When cloned this way, all the datasets are included and can be found in the ```past_data``` folder. However, when the 2021-2022 NBA season will start, running ```Setup.py``` will be needed. 


## Disclaimer

Gambling involves risk. The author does not encourage or promote gambling in any way, nor the author takes any responsibilities for losses associated with the utilization of the software. 
The utilization NBA Bet is only intended for legal age people. 
