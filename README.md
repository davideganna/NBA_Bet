[![codecov](https://codecov.io/gh/davideganna/NBA_Bet/graph/badge.svg?token=DHDLX2UXH0)](https://codecov.io/gh/davideganna/NBA_Bet)

# NBA Bet 🏀

NBA Bet is an experimental software which aims at predicting the outcome probability of NBA games by the utilization of ML (and non-ML) models.
After a result has been predicted, NBA Bet reaches a bookmaker of choice (currently testing with *SkyBet*) to get the odds for the predicted match.

The software is suitable for running continuously (e.g., by running it on a Raspberry Pi) and features a [telegram integration module](https://github.com/davideganna/NBA_Bet/blob/435dd874b8ccd60744a2b51cdb09f1aa9bfe320e/NBABet/telegram.py) which can be used to get notified if a particular match is profitable.

But what does _profitable_ mean?

## Mathematical foundations behind match profitability: Introducing EV
The Expected Value (EV) of a bet can be found as:

**EV = P(X) &middot; [Bet<sub>am</sub> &middot; (Odds-1)] - [1-P(X)] &middot; Bet<sub>am</sub>**

Where:

P(X) is the probability that an event occurs (returned by NBA bet);

Bet<sub>am</sub> is the amount of money invested;

Odds are the odds of that event (returned by the bookmaker).

####

The result obtained after the computation of the above formula is the expected gain/loss of the bet. An example follows:

Consider a NBA match between _HomeTeam_ and _AwayTeam_, where NBA bet returned a probability of _HomeTeam_ winning equal to 0.75, and the odds offered by a particular bookmaker are 1.38. With this information, if we choose to bet 100$ on the said match, the EV of the bet is:

EV = 0.75 &middot; (100$ &middot; 0.38) - 0.25 &middot; 100$

EV = 3.5$

Thus making the bet +EV, or _profitable_.

### A shortcut for calculating EV

It can be shown that the EV of a bet can be quickly calculated as the reciprocal of the probability of an event being lower than the odds offered by the bookmaker. Any bet that satisfies this criterion, has a positive EV. In the example above, 1/0.75 = 1.333 is less than 1.38, therefore fulfilling the criterion.

## How much should I bet?
The optimal bet amount is a function of the probability of the event occurring, the odds of the event and the available bankroll. It takes the name of _Kelly's criterion_, and can be found [here](https://en.wikipedia.org/wiki/Kelly_criterion).

## Main structure
The diagram below shows the execution flow of NBA Bet:

![image](https://user-images.githubusercontent.com/52606991/127752880-cef2e6c3-4e72-406f-b16b-9723b6a289fd.png)

### Best Hyperparameters Selection (latest software version)
The simulated results from NBA Season 2020 can be found below:
```
Random Forest (500 trees) + Elo

leave_out = '2020'
margin = 0
betting_limiter = True
betting_limit = 0.125
prob_threshold = 0.65
prob_2x_bet = 0.8
average_N = 3
skip_n = 0

Net return per €: 2.10

```

![Figure_1](https://user-images.githubusercontent.com/52606991/145692119-f67c513c-2d7c-48a1-bd85-de689b8f5a7b.png)

The simulation combines a 500-tree RF model stacked with an Elo model to bet only on matches predicted by both models. Bets amount are found using Kelly's criterion, with an initial bankroll of 100€.

## Disclaimer

Gambling involves risk. The author does not encourage or promote gambling in any way, nor the author takes any responsibilities for losses associated with the utilization of the software.
The utilization NBA Bet is only intended for legal age people.
