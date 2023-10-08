# EasyBt
Python Library that helps the user to backtest any strategy or combinations of strategies and generate buy-sell signals like making lemonade

- Option Data Downloader (using ICICI free API)
  - Input API key, secret key
  - Select date Data range
  - Select ticker.... Done!
  - Comes with exception handling, it takes into consideration expiry dates (also compatible with current change in expiry date for all midcap, Nifty, BankNifty)  

- Multi Indicator (using Tulipy library [for C++ calculations])
  - Indicator_list.csv: list of indicators and metrics
  - main.py: main indicator testing library (only generate signals any amount of combinations either maths or any indicator)
  - Easy to implement new functions or custom calculations!
  - use_main.ipynb: example to use the above library.
  

!Screenshot 2023-10-08 103726

-- Yashaswa Varshney
-- Connect with me on LinkedIn: https://www.linkedin.com/in/yashaswa-varshney/  
