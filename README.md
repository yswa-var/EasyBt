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
  
How to use??
![Screenshot 2023-10-08 103726](https://github.com/bbmusa/EasyBt/assets/65719349/608aff3f-66b3-42d1-80b7-e62a1e7a1704)

Identifier,Indicator Name,Type,Inputs,Options,Outputs
abs,Vector Absolute Value,simple,1,0,1
acos,Vector Arccosine,simple,1,0,1
ad,Accumulation/Distribution Line,indicator,4,0,1
*add,Vector Addition,simple,2,0,1
adosc,Accumulation/Distribution Oscillator,indicator,4,2,1
adx,Average Directional Movement Index,indicator,3,1,1
adxr,Average Directional Movement Rating,indicator,3,1,1
ao,Awesome Oscillator,indicator,2,0,1
apo,Absolute Price Oscillator,indicator,1,2,1
aroon,Aroon,indicator,2,1,2
aroonosc,Aroon Oscillator,indicator,2,1,1
asin,Vector Arcsine,simple,1,0,1
atan,Vector Arctangent,simple,1,0,1
atr,Average True Range,indicator,3,1,1
avgprice,Average Price,overlay,4,0,1
upper_bband,upper Bollinger Bands,overlay,1,2,1
mid_bband,mid Bollinger Bands,overlay,1,2,1
lower_bband,lower Bollinger Bands,overlay,1,2,1
bop,Balance of Power,indicator,4,0,1
cci,Commodity Channel Index,indicator,3,1,1
ceil,Vector Ceiling,simple,1,0,1
cmo,Chande Momentum Oscillator,indicator,1,1,1
cos,Vector Cosine,simple,1,0,1
cosh,Vector Hyperbolic Cosine,simple,1,0,1
-crossany,Crossany,math,2,0,1
-crossover,Crossover,math,2,0,1
cvi,Chaikins Volatility,indicator,2,1,1
decay,Linear Decay,math,1,1,1
dema,Double Exponential Moving Average,overlay,1,1,1
di,Directional Indicator,indicator,3,1,2
div,Vector Division,simple,2,0,1
dm,Directional Movement,indicator,2,1,2
dpo,Detrended Price Oscillator,indicator,1,1,1
dx,Directional Movement Index,indicator,3,1,1
edecay,Exponential Decay,math,1,1,1
ema,Exponential Moving Average,overlay,1,1,1
emv,Ease of Movement,indicator,3,0,1
exp,Vector Exponential,simple,1,0,1
fisher,Fisher Transform,indicator,2,1,2
floor,Vector Floor,simple,1,0,1
fosc,Forecast Oscillator,indicator,1,1,1
hma,Hull Moving Average,overlay,1,1,1
kama,Kaufman Adaptive Moving Average,overlay,1,1,1
kvo,Klinger Volume Oscillator,indicator,4,2,1
lag,Lag,math,1,1,1
linreg,Linear Regression,overlay,1,1,1
linregintercept,Linear Regression Intercept,indicator,1,1,1
linregslope,Linear Regression Slope,indicator,1,1,1
ln,Vector Natural Log,simple,1,0,1
log10,Vector Base-10 Log,simple,1,0,1
macd,Moving Average Convergence/Divergence,indicator,1,3,3
marketfi,Market Facilitation Index,indicator,3,0,1
mass,Mass Index,indicator,2,1,1
max,Maximum In Period,math,1,1,1
md,Mean Deviation Over Period,math,1,1,1
medprice,Median Price,overlay,2,0,1
mfi,Money Flow Index,indicator,4,1,1
min,Minimum In Period,math,1,1,1
mom,Momentum,indicator,1,1,1
msw,Mesa Sine Wave,indicator,1,1,2
mul,Vector Multiplication,simple,2,0,1
natr,Normalized Average True Range,indicator,3,1,1
nvi,Negative Volume Index,indicator,2,0,1
obv,On Balance Volume,indicator,2,0,1
ppo,Percentage Price Oscillator,indicator,1,2,1
psar,Parabolic SAR,overlay,2,2,1
pvi,Positive Volume Index,indicator,2,0,1
qstick,Qstick,indicator,2,1,1
roc,Rate of Change,indicator,1,1,1
rocr,Rate of Change Ratio,indicator,1,1,1
round,Vector Round,simple,1,0,1
rsi,Relative Strength Index,indicator,1,1,1
sin,Vector Sine,simple,1,0,1
sinh,Vector Hyperbolic Sine,simple,1,0,1
sma,Simple Moving Average,overlay,1,1,1
sqrt,Vector Square Root,simple,1,0,1
stddev,Standard Deviation Over Period,math,1,1,1
stderr,Standard Error Over Period,math,1,1,1
stoch,Stochastic Oscillator,indicator,3,3,2
stochrsi,Stochastic RSI,indicator,1,1,1
sub,Vector Subtraction,simple,2,0,1
sum,Sum Over Period,math,1,1,1
tan,Vector Tangent,simple,1,0,1
tanh,Vector Hyperbolic Tangent,simple,1,0,1
tema,Triple Exponential Moving Average,overlay,1,1,1
todeg,Vector Degree Conversion,simple,1,0,1
torad,Vector Radian Conversion,simple,1,0,1
tr,True Range,indicator,3,0,1
trima,Triangular Moving Average,overlay,1,1,1
trix,Trix,indicator,1,1,1
trunc,Vector Truncate,simple,1,0,1
tsf,Time Series Forecast,overlay,1,1,1
typprice,Typical Price,overlay,3,0,1
ultosc,Ultimate Oscillator,indicator,3,3,1
var,Variance Over Period,math,1,1,1
vhf,Vertical Horizontal Filter,indicator,1,1,1
vidya,Variable Index Dynamic Average,overlay,1,3,1
volatility,Annualized Historical Volatility,indicator,1,1,1
vosc,Volume Oscillator,indicator,1,2,1
vwma,Volume Weighted Moving Average,overlay,2,1,1
wad,Williams Accumulation/Distribution,indicator,3,0,1
wcprice,Weighted Close Price,overlay,3,0,1
wilders,Wilders Smoothing,overlay,1,1,1
willr,Williams %R,indicator,3,1,1
wma,Weighted Moving Average,overlay,1,1,1
zlema,Zero-Lag Exponential Moving Average,overlay,1,1,1

-- Yashaswa Varshney
-- Connect with me on LinkedIn: https://www.linkedin.com/in/yashaswa-varshney/  


