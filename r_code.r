---
title: "GEX"
output: html_document
date: "2023-09-20"
---

```{r setup, include=FALSE}
# require packages
require("RQuantLib");require("derivmkts");require("highcharter");require("dplyr");require("quantmod");require("plyr");require("pbapply")
# load functions to import option data
#source("getOptionSymbolsDB.R")
```

## Data Import

Read in options data for $SPY along with the 1-year Risk-Free Rate.
```{r dataImport}
# grab all optons chains for  SPY
#ops <- getOpByTicker(ticker="SPY",from = "2023-09-19")
ops <- readRDS("spyOps.rds")
# set run date
FROM = Sys.Date()
# grab 
getSymbols("DGS1",src="FRED")
# convert to daily rates for option calculation
DGS1 <- DGS1/100/360
if(index(DGS1)[nrow(DGS1)] != FROM){
  DGS1 = rbind(DGS1, xts(coredata(DGS1[nrow(DGS1)]),order.by = FROM))
}

```

## Calculate GEX

In this block we will separate the calls/puts. For each strike we calculate the theoretical option price for a wide range of underlying prices. We can then calculate the gamma exposure for all the option chains at different levels of the underlying price.

Option Columns required:
1. Option Flag - call or put
2. Open Interest
3. Strike
4. Implied Volatility
5. Expiration Date

How to Calculate GEX: <https://perfiliev.co.uk/market-commentary/how-to-calculate-gamma-exposure-and-zero-gamma-level/>
```{r gex, echo=FALSE}

# assign trading date for options data
x = "2023-09-19"

# extract ops on that tDay
opps = subset(ops, ops$Date == x)

calls = subset(opps,opps$flag=="C")
puts = subset(opps,opps$flag=="P")

# underlying close
uPrice = calls$stkClose[1]
# create range of underlying prices
uPrice = round_any(as.integer(calls$stkClose)[1],accuracy = 10)
minPrc = uPrice - 50
maxPrc = uPrice + 50

stkRange = seq(minPrc,maxPrc,1)
# gex for calls
call_gamma  = pblapply(as.list(1:nrow(calls)), function(ii){
  # subset option
  thisOp <- calls[ii,]
  # volatility
  thisVol <- ifelse(thisOp$iv==0,thisOp$calc_IV,thisOp$iv)
  # interest rate (1-year)
  thisRate <- as.numeric(DGS1[thisOp$Date])*thisOp$days2Exp
  # expiration in years
  thisExp <- thisOp$days2Exp/365
  # calculate option price for a range of stk prices
  GREEKS <- suppressWarnings(derivmkts::greeks2(fn = bscall,list(s = stkRange,k = thisOp$strike, v = thisVol,r = thisRate,tt = thisExp,d = 0)))
  GREEKS <- data.frame(t(GREEKS),row.names = NULL)
  GREEKS$OI <- thisOp$open_interest
  GREEKS$underlyingClose <- stkRange
  GREEKS <- GREEKS[,c("Gamma","OI","underlyingClose")]
  GREEKS$strike <- thisOp$strike
  GREEKS$direction <- 1
  GREEKS
})
call_gamma = rbindlist(call_gamma,use.names = TRUE,fill = TRUE) %>% as.data.frame
call_gamma = call_gamma %>% group_by(underlyingClose) %>% reframe(calls = sum(Gamma*100*OI*underlyingClose*underlyingClose*0.01*direction))

# gex for puts
put_gamma  = pblapply(as.list(1:nrow(puts)), function(ii){
  # subset option
  thisOp <- puts[ii,]
  # volatility
  thisVol <- ifelse(thisOp$iv==0,thisOp$calc_IV,thisOp$iv)
  # interest rate (1-year)
  thisRate <- as.numeric(DGS1[thisOp$Date])*thisOp$days2Exp
  # expiration in years
  thisExp <- thisOp$days2Exp/365
  # calculate option price for a range of stk prices
  GREEKS <- suppressWarnings(derivmkts::greeks2(fn = bsput,list(s = stkRange,k = thisOp$strike, v = thisVol,r = thisRate,tt = thisExp,d = 0)))
  GREEKS <- data.frame(t(GREEKS),row.names = NULL)
  GREEKS$OI <- thisOp$open_interest
  GREEKS$underlyingClose <- stkRange
  GREEKS <- GREEKS[,c("Gamma","OI","underlyingClose")]
  GREEKS$strike <- thisOp$strike
  GREEKS$direction <- -1
  GREEKS
})
put_gamma = rbindlist(put_gamma,use.names = TRUE,fill = TRUE) %>% as.data.frame
put_gamma = put_gamma %>% group_by(underlyingClose) %>% reframe(puts = sum(Gamma*100*OI*underlyingClose*underlyingClose*0.01*direction))

# merge call/put data
call_gamma$puts <- put_gamma$puts
call_gamma$total <- call_gamma$calls + call_gamma$puts
all_gamma <- call_gamma

```


## Plot GEX Profile - All expirations

Plot the underlying prices against the GEX. We will add the zero-line and calculate where the dealers might 'flip':
```{r gexProfile}
# find where the 'total' goes from negative to positive
flip = which(diff(sign(all_gamma$total))!=0)
# subset rows to fill negative/positive sides
neg2zero = all_gamma[flip,]
pos2zero = all_gamma[flip+1,]
# Create two vectors containing the coordinates of the two points.
p1 <- c(neg2zero$underlyingClose, neg2zero$total)
p2 <- c(pos2zero$underlyingClose, pos2zero$total)
# Find the x-coordinate of the point of intersection.
x_intersection <- (p1[1] * p2[2] - p1[2] * p2[1]) / (p1[2] - p2[2])
# Calculate the y-coordinate of the point of intersection.
y_intersection <- -p1[1] * -x_intersection / p1[2]
# Make the x-intersection positive.
if (x_intersection < 0) {
  x_intersection <- -x_intersection
}
# add intersection points 
neg2zero$total = round(y_intersection,2)
pos2zero$total = round(y_intersection,2)
pos2zero$underlyingClose <- round(x_intersection,2)
neg2zero$underlyingClose <- round(x_intersection,2)
# create flip price
flip_prc <- round(x_intersection,2)
# row bind gamma totals along with calculated levels
all_gamma <- rbind(all_gamma[1:flip,],neg2zero,pos2zero,all_gamma[(flip+1):nrow(all_gamma),])
# highchart plot
highchart() %>%
  hc_add_series(data = all_gamma[all_gamma$underlyingClose <= flip_prc,],hcaes(x = 'underlyingClose', y = 'total'), type = 'area',color='red') %>% 
  hc_add_series(data = all_gamma[all_gamma$underlyingClose >= flip_prc,],hcaes(x = 'underlyingClose', y = 'total'), type = 'area',color='green') %>%
  hc_title(text=paste0("$SPY GEX - All Option Chains"), align="center") %>%
  hc_subtitle(text=paste0("Flip Price: $",flip_prc)) %>%
  hc_legend(enabled = FALSE) %>%
  hc_tooltip(formatter = JS(
    "function() {
    return this.x + ' | ' + (this.y / 1000000000).toFixed(2) + 'B'
  }"
  ))

```





## Plot GEX Vs $SPY

We will plot the flip price, max, & min levels on the next day $SPY bars
```{r gexVSspy, fig.keep='last'}
spy <- readRDS("SPY230920.rds")
spy <- to.period(spy,period = "minutes",k=5,indexAt = "startof",name = "SPY",OHLC = TRUE)

min_price = all_gamma$underlyingClose[which.min(all_gamma$total)]
max_price = all_gamma$underlyingClose[which.max(all_gamma$total)]

chartSeries(spy["20230919::"],name = "$SPY")
quantmod::addLines(h=min_price,col='red')
quantmod::addLines(h=flip_prc,col='white')
quantmod::addLines(h=max_price,col='green')

```
