clear all

*----------------------------------------------------------------------
* National Cotton Acreage Forecasting (1990-2024)
* Adapted from the original Georgia-level workflow (cotton-vs-peanuts)
* to the national dataset (cotton-vs-corn), matching the model published
* in Parajuli & Liu (2025), "Projections for 2025 Cotton Acreage,"
* Southern Ag Today 5(18.3).
*----------------------------------------------------------------------

import delimited "cotton_national_1990_2024.csv", clear

sort year //sorting data based on year in ascending order
gen t = _n
tsset t //set time variable

//label the variables
label variable area_1000ac "area of cotton (in 1000 acres)"
label variable yield_lb_per_ac "cotton yield (in lb per acre)"
label variable pratio_cotton_corn "cotton-to-corn price ratio"
label variable price_cotton_cents_lb "price of cotton (cents per lb)"
label variable price_corn_dollar_bu "price of corn ($ per bu)"

//testing autocorrelation function
ac area_1000ac, lag(10) name(ACF1, replace)
ac pratio_cotton_corn, lag(10) name(ACF2, replace)
graph combine ACF1 ACF2
graph export "ACF of key variable.jpg", replace

pac area_1000ac, lag(10) name(PACF1, replace)
pac pratio_cotton_corn, lag(10) name(PACF2, replace)
graph combine PACF1 PACF2
graph export "PACF of key variable.jpg", replace

//generate lagged variables
gen area_lag1 = L1.area_1000ac
gen area_lag2 = L2.area_1000ac
gen pratio_lag1 = L1.pratio_cotton_corn
gen pratio_lag2 = L2.pratio_cotton_corn
gen price_cotton_lag1 = L1.price_cotton_cents_lb

//Model: AREA_t = b0 + b1*AREA_(t-1) + b2*PRATIO_(t-1) + b3*PRICE_COTTON_t
reg area_1000ac area_lag1 pratio_lag1 price_cotton_cents_lb
outreg2 using myreg.doc, replace label ctitle (National Model) title(Table: National Cotton Acreage Regression)

//Estimating the model using OLS and Newey-West
reg area_1000ac area_lag1 pratio_lag1 price_cotton_cents_lb
predict uhat, residual
dwstat
outreg2 using myreg2.doc, replace label ctitle (OLS estimation) title(Table: National Cotton Acreage Regression)

newey area_1000ac area_lag1 pratio_lag1 price_cotton_cents_lb, lag(1)
predict ehat, residual
outreg2 using myreg2.doc, append label ctitle (Newey-West estimation) title(Table: National Cotton Acreage Regression)

//Augmented Dickey-Fuller test for unit root in residuals
dfuller ehat, lag(2) //Null hypothesis of Unit root (H0: theta = 0)

asdoc sum uhat ehat //summary of residuals from OLS estimation and Newey-West estimation
