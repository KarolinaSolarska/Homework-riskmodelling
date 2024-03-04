################################################################################ 
##############################  Karolina Solarska  #############################
###################################  410858  ###################################
################################################################################ 

library(xts)
library(rugarch)
library(ROCR)
library(ggplot2)

Sys.setlocale("LC_TIME", "English")
setwd("...")

# Exercise 1.1
# a) Import quotations for any asset of your choice using either provided files 
# or quantmod package from first labs.  Then calculate logarithmic returns on 
# that asset and compare Value-at-Risk estimates for 2021 from 
# b) historical simulation, 
# c) GARCH models and 
# d) filtered historical simulation (models should be built on the data prior to  
# 01-01-2021 with moving window method).  
# e) Does the quality of the models change? Are some approaches much better than 
# the others? What caused such behavior? 
# f) Plot the VaR estimates together with data.

# a) Import quotations and calculate logarithmic returns
data <- read.csv("wig.csv", stringsAsFactors = F)
data <- xts(data[,-1], as.Date(data[, 1], "%Y-%m-%d"))
log_returns <- diff(log(data$Zamkniecie), lag = 1)

# setting p_value and forecast_horizon
p_value <- 0.05
forecast_horizon <- 250

# subsetting the data to end before 01-01-2021 for model training
data <- log_returns["2010/2021"]
training_sample_len <- nrow(data) - forecast_horizon


# b) historical simulation. 
HS <- rollapplyr(data, training_sample_len, 
                     function(w) {
                       quantile(w, p_value)
                     })

testHS <- tail(lag(HS, 1), 250)

# c) GARCH models 

# GARCH(1,1) with normal distribution
# GARCH model specification 
specnorm1 <-
  ugarchspec(
    variance.model = list(model="sGARCH", garchOrder=c(1, 1)),
    mean.model = list(armaOrder = c(0, 0), include.mean = FALSE),
    distribution.model = 'norm'
  )

# Implementation of moving window approach for GARCH model
Garch1 <- ugarchroll(specnorm1, data,
                       n.ahead = 1,
                       forecast.length = forecast_horizon,
                       refit.every = 1,
                       refit.window = 'moving', # moving window approach
                       keep.coef = TRUE,
                       calculate.VaR = TRUE,
                       VaR.alpha = p_value)

# Extracting VaR estimates
testGarch1 <- xts(Garch1@forecast$VaR,
                     as.Date(rownames(Garch1@forecast$VaR)))[, 1]

# plotting the results
plot(cbind(tail(data, forecast_horizon), Garch1@forecast$density$Sigma))

# GARCH(1,1) with skewed normal distribution
# GARCH model specification 
specnorm2 <-
  ugarchspec(
    variance.model = list(model="sGARCH", garchOrder=c(1, 1)),
    mean.model = list(armaOrder = c(0, 0), include.mean = FALSE),
    distribution.model = 'snorm'
  )

# Implementation of moving window approach for GARCH model
Garch2 <- ugarchroll(specnorm2, data,
                     n.ahead = 1,
                     forecast.length = forecast_horizon,
                     refit.every = 1,
                     refit.window = 'moving',
                     keep.coef = TRUE,
                     calculate.VaR = TRUE,
                     VaR.alpha = p_value)

# Extracting VaR estimates
testGarch2 <- xts(Garch2@forecast$VaR,
                  as.Date(rownames(Garch2@forecast$VaR)))[, 1]

# plotting the results
plot(cbind(tail(data, forecast_horizon), Garch2@forecast$density$Sigma))


# GARCH(1,1) with skewed t distribution
# GARCH model specification 
specnorm3 <-
  ugarchspec(
    variance.model = list(model="sGARCH", garchOrder=c(1, 1)),
    mean.model = list(armaOrder = c(0, 0), include.mean = FALSE),
    distribution.model = 'sstd'
  )

# Implementation of moving window approach for GARCH model
Garch3 <- ugarchroll(specnorm3, data,
                     n.ahead = 1,
                     forecast.length = forecast_horizon,
                     refit.every = 1,
                     refit.window = 'moving',
                     keep.coef = TRUE,
                     calculate.VaR = TRUE,
                     VaR.alpha = p_value)

# Extracting VaR estimates
testGarch3 <- xts(Garch3@forecast$VaR,
                  as.Date(rownames(Garch3@forecast$VaR)))[, 1]

# plotting the results
plot(cbind(tail(data, forecast_horizon), Garch3@forecast$density$Sigma))

# d) filtered historical simulation 
FHS <- rollapplyr(data, training_sample_len,
                      function(data) {
                        fit <- ugarchfit(specnorm1, data)
                        res <- rugarch::residuals(fit, standardize = T)
                        hat <-
                          sqrt(
                            fit@fit$coef['omega'] + fit@fit$coef['alpha1'] * tail(rugarch::residuals(fit), 1) ^
                              2 + fit@fit$coef['beta1'] * tail(sigma(fit), 1) ^ 2
                          )
                        draw <- sample(res, 20000, replace = T)
                        draw_var <- quantile(draw, p_value)
                        var <- draw_var * as.numeric(hat)
                        return(var)
                      })
testFHS <- tail(lag(FHS, 1), forecast_horizon)

# e) Does the quality of the models change? Are some approaches much better than 
# the others? What caused such behavior? 

testRealised <- last(data, 250)
var_predictions <- cbind(testHS,
                         testFHS,
                         testGarch1,
                         testGarch2,
                         testGarch3)
colnames(var_predictions) <-
  c("HS", "FHS", "GARCH Norm", "GARCH SNORM", "GARCH ST")

# function to calculate basic number of exceptions
excess_count <- function(var, true) {
  # if VaR > true realizaion (lower in absolute terms) then we have an exception
  return(sum(ifelse(coredata(var) > coredata(true), 1, 0)))
}

# we apply this function to each column in out VaR estimates object
sapply(var_predictions, excess_count, true = testRealised)

# we can also run Kupiec and Christoffersen tests to see whether conditional and
# unconditional hypotheses should be rejected or not
VaRTest(p_value, testRealised, testHS)

# let's apply this function to each of our VaR forecasts
sapply(var_predictions, function(var) {
  c(
    "Kupiec"=VaRTest(p_value, testRealised, var)$uc.Decision,
    "Christoffersen"=VaRTest(p_value, testRealised, var)$cc.Decision
  )
})

# Based on the provided results, the quality of the models does not significantly 
# change across different approaches (historical simulation, GARCH models, and 
# filtered historical simulation). No single approach is consistently much better 
# than the others, as indicated by the number of exceedances and the outcomes of 
# the Kupiec and Christoffersen tests. The similar performance suggests that for 
# the asset and time period analyzed, differences in model assumptions (e.g., 
# distributional assumptions in GARCH models) do not drastically affect the 
# accuracy of VaR estimates. This behavior is likely caused by the characteristics 
# of the asset's return distribution and the efficacy of each modeling approach 
# in capturing the underlying risk within the specified confidence level and 
# forecast horizon.

# f) Plot the VaR estimates together with data.
ggplot(testRealised, aes(index(testRealised), indeks)) +
  geom_line(aes(y = testGarch1, colour = "Garch NORM"), size = 1) +
  geom_line(aes(y = testGarch2, colour = "Garch SNORM"), size = 1) +
  geom_line(aes(y = testGarch3, colour = "Garch ST"), size = 1) +
  geom_line(aes(y = testHS, colour = "HS"), size = 1) +
  geom_line(aes(y = testFHS, colour = "FHS"), size = 1) +
  geom_point(aes(y = testRealised), size = 1) +
  scale_x_date(date_minor_breaks = "1 day") +  scale_colour_manual(
    "",
    breaks = c("Garch NORM", "Garch SNORM", "Garch ST", "HS", "FHS"),
    values = c("red", "green", "blue", "yellow", "brown")
  ) +
  xlab("") + ylab("Returns and VaRs")

# The VaR estimates from all methods seem to follow similar paths, suggesting 
# that the volatility patterns estimated by each model are close to each other.

# There are a number of exceedances where the actual returns fall below the VaR 
# estimates for all models, indicating days where the loss was greater than predicted.

# The HS method seems to provide the highest (least negative) VaR estimates, 
# which means it predicts a smaller loss compared to the GARCH-based methods. 
# This could either indicate a more conservative approach or a less sensitive 
# response to recent market volatility.




















