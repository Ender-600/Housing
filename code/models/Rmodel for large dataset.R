#PART ONE
#Load data, Clean data
#Seperate data for model & test
install.packages("dplyr")
install.packages("caret")
install.packages("broom")
library(readxl)
library(dplyr)
library(caret)
library(scales)
library(broom)
setwd("/Users/xzy/Desktop")
Housing_1920Data_Points <- read_excel("Cleaned Zillow Data 1980pts.xlsx")

str(Housing_1920Data_Points$soldPrice)
Housing_1920Data_Points$soldPrice <- as.numeric(gsub("[\\$,]", "", Housing_1920Data_Points$soldPrice))
sum(is.na(Housing_1920Data_Points$soldPrice))
Housing_1920Data_Points <- Housing_1920Data_Points[!is.na(Housing_1920Data_Points$soldPrice), ]
sum(is.na(Housing_1920Data_Points$soldPrice))

Housing_Cleaned <- Housing_1920Data_Points %>% select(area, baths, beds, livingArea, rentZestimate, 
                                                      daysOnZillow, homeType, addressCity, Zipcode, soldPrice) %>%
na.omit() 

Housing_Cleaned$homeType <- as.factor(Housing_Cleaned$homeType)
Housing_Cleaned$addressCity <- as.factor(Housing_Cleaned$addressCity)
Housing_Cleaned$Zipcode <- as.factor(Housing_Cleaned$Zipcode)

set.seed(666)
Housing_Seperating_IDX <- createDataPartition(Housing_Cleaned$soldPrice, p = 0.85, list = FALSE)

Housing_Train <- Housing_Cleaned[Housing_Seperating_IDX, ]
Housing_Test  <- Housing_Cleaned[-Housing_Seperating_IDX, ]
nrow(Housing_Train)
nrow(Housing_Test)

#Part TWO
#Simple regressions for test
#================================================================================
#LM for core varibles: price ~ baths, beds, living area, Zestimate rent
Housing_lm_core <- lm(soldPrice ~ livingArea + beds + baths + rentZestimate, data = Housing_Train)
summary(Housing_lm_core)
glance(Housing_lm_core)
#weird, beds coefficient still negative
#this might because when holding other variables stable, increasing bdrms means
#decreasing living area of each bdrm.
lm(soldPrice ~ beds, data = Housing_Train)
#coefficient positive!

Housing_pred_core <- predict(Housing_lm_core, newdata = Housing_Test)

RMSE_core <- sqrt(mean((Housing_pred_core - Housing_Test$soldPrice)^2, na.rm = TRUE))
R2_core   <- cor(Housing_pred_core, Housing_Test$soldPrice, use = "complete.obs")^2
cat("Baseline RMSE:", round(RMSE_core, 2), "\n")
cat("Baseline R^2 :", round(R2_core, 4), "\n")

#Visualization shows the diff between predict and actual
results_df <- data.frame(
  Predicted = Housing_pred_core,
  Actual = Housing_Test$soldPrice
)

ggplot(results_df, aes(x = Predicted, y = Actual)) +
  geom_point(color = "steelblue", alpha = 0.6) +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed", linewidth = 1) +
  scale_x_continuous(labels = label_dollar(scale = 1)) +
  scale_y_continuous(labels = label_dollar(scale = 1)) +
  labs(
    title = "LM_4 Actual vs Predicted House Prices",
    x = "Predicted Price",
    y = "Actual Sold Price"
  ) +
  theme_minimal() + coord_equal()
#================================================================================
#LM: price ~ baths, beds, living area
Housing_lm_3 <- lm(soldPrice ~ livingArea + beds + baths, data = Housing_Train)
summary(Housing_lm_3)

Housing_pred_core <- predict(Housing_lm_core, newdata = Housing_Test)

RMSE_core <- sqrt(mean((Housing_pred_core - Housing_Test$soldPrice)^2, na.rm = TRUE))
R2_core   <- cor(Housing_pred_core, Housing_Test$soldPrice, use = "complete.obs")^2
cat("Baseline RMSE:", round(RMSE_core, 2), "\n")
cat("Baseline R^2 :", round(R2_core, 4), "\n")

#Visualization shows the diff between predict and actual
results_df <- data.frame(
  Predicted = Housing_pred_core,
  Actual = Housing_Test$soldPrice
)

ggplot(results_df, aes(x = Predicted, y = Actual)) +
  geom_point(color = "steelblue", alpha = 0.6) +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed", linewidth = 1) +
  scale_x_continuous(labels = label_dollar(scale = 1)) +
  scale_y_continuous(labels = label_dollar(scale = 1)) +
  labs(
    title = "LM_3 Actual vs Predicted House Prices",
    x = "Predicted Price",
    y = "Actual Sold Price"
  ) +
  theme_minimal() + coord_equal()
