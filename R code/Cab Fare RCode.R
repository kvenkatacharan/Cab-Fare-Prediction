rm(list = ls(all=T))
install.packages(chron)
install.packages(geosphere)
install.packages(xgboost)
############################## importing required libaries ##########################################################
library(geosphere)
library(ggplot2)
library(scales)
library(psych)
library(gplots)
require(xgboost)
library(gridExtra)
library(DMwR)
library(corrgram)
library(forecast)
library(outliers)
library(plyr)
library(lubridate)
library(chron)
library(distHaversine)
library(caret)
library(randomForest)
###########################################Explore the data##########################################
train=read.csv("train_cab.csv", header = T, na.strings = c(" ", "", "NA"))
str(train)
head(train)

####### extracting  hour , month,date,year dday of the week from the pickup datetime
train$pickup_date=as.Date(train$pickup_datetime)
dtimes = c(as.character(train$pickup_datetime))
dtparts = t(as.data.frame(strsplit(dtimes,' ')))
row.names(dtparts) = NULL
thetimes = chron(dates=dtparts[,1],times=dtparts[,2],format=c('y-m-d','h:m:s'))

train$pickup_hour =hour(thetimes)
train$pickup_month=month(thetimes)	
train$pickup_year=year(thetimes)
train$pickup_day_of_week= wday(thetimes)
train$pickup_day=day(thetimes)
head(train)

col=colnames(train)
drops="pickup_datetime"
train=train[,!(col%in% drops)] # droping pickup_datetime


#########################################  Missing Value Analysis ################################ 

mis_val=data.frame(apply(train, 2,function(x){sum(is.na(x))}))
mis_val$columns=row.names(mis_val)
names(mis_val)[1] =  "missing_percentage"
mis_val$missing_percentage= (mis_val$missing_percentage/nrow(train))*100
mis_val=mis_val[order(-mis_val$missing_percentage),]
row.names(mis_val) = NULL
mis_val = mis_val[,c(2,1)]
write.csv(mis_val, "Miising_perc.csv", row.names = F)
mis_val

train=train[!(is.na(train$pickup_date) ),]

##Data Manupulation;
train$fare_amount=as.numeric(as.character(train$fare_amount))
train$pickup_date=as.factor(train$pickup_date)

#convert string categories into factor numeric
for(i in 1:ncol(train)){
  
  if(class(train[,i]) == 'factor'){
    
    train[,i] = factor(train[,i], labels=(1:length(levels(factor(train[,i])))))
    
     }
}


#kNN Imputation
train = knnImputation(train, k = 3)

#Confirming there are no na's left
sum(is.na(train))

#Structure of data after imputing
str(train)

#Summary of data
summary(train)
#########################################  Data Preparation ########################################
# # Let us round passenger_count
train$passenger_count=round(train$passenger_count)
# Let us round fare to 2 decimal places
train$fare_amount=round(train$fare_amount,2)

#'Maximum Fare: %d  Min Fare:  %d'
max(train$fare_amount)
min(train$fare_amount)
# removing any -negative values
train[,"fare_amount"]=train[,"fare_amount"][ train[,"fare_amount"] >=0]

#Let us set the boundary for the train data also based on test data lat lng boundaries.We will mark the outlier locations as 1 and remove them for further analysis
boundary <- c(-74.263242, 40.573143, -72.986532,41.709555)
names(boundary) <- c('min_lng','min_lat','max_lng',  'max_lat')

train[!((train$pickup_longitude >= boundary['min_lng'] ) & (train$pickup_longitude <= boundary['max_lng']) &
              (train$pickup_latitude >= boundary['min_lat']) & (train$pickup_latitude <= boundary['max_lat']) &
              (train$dropoff_longitude >= boundary['min_lng']) & (train$dropoff_longitude <= boundary['max_lng']) &
              (train$dropoff_latitude >=boundary['min_lat']) & (train$dropoff_latitude <= boundary['max_lat'])),'is_outlier_loc']=1

train[((train$pickup_longitude >= boundary['min_lng'] ) & (train$pickup_longitude <= boundary['max_lng']) &
          (train$pickup_latitude >= boundary['min_lat']) & (train$pickup_latitude <= boundary['max_lat']) &
          (train$dropoff_longitude >= boundary['min_lng']) & (train$dropoff_longitude <= boundary['max_lng']) &
          (train$dropoff_latitude >=boundary['min_lat']) & (train$dropoff_latitude <= boundary['max_lat'])),'is_outlier_loc']=0

# Outlier vs Non Outlier Counts
table(train$is_outlier_loc)

# Let us drop rows, where location is outlier
train=train[which(train['is_outlier_loc']==0),]
drops="is_outlier_loc"
train=train[,!(colnames(train)%in% drops)] 

head(train)


# trip distance 
#calculate trip distance in miles
haversine <- function(lat_from, lon_from, lat_to, lon_to, r = 6378137){
  radians <- pi/180
  lat_to <- lat_to * radians
  lat_from <- lat_from * radians
  lon_to <- lon_to * radians
  lon_from <- lon_from * radians
  dLat <- (lat_to - lat_from)
  dLon <- (lon_to - lon_from)
  a <- (sin(dLat/2)^2) + (cos(lat_from) * cos(lat_to)) * (sin(dLon/2)^2)
  return(2 * atan2(sqrt(a), sqrt(1 - a)) * r)
}

train=train %>% mutate(trip_distance = distHaversine(cbind(pickup_longitude, pickup_latitude), cbind(dropoff_longitude, dropoff_latitude))/1609.344)

max(train$trip_distance)

##################################### Outlier Analysis######################################################## ### ################ 

numeric_index=sapply(train,is.numeric)# selecting only numeric index
numeric_variables=c('fare_amount', 'passenger_count')
numeric_data= train[,numeric_variables]        

cnames = colnames(numeric_data)

for (i in 1:length(cnames))
{
  assign(paste0("gn",i), ggplot(aes_string(y = (cnames[i])), data = subset(train))+ 
           stat_boxplot(geom = "errorbar", width = 0.5) +
           geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,
                        outlier.size=1, notch=FALSE) +
           theme(legend.position="bottom")+
           labs(y=cnames[i])+
           ggtitle(paste("Box plot of ",cnames[i])))
}

# ## Plotting together
gridExtra::grid.arrange(gn1,gn2,ncol=2)

max(train$passenger_count)
# here passengers count is 5345.0 
#there are max of 6 passengers in test data so lets remove records having passengers count more than  6
dim(train[(train$passenger_count > 6 ),])
train=train[(train$passenger_count <= 6 ),]

#Replace all outliers with NA and impute
val =train[,'fare_amount'][train[,'fare_amount'] %in% boxplot.stats(train[,'fare_amount'])$out]
train[,'fare_amount'][train[,'fare_amount'] %in% val] = NA

# knnimputation
train = knnImputation(train, k = 3)
sum(is.na(train))
train$fare_amount=round(train$fare_amount,2)

######################################## visualization##########################################



#Visualizing data
#Univariate analysis of important continuous variable
#Making the density plot
g1 = ggplot(train, aes(x = fare_amount)) + geom_density()
g2 = ggplot(train, aes(x = passenger_count)) + geom_density()

grid.arrange(g1, g2, ncol=2)


groupAndPlot=function(data,groupby_key,value,aggregate='mean'){
  
              aggregate_data=aggregate(x = data[,value], 
                                       by = list(train[,groupby_key]), 
                                       FUN = mean)
              colnames(aggregate_data)=c(groupby_key,value)
              aggregate_data[,value]=round(aggregate_data[,value],2)
              aggregate_data[,groupby_key]=factor(aggregate_data[,groupby_key])
              data[,groupby_key]=factor(data[,groupby_key])
              
              g1= ggplot(data=data, aes_string(x=groupby_key,fill = groupby_key)) +
                geom_bar(stat="count",width=0.5, fill="steelblue")+geom_label(stat='count',aes(label=..count..), size=2)+
                theme_minimal(base_size = 10)
              
              g2=ggplot(data=aggregate_data, aes_string(x=groupby_key, y=value)) +
                geom_bar(stat="identity",width=0.5, fill="steelblue")+
                geom_label(aes_string(label=value), size=2)+
                theme_minimal(base_size = 10)
              
                  grid.arrange(g1, g2, ncol=1)
}

groupAndPlot(train,'pickup_year','fare_amount')
# Avg Fare amount has beern increasing over the years.


groupAndPlot(train,'pickup_month','fare_amount')
# Fares across months are fairly constant, though number of trips are lower from june to decemeber


groupAndPlot(train,'pickup_day_of_week','fare_amount')
# Sunday=1, Monday=2....Saturday=7
# Saturday has low avg fare amount, compared to other days though there are a lot of trips of saturday. On sunday and monday though the number of trips are lower, avg fare amount is higher


groupAndPlot(train,'pickup_hour','fare_amount')
#The avg fare amount at 4am and 5 am are the higher while the number of trips at 4 am and 5 am are the least. The number of trips are highest in 18 and 19 hours


groupAndPlot(train,'passenger_count','fare_amount')
#There are trips with 0 passengers as well. In these cases are drop and pickup location the same? If so it would mean that passenger didnt take the cab after the cab arrived and a cancellation fee was charged - there were 56 such records.

#Fare amount is higher for 6 passengers .no of trips for 6 passengers is very low -there were 295 records

#write.csv(train, "train_cleaned.csv", row.names = T)

#train=read.csv("train_cleaned.csv") 
## Correlation Plot 
corrgram(train[,numeric_index], order = F,
         upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot")

################################# Cleaning test data  ##########################################

test=read.csv("test.csv", header = T, na.strings = c(" ", "", "NA"))

test$pickup_date=as.Date(test$pickup_datetime)
dtimes = c(as.character(test$pickup_datetime))
dtparts = t(as.data.frame(strsplit(dtimes,' ')))
row.names(dtparts) = NULL
thetimes = chron(dates=dtparts[,1],times=dtparts[,2],format=c('y-m-d','h:m:s'))

test$pickup_hour =hour(thetimes)
test$pickup_month=month(thetimes)	
test$pickup_year=year(thetimes)
test$pickup_day_of_week= wday(thetimes)
test$pickup_day=day(thetimes)

col=colnames(test)
drops="pickup_datetime"
test=test[,!(col%in% drops)] # droping pickup_datetime

# trip distance 
test=test %>% mutate(trip_distance = distHaversine(cbind(pickup_longitude, pickup_latitude), cbind(dropoff_longitude, dropoff_latitude))/1609.344)
test$pickup_date=as.factor(test$pickup_date)
str(test)
for(i in 1:ncol(test)){
  
  if(class(test[,i]) == 'factor'){
    
    test[,i] = factor(test[,i], labels=(1:length(levels(factor(test[,i])))))
    
  }
}
test$pickup_date= as.numeric(factor(test$pickup_date))
#write.csv(test, "test_cleaned.csv", row.names = T)

###############################   Process data for Modelling ##########################################


set.seed(1234)
train.index = createDataPartition(train$fare_amount, p = .80, list = FALSE)
mtrain =train[ train.index,]
mtest  =train[-train.index,]


RMSE = function(predicted, actual){
  sqrt(mean((predicted - actual)^2))
}


# ##################################### linear  regression model##########################################


lm_model = lm(fare_amount ~., data = mtrain)

#Summary of the model
summary(lm_model)

#Predict
lr_predictions = predict(lm_model, mtest[,2:13])
lr_rmse =RMSE(lr_predictions,mtest[,1])
lr_rmse

# ########################################### random forest model ##########################################

rf_model <-randomForest(fare_amount ~., # Standard formula notation
                         data=mtrain,  # Excclude 'id'
                         method="rf",              # randomForest
                         nodesize= 10,              # 10 data-points/node. Speeds modeling
                         ntree =500,               # Default 500. Reduced to speed up modeling
                         trControl=trainControl(method="repeatedcv", number=2,repeats=1),  # cross-validation strategy
                         tuneGrid = expand.grid(mtry = c(123))
)

rf_model
rf_predictions <- predict(rf_model,mtest[,2:13])
rf_rmse =RMSE(rf_predictions,mtest[,1])

rf_rmse

# ########################################### xgboost model  ##########################################

set.seed(123)
## Model parameters trained using xgb.cv function
xgbFit = xgboost(data = as.matrix(mtrain[, -1]), nfold = 5, label = as.matrix(mtrain$fare_amount), 
                 nrounds = 2200, verbose = FALSE, objective = "reg:linear", eval_metric = "rmse", 
                 nthread = 8, eta = 0.01, gamma = 0.0468, max_depth = 6, min_child_weight = 1.7817, 
                 subsample = 0.5213, colsample_bytree = 0.4603)
## print(xgbFit)

## Predictions
preds2 <- predict(xgbFit, newdata = as.matrix(mtest[, -1]))
xgb_rmse=RMSE( preds2,mtest[,1])



# Feature Importance
importance_matrix <- xgb.importance(colnames(mtrain[,-1]),model=xgbFit)
xgb.plot.importance(importance_matrix[1:10,])

xgb_rmse
rf_rmse
lr_rmse

#> xgb_rmse= 2.116817
#rf_rmse =2.136985
#lr_rmse =3.390334

# We will consider XGBOOST as  our  model
# ################################ final predictions using xgboost  ##########################################

final_predictions=predict(xgbFit, newdata = as.matrix(test))
final_predictions

