library(glmnet)

DATA_PATH = "/Users/hilsts/Documents/linear-models-benchmark/data/"
df <- read.csv(paste(DATA_PATH,"song_data.csv", sep = ""))

set.seed(100)

df$song_name <- NULL

index = sample(1:nrow(df), 0.75 * nrow(df))

train = df[index, ]
test = df[-index, ]

x = as.matrix(train)
y_train = train$song_popularity

x_test = as.matrix(test)
y_test = test$song_popularity

lambdas <- 10^seq(2, -3, by = -.1)

elastic_net <- function(x, y_train, x_test) {
    en <- cv.glmnet(x, y_train, alpha = 0.5, lambda = lambdas)
    optimal_lambda <- en$lambda.min
    predictions_test <- predict(en, s = optimal_lambda, newx = x_test)
}

start_time <- Sys.time()
elastic_net(x, y_train, x_test)
end_time <- Sys.time()
print(end_time - start_time)