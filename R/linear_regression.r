DATA_PATH = "/Users/hilsts/Documents/linear-models-benchmark/data/"
df <- read.csv(paste(DATA_PATH,"song_data.csv", sep = ""))

set.seed(100)

df$song_name <- NULL

index = sample(1:nrow(df), 0.75 * nrow(df))

train = df[index, ]
test = df[-index, ]

linear_regression <- function(train, test) {
    lr = lm(song_popularity~., data = train)
    pred1 <- predict(lr, newdata = test)
}

start_time <- Sys.time()
linear_regression(train, test)
end_time <- Sys.time()
print(end_time - start_time)