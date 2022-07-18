using CSV
using DataFrames
using Lathe.preprocess: TrainTestSplit
using CSV
using DataFrames
using MLJ
using GLM
using Lasso
using MLBase




df = DataFrame(CSV.File("/Users/hilsts/Documents/linear-models-benchmark/data/song_data.csv"))
df = select!(df, Not(:song_name))
train, test = TrainTestSplit(df,.75)
X_train = convert(Matrix, train[:, Not(:song_popularity)])
y_train = train.song_popularity
X_test = convert(Matrix, test[:, Not(:song_popularity)])

function elastic_net(X_train, y_train, X_test)

    en = fit(LassoPath, X_train, y_train)
    predict(en, X_test)
end

@time elastic_net(X_train, y_train, X_test)


