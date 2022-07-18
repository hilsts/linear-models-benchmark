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

function bench_linear_regression(train,test)

    linearRegressor = lm(Term(:song_popularity) ~ sum(Term.(Symbol.(names(train[:, Not(:song_popularity)])))), train)
    predict(linearRegressor, test)
    
end

@time bench_linear_regression(train, test)


