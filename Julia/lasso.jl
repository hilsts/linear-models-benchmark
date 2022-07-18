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

function lasso(train,test)

    lasso = fit(LassoModel, Term(:song_popularity) ~ sum(Term.(Symbol.(names(train[:, Not(:song_popularity)])))), train)
    predict(lasso, test)
    
end

@time lasso(train, test)


