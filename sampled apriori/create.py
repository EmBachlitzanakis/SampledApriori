import pandas as pd
def readMovies():
    movies_df = pd.DataFrame(pd.read_csv('movies.csv'),columns=['movieId','title','genres'])
    #print(movies_df)
   # print(movies_df['movieId'])
    return movies_df
def TriangularMatrixOfPairsCounters():
    movies_df=readMovies()
    pleiada=[]
    mikros=[]
    print(len(movies_df['movieId']))
    for i in range (9742):
        k=i+1
        for j in range (k,9742):
            mikros.append(movies_df['movieId'][i])
            mikros.append(movies_df['movieId'][j])
        pleiada.append(mikros)
        mikros=[]
    print(pleiada)
TriangularMatrixOfPairsCounters()
