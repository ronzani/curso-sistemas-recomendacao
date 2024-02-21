from calc_similaridade import SingleRatingMatrix
import pandas as pd

df_test = pd.read_csv('simple.csv')

m = SingleRatingMatrix.build_from_dataframe(df_test, user_column='UserID', item_column='ItemID')
m.print("Matriz de avaliações")
m.normalize()
m.print("Matriz de avaliações normalizada")
