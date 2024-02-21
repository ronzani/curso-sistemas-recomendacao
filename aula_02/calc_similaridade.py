from fancy_matriz import FancyMatrix
import pandas as pd
import numpy as np


class SingleRatingMatrix(FancyMatrix):
    """Representa uma matrix de avaliação.

    As linhas indicam os usuários e nas colunas os itens.
    Os usuários e os itens podem ser quaisquer elementos imutáveis (hasheable).
    Mantemos um índice para cada um deles, que transforma o item ou usuário em um índice da matriz.
    A matriz é um numpy array.
    """
    def __init__(self, num_of_users, num_of_items):
        super().__init__(num_of_users, num_of_items)

    def get_user_index(self):
        return self.line_index

    def get_item_index(self):
        return self.column_index

    def get_all_user_ratings_for(self, item: int) -> np.array:
        """ Retorna o vetor de avaliações dos usuários para um determinado item. """
        item_index = self.get_item_index()
        if item not in item_index:
            raise KeyError(f"Item {item} has not been found in the index.")
        i = item_index[item]
        return self.data[:,i]

    def get_index_of_user(self, user):
        if user in self.line_index:
            return self.line_index[user]
        raise KeyError(f"Não há o usuário {user} no índice de usuários. Ele fez alguma avaliação?")

    def normalize(self):
        """Normaliza a matriz de avaliações subtraindo de cada avaliação a média da avaliação do usuário.
           Para o cálculo da média, não se considera os valores iguais a zero e apenas com usuários
           que avaliaram pelo menos 2 itens.
        """
        for user_index in self.get_user_index().values():
            sum_items = self.data[user_index, :].sum()
            non_zero_count = np.count_nonzero(self.data[user_index,:])
            # print(f"user={user_index}, sum_items={sum_items}, non_zero_count={non_zero_count}")
            # Normalizar um vetor com apenas um elemento não nulo vai zerá-lo.
            if non_zero_count < 2:
                continue
            non_zero_items = self.data[user_index, :].nonzero()
            self.data[user_index, non_zero_items] -= sum_items/non_zero_count
        return self

    @staticmethod
    def build_from_dataframe(ratings_df: pd.DataFrame, **kargs):
        """Builds a ratings matrix from a pandas dataframe.

        Args
        ratings_df : pandas.DataFrame containing the ratings.

        Allowed Keyword arguments are:
          item_column : str, optional
            Name of the column containing the items. The default is "WineID".
          user_column : str, optional
            Name of the column containing the users. The default is "UserID".

        Returns
          SingleRatingMatrix: Matrix containing the ratings.
        """
        item_column = kargs['item_column'] if 'item_column' in kargs else "WineID"
        user_column = kargs['user_column'] if 'user_column' in kargs else "UserID"
        rating_column = kargs['rating_column'] if 'rating_column' in kargs else "Rating"

        item_ids = ratings_df[item_column].unique()
        user_ids = ratings_df[user_column].unique()
        qty_items = len(item_ids)
        qty_users = len(user_ids)
        _m = SingleRatingMatrix(qty_users, qty_items)
        # display(user_ids)
        for i in range(qty_users):
            user = user_ids[i]
            # print("Fetching the ratings of user %d" % user)
            ratings_user = ratings_df[ratings_df[user_column] == user]
            # print(ratings_user)
            for index, row in ratings_user.iterrows():
                item = row[item_column]
                rating = row[rating_column]
                # print(wine, rating)
                _m[user, item] = rating
        return _m


class ItemSimMatrix(FancyMatrix):
    def __init__(self, number_of_items):
        self.verbose = False
        super().__init__(number_of_items, number_of_items)

    def get_item_index(self):
        return self.line_index

    def item_to_index(self, item):
        return self.get_item_index()[item]

    def index_to_item(self, item_index):
        return list(self.get_item_index().keys())[item_index]

    def __setitem__(self, key, value: np.float16):
        if isinstance(key, tuple):
            item_a, item_b = key
            super().__setitem__((item_a, item_b), value)
            super().__setitem__((item_b, item_a), value)
        else:
            raise KeyError("Key must be a tuple")

    @staticmethod
    def calc_adjusted_cos_sim_nozero(normalized_ratings_item_a, normalized_ratings_item_b):
        '''Cálculo da similaridade porém levando em consideração apenas os pares onde houve avaliação.
        '''
        #Primeiro os itens onde é zero (nao há avaliação) não entram na conta.
        # Assim, trabalhamos apenas com os valores não zerados.

        non_zero_indexes_of_item_a = set(np.flatnonzero(normalized_ratings_item_a))
        non_zero_indexes_of_item_b = set(np.flatnonzero(normalized_ratings_item_b))
        non_zero_indexes = list(non_zero_indexes_of_item_a & non_zero_indexes_of_item_b)
        #print(f" Selected indexes: {non_zero_indexes}")
        normalized_ratings_item_a = normalized_ratings_item_a[non_zero_indexes]
        normalized_ratings_item_b = normalized_ratings_item_b[non_zero_indexes]


        den_part_a = np.sqrt(sum(np.square(normalized_ratings_item_a)))
        #print(f"Vector of item a: {normalized_ratings_item_a}")
        #print(f"Vector of item b: {normalized_ratings_item_b}")
        num = sum(np.multiply(normalized_ratings_item_a,normalized_ratings_item_b))
        den_part_b = np.sqrt(sum(np.square(normalized_ratings_item_b)))
        #print(f"Formula: {num}/({den_part_a}.{den_part_b})")
        sim_a_b = num/(den_part_a*den_part_b)
        return sim_a_b

    @staticmethod
    def build_from_single_ratings_matrix(ratings_matrix: SingleRatingMatrix, verbose=False):
        item_index = ratings_matrix.get_item_index()
        M = ItemSimMatrix(len(item_index))
        for item_a in item_index:
            user_ratings_for_item_a = ratings_matrix.get_all_user_ratings_for(item_a)
            #den_part_a = np.sqrt(sum(np.square(user_ratings_for_item_a)))
            for item_b in item_index:
                if verbose:
                    print("--------------------------------------")
                    print(f"Calculando a similaridade entre {item_a} e {item_b}")
                user_ratings_for_item_b = ratings_matrix.get_all_user_ratings_for(item_b)
                #num = sum(np.multiply(user_ratings_for_item_a,user_ratings_for_item_b))
                #den_part_b = np.sqrt(sum(np.square(user_ratings_for_item_b)))
                sim_a_b = ItemSimMatrix.calc_adjusted_cos_sim_nozero(user_ratings_for_item_a, user_ratings_for_item_b)
                M[item_a, item_b] = sim_a_b
                if verbose:
                    print(f"Sim entre {item_a, item_b}={sim_a_b}")
        return M
