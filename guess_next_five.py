#!/usr/bin/env python3

"""guess next five"""

from fastai.collab import *
from fastai.tabular.all import *


#We need a model to train the data on,
class DotProduct(Module):
    def __init__(self, n_users, n_items, n_factors, y_range=(-9.5, 10.5)):
        self.user_factors = Embedding(n_users, n_factors)
        self.user_bias = Embedding(n_users, 1)
        self.item_factors = Embedding(n_items, n_factors)
        self.item_bias = Embedding(n_items, 1)
        self.y_range = y_range
        pass
    
    def forward(self, x):
        users = self.user_factors(x[:,0])
        items = self.item_factors(x[:,1])
        res = (users * items).sum(dim=1, keepdim=True)
        res += self.user_bias(x[:,0]) + self.item_bias(x[:,1])
        return sigmoid_range(res, *self.y_range)



# Load our dataframe.
ratings_df = pd.read_pickle('./mini_ratings-df.pkl')

# Load our learner.
learn_inf = load_learner('./export-3-10.pkl')


if __name__ == "__main__":
    dls_inf = learn_inf.dls
    joke_factors_inf = learn_inf.model.item_factors.weight
    idx_int = 19
    cls_idx = tensor(dls_inf.classes['jokeId'].o2i[idx_int])
    int_joke_emb = joke_factors_inf[cls_idx, None]
    distances = nn.CosineSimilarity(dim=1)(joke_factors_inf, int_joke_emb)
    closest_emb_idx = distances.argsort(descending=True)[1:6] #Top 5 closest jokes.
    closest_idx = dls_inf.classes['jokeId'][closest_emb_idx]
    # Print the actual joke.
    print('The actual joke is:')
    print(ratings_df[ratings_df['jokeId'] == idx_int].head(1).jokeText)
    print('========================================')
    closest_jokes = ratings_df[ratings_df["jokeId"].isin(closest_idx)].drop_duplicates("jokeId")["jokeText"]
    # closest_jokes = closest_jokes.str.replace("\n", " ")
    print('The closest jokes are:')
    for joke in closest_jokes:
        print(joke, "\n\n")
