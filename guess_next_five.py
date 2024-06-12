#!/usr/bin/env python3

"""guess next five"""

from fastai.collab import *
from fastai.tabular.all import *

# Load our dataframe.
ratings_df = pd.read_pickle('./mini_ratings-df.pkl')

if __name__ == "__main__":
    learn_inf = load_learner('./export.pkl')
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
