from scipy import spatial
import pandas as pd
import random 

## Set a random seed for reproducibility
# random.seed(1337)

## Generate some random vectors
# embedding_01 = [random.uniform(-1, 1) for i in range(10)]
# embedding_02 = [random.uniform(-1, 1) for i in range(10)]

## Cosine similarity.  
## 1 - spatial.distance.cosine because higher = more similar
def similarity(a, b):
    return 1 - spatial.distance.cosine(a, b)


## Euclidian distance
## You'll definitely want to figure out a way of normalizing this.  One way is probably dist/max_dist 
# def similarity(a, b):
#     return spatial.distance.euclidean(a, b)

    

df = pd.read_csv('federal_samples.csv')
challenge = pd.read_csv('challenge.csv')
output = pd.DataFrame(columns=['challenge', 'sample', 'similarity'])

for i in range(len(challenge.index)):
    max = 0
    max_j = 0
    embd0 = [float(x) for x in challenge['embeddings'][i][1:-1].split(", ")]
    for j in range(len(df.index)):
        embd1 = [float(x) for x in df['embeddings'][j][1:-1].split(", ")]
        if(similarity(embd0, embd1) > max):
            max = similarity(embd0, embd1)
            max_j = j
        # print(f'Challenge: {i} to Sample: {j}     {similarity(embd0, embd1)}')
        output.loc[len(output.index)] = [i, j, similarity(embd0, embd1)]
    print(f'Max for #{i} is sample: {max_j} with {max}\ntext: {df["text"][max_j]}\n\n')
    
output = output.sort_values(by='challenge')
output = output.sort_values(by='similarity',ascending=False)
output.to_csv('out2.csv')





        



## See https://docs.scipy.org/doc/scipy/reference/spatial.distance.html#module-scipy.spatial.distance for other distance measures you might be able to leverage.

