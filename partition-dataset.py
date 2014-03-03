
import pandas as pd
import operator
df = pd.read_csv('../datasets/movielens-synthesized/ratings-synthesized-1m.csv')


counts = {}
user_ratings = df.groupby('user')


for k, df in user_ratings:
    # do something with group
    counts[k] = df['user'].count()

print "num users", len(counts)

sorted_x = sorted(counts.iteritems(), key=operator.itemgetter(0), reverse=True)

f = open('myfile','w')

for (user,count) in sorted_x[:400]:
   for u, rating in user_ratings.set_index('user',inplace=True)[user]:
       f.write('') # python will convert \n to os.linesep

f.close()

