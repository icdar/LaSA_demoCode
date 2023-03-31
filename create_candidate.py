import numpy as np
import codecs, sys
import random
from dict_trie import Trie
# from editdistance import eval

vocabFile = codecs.open(sys.argv[1],'r',encoding='utf8')
dictionary = vocabFile.read().split()

print("Total words: ", len(dictionary))
random.shuffle(dictionary)
trie = Trie(dictionary)

def give_candidates(target, how_many=10):
  candidates = list(trie.all_levenshtein(target, 3))
  candidates = list(set(candidates)-set([target]))
  candidates.insert(0,target)
  while len(candidates) < how_many:
      candidates.append("###")
  candidates = candidates[:how_many]
  return candidates

candidates = []
i = 1
file = open(sys.argv[2], 'w')
for word in dictionary:
    if i % 1000 == 0:
        for c in candidates:
            # print(c)
            for w in range(len(c)):
                if w == len(c)-1:
                    file.write(c[w]+'\n')
                else:
                    file.write(c[w]+',')
        candidates = []
        print("finished:",i,flush=True)
    i+=1
    candidates.append(give_candidates(word))

file.close()