import numpy as np

print("loading model...")

word_dict = {}
model = {}
with open("sgns.sogou.word/sgns.sogou.word","r", encoding='utf-8') as f:
    f.readline()
    for line in f.readlines():
        word = line.split(' ')[0]
        vec = np.array(line.split(' ')[1:301],dtype=np.float32)
        model[word] = vec

def file_to_word(filename, dictionary):
    with open(filename, "r", encoding='utf-8') as f:
        _ = f.readline()
        while True:
            line = f.readline().strip('\n')
            if not line:
                break
            for word in line.split(' '):
                if word in dictionary: dictionary[word] += 1
                else: dictionary[word] = 1


import pdb
#pdb.set_trace()
print("loading sentences...")
file_to_word('news_sample/content.txt', word_dict)
file_to_word('news_sample/title.txt', word_dict)
# tmp = {}
# for word in word_dict.keys():
#     if word in model:
#         tmp[word] = word_dict[word]
#     else:
#         print(word)
# word_dict = tmp

ls = sorted(word_dict.items(), key=lambda x: x[1], reverse=1)

pdb.set_trace()
print("generating embeddings...")
with open("embedding/word2vec_norm.txt", "w", encoding='utf-8') as f:
    cnt = 0
    for word,_ in ls:
        if word in model.keys():
            cnt += 1
    f.write("{} {}\n".format(cnt, 300))
    for word, cnt in ls:
        if word in model.keys():
            f.write("%s " % word)
            out = ''
            for val in model[word]:
                out = out + ("%lf " % val)
            out = out[:-1] + '\n'
            f.write(out)