import re
import os

def isnttrash(x):
    return not x in '　：？！'

with open("news_sample/title.txt","r",encoding='utf-8') as f:
    with open("news_sample/title.1.txt","w",encoding='utf-8') as g:
        for line in f.readlines():
            line = ''.join(filter(isnttrash, line))
            words = re.split(r'\s+', line)
            g.write("%s\n" % (' '.join(words)))
if os.path.exists("news_sample/title.txt"):
    os.remove("news_sample/title.txt")
os.rename("news_sample/title.1.txt","news_sample/title.txt")

with open("news_sample/content.txt","r",encoding='utf-8') as f:
    with open("news_sample/content.1.txt","w",encoding='utf-8') as g:
        for line in f.readlines():
            line = ''.join(filter(isnttrash, line))
            words = re.split(r'\s+', line)[:50]
            if len(list(filter(lambda x: x!='', words))) == 0:
                words=['<eos>']
            g.write("%s\n" % (' '.join(words)))
if os.path.exists("news_sample/content.txt"):
    os.remove("news_sample/content.txt")
os.rename("news_sample/content.1.txt","news_sample/content.txt")

with open("news_sample/content_test.txt","r",encoding='utf-8') as f:
    with open("news_sample/content_test.1.txt","w",encoding='utf-8') as g:
        for line in f.readlines():
            line = ''.join(filter(isnttrash, line))
            words = re.split(r'\s+', line)[:50]
            if len(list(filter(lambda x: x!='', words))) == 0:
                words=['<eos>']
            g.write("%s\n" % (' '.join(words)))
if os.path.exists("news_sample/content_test.txt"):
    os.remove("news_sample/content_test.txt")
os.rename("news_sample/content_test.1.txt","news_sample/content_test.txt")