import random

with open("news/content.txt","r",encoding="utf-8") as f:
    lines = random.sample(f.readlines(), 384)
    for line in lines:
        if line.strip(' \n')=='':
            line = '<sos>\n'
    with open("news_sample/content_test.txt","w",encoding='utf-8') as g:
        g.write(''.join(lines))
