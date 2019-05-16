with open("news/title.txt","r",encoding="utf-8") as f:
    lines = f.readlines()[:2000]
    with open("news_sample/title.txt","w",encoding='utf-8') as g:
        g.write(''.join(lines))


with open("news/content.txt","r",encoding="utf-8") as f:
    lines = f.readlines()[:2000]
    with open("news_sample/content.txt","w",encoding='utf-8') as g:
        g.write(''.join(lines))
