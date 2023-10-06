import commune as c


s = c.module('subspace')()

urls = s.urls()

c.print(urls)
bad_urls  = []

for url in urls:
    try:
        s.set_network(url=url)
        c.print(s.block) 
        c.print(url, color='green')

    except Exception as e:
        bad_urls += [url]
        c.print(e, color='red')
        continue

c.print(bad_urls)