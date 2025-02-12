from urllib.parse import urljoin

base_url = "https://dev.truchiwoman.es"
pages = [
    "page1",
    "page2",
    "page3",
]

with open("sitemap.xml", "w") as f:
    f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    f.write('<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n')

    for page in pages:
        url = urljoin(base_url, page)
        f.write(f"  <url>\n    <loc>{url}</loc>\n  </url>\n")

    f.write('</urlset>')
