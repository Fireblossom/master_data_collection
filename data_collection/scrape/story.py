import requests
import AdvancedHTMLParser
import json
import time
import asyncio
from aiohttp import ClientSession, TCPConnector
import re

# first step
'''
URL = 'https://sayitforward.org/stories/page/'
story_urls = []
parser = AdvancedHTMLParser.AdvancedHTMLParser()
for i in range(1, 82):
    url = URL + str(i) + '/'
    text = requests.get(url).text
    parser.parseStr(text)
    elements = list(parser.getElementsByClassName('read_more_button read_more_small'))[:-1]
    for element in elements:
        story_urls.append(element.getAttributesDict()['href'])
        print(story_urls[-1])
json.dump(story_urls, open('story_urls.json', 'w'), indent=2)
'''
# second step
story_urls = json.load(open('story_urls.json'))
# print(story_urls)
'''
tasks = []
async def hello(url):
    conn = TCPConnector(limit=1)
    async with ClientSession(connector=conn) as session:
        async with session.get(url) as response:
            # print(response)
            print('Another one story:%s' % time.time(), response.status)
            assert response.status == 200

            return await response.text()


def run():
    parser = AdvancedHTMLParser.AdvancedHTMLParser()
    texts = []
    for i in story_urls:
        # print(i)
        task = asyncio.ensure_future(hello(i))
        tasks.append(task)
    result = loop.run_until_complete(asyncio.gather(*tasks))
    for resp in result:
        parser.parseStr(resp)
        # print(str(task))
        element = parser.getElementsByClassName('entry-content')[0]
        # print(element)
        text = element.textContent
        text = text.replace('\n', '')
        text = text.replace('\t', '')
        text = text.replace('<!-- www.crestaproject.com Social Button in Content Start -->', '')
        text = text.replace('<!-- www.crestaproject.com Social Button in Content End -->', '')
        texts.append(text)
    return texts


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    story = run()
    sentences = []
    for s in story:
        sentences += re.split(r'[\.\?!]', s)
    print(sentences)
    json.dump(sentences, open('story.json', 'w'), indent=2)
'''
sentences = []
parser = AdvancedHTMLParser.AdvancedHTMLParser()
for url in story_urls:
    r = requests.get(url)
    print(r.status_code)
    while r.status_code != 200:
        time.sleep(5)
        r = requests.get(url)
        print(r.status_code)

    parser.parseStr(r.text)
    # print(str(task))
    element = parser.getElementsByClassName('entry-content')[0]
    # print(element)
    text = element.textContent
    text = text.replace('\n', '')
    text = text.replace('\t', '')
    text = text.replace('<!-- www.crestaproject.com Social Button in Content Start -->', '')
    text = text.replace('<!-- www.crestaproject.com Social Button in Content End -->', '')
    sentences += re.split(r'[\.\?!]', text)
    json.dump(sentences, open('story.json', 'w'), indent=2)