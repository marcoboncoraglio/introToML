from urllib.error import HTTPError
from urllib.request import urlopen, Request, urlretrieve
from bs4 import BeautifulSoup
import os

turtleUrl = 'http://images.search.yahoo.com/search/images;_ylt=AwrEwM9Q05dazr8AnoOLuLkF;_ylc=X1MDOTYwNTc0ODMEX3IDMgRiY2sDMmIwZm01MWQ3dnB1NiUyNmIlM0QzJTI2cyUzRDdlBGZyAwRncHJpZANmNUhJUzVLQlExT3ozeWQzUXhmZ1BBBG10ZXN0aWQDbnVsbARuX3N1Z2cDMTAEb3JpZ2luA2ltYWdlcy5zZWFyY2gueWFob28uY29tBHBvcwMwBHBxc3RyAwRwcXN0cmwDBHFzdHJsAzgEcXVlcnkDdG9ydG9pc2UEdF9zdG1wAzE1MTk4OTk0NzcEdnRlc3RpZANCNTQ0Mg--?gprid=f5HIS5KBQ1Oz3yd3QxfgPA&pvid=pf.lHjEwLjElgfYoWn_nxgj6MjAwMQAAAAAsKWji&fr2=sb-top-images.search.yahoo.com&p=tortoise&ei=UTF-8&iscqry=&fr=sfp'
nonTurtleUrl = 'http://images.search.yahoo.com/search/images;_ylt=AwrEwSgb2ZdarjoABCCJzbkF;_ylu=X3oDMTBsZ29xY3ZzBHNlYwNzZWFyY2gEc2xrA2J1dHRvbg--;_ylc=X1MDOTYwNjI4NTcEX3IDMgRhY3RuA2NsawRiY2sDMmIwZm01MWQ3dnB1NiUyNmIlM0QzJTI2cyUzRDdlBGNzcmNwdmlkA01UOTk3akV3TGpFbGdmWW9Xbl9ueGhHNE1qQXdNUUFBQUFDRWh5NjIEZnIDc2ZwBGZyMgNzYS1ncARncHJpZANJZW5zYVFsRFJ0LnJ5djk0MXpNN3RBBG10ZXN0aWQDVUkwMSUzREI1NDQyBG5fc3VnZwMxMARvcmlnaW4DaW1hZ2VzLnNlYXJjaC55YWhvby5jb20EcG9zAzAEcHFzdHIDBHBxc3RybAMEcXN0cmwDMTAEcXVlcnkDbGFuZHNjYXBlcwR0X3N0bXADMTUxOTkwMDk4MQR2dGVzdGlkA0I1NDQy?gprid=IensaQlDRt.ryv941zM7tA&pvid=MT997jEwLjElgfYoWn_nxhG4MjAwMQAAAACEhy62&p=landscapes&fr=sfp&fr2=sb-top-images.search.yahoo.com&ei=UTF-8&n=60&x=wrt'

# run only first time
def prepareFilestructure():
    os.mkdir('./crawlerImgs')
    os.mkdir('./crawlerImgs/turtle')
    os.mkdir('./crawlerImgs/nonturtle')

prepareFilestructure()

# Download turtle pictures
req = Request(turtleUrl)
try:
    connection = urlopen(req)
except HTTPError as e:
    print (e.fp.read())

page = connection.read()
soup = BeautifulSoup(page, 'html.parser')

counter = 0
print('downloading pictures of turtles...')
for img in set(soup.find_all('img')):
    imgLink = str(img.get('src'))
    if imgLink != 'None':
        urlretrieve(imgLink, './crawlerImgs/turtle/turtle.' + str(counter) + '.jpg')
        counter += 1
    if counter == 100:
        break

connection.close()

# Download non turtle pictures
req = Request(nonTurtleUrl)

try:
    connection = urlopen(req)
except HTTPError as e:
    print (e.fp.read())

page = connection.read()
soup = BeautifulSoup(page, 'html.parser')

counter = 0
print('downloading pictures of landscapes(non-turtles)...')
for img in set(soup.find_all('img')):
    imgLink = str(img.get('src'))
    if imgLink != 'None':
        urlretrieve(imgLink, './crawlerImgs/nonturtle/nonturtle.' + str(counter) + '.jpg')
        counter += 1
    if counter == 100:
        break

connection.close()
