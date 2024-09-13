import os
import random
import time
import requests
from bs4 import BeautifulSoup

# Headers to mimic a browser request (based on raw inspector data)
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'Accept-Encoding': 'gzip, deflate, br, zstd',
    'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'Referer': 'https://www.google.com/',
    'Cache-Control': 'max-age=0'
}

base_url = 'https://www.zara.com/us/en/woman-new-in-l1180.html?v1=2419517&initialBlockId=HOME2'
save_directory = './zara_images'
