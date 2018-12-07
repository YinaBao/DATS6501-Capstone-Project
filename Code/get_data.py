import wget
import os

urls = 'https://s3.amazonaws.com/gwu.dats6501.rawdata/yina_bao/clean_combined_data.csv'
wget.download(urls)