import os
import httplib2
from urllib.parse import urlparse


def get_link_meta(url):
    if not url:
        return 'template'
    return urlparse(url).hostname


log_file = '{}__{}'.format(hostname, args.l)
