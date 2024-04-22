import os
import sys
import subprocess
from urllib.parse import urlparse

from bs4 import BeautifulSoup

html_path="html"
image_attributes=[
    'src', 'data-wood_src', 'data-large_image',
    'data-srcset', 'srcset', 'data-src', 'content'
]
filter_tags=[]
link_tags=['img', 'source', 'meta']

def terminal_fetch_remote_asset(asset_link, log_file, output_path):
    if sys.platform == 'win32':
        subprocess.call(['curl', asset_link, '-o', output_path],
                        stdout=open(log_file, 'a'),
                        stderr=subprocess.STDOUT)
    else:
        subprocess.call(['wget', asset_link, '-O', output_path],
                        stdout=open(log_file, 'a'),
                        stderr=subprocess.STDOUT)
    return


def get_url_without_extension(url):
    return url.split('/')[-2]


def is_html(filename):
    return os.path.isfile(os.path.join(html_path, filename)) and filename.endswith('.html')


def get_html_files():
    directory_info = os.listdir(html_path)
    return [os.path.join(html_path, f) for f in directory_info if is_html(f)]


def image_has_extension(image_path):
    image_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif', '.svg')
    return image_path\
        .lower()\
        .endswith(image_extensions)


def add_image_index(name, index):
    image_sub_names = name.split('.')
    image_full_name = '_'.join(image_sub_names[:-1])
    return '{}__{}.{}'.format(image_full_name, index, image_sub_names[-1])


def get_image_name(short_image_link, index, extension):
    image_names = short_image_link.split("/")
    image_name = image_names[-1]
    if not image_name:
        image_name = image_names[-2]
    if not image_has_extension(image_name):
        image_name = '{}.{}'.format(image_name, extension)
    return add_image_index(image_name, index)


def get_short_image_link(image_link):
    question_index = image_link.find('?')
    if question_index < 0:
        return image_link
    return image_link[:question_index]


def progress_bar(fill, total):
    return '=' * fill + '-' * (total - fill)


def inline_progress(count, total, status=''):
    bar_len = 100
    filled_len = int(round(bar_len * count / float(total)))
    percent = round(100.0 * count / float(total), 1)
    bar = progress_bar(filled_len, bar_len)
    sys.stdout.write('[{}] {}% ...{}\r'.format(bar, percent, status))
    sys.stdout.flush()


def get_image_information(images):
    all_images = []
    for image_index in range(len(images)):
        extracted_images = []
        image_tag = images[image_index]
        for attr in image_attributes:
            long_image_string = image_tag.get(attr, '')
            spaced_strings = long_image_string.split(' ')
            for string in spaced_strings:
                if 'https' in string or 'http' in string:
                    extracted_images += [(string, attr)]
        all_images += extracted_images
    return all_images


def fetch_images(html_text, filter_range=True):
    try:
        print('Starting Image Fetch...')
        if not filter_tags:
            filters = ['html']
        else:
            filters = filter_tags
        soup_tree = BeautifulSoup(html_text, 'html.parser')

        filter_images = []
        if not filter_range:
            for link_tag in link_tags:
                filter_images.extend(soup_tree.findAll(link_tag))
        else:
            for html_filter in filters:
                sub_tags = soup_tree.findAll(html_filter)
                for component in sub_tags:
                    for link_tag in link_tags:
                        filter_images.extend(component.findAll(link_tag))

        return get_image_information(filter_images)
    except ():
        print('Image Fetch Failed! Aborted!\n')
