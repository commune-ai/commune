import os
import sys
import httplib2
from urllib.parse import urlparse

from random import choice
from argparse import ArgumentParser
from string import ascii_uppercase as au, \
                    ascii_lowercase as al, \
                    digits

from commune..model.beautifulsoap.image_fetcher.helpers import get_html_files, get_image_name, inline_progress, \
    get_short_image_link, progress_bar, fetch_images, terminal_fetch_remote_asset

http_client=httplib2.Http()
current_dir=os.getcwd()
url_length="long"
file_extension="png"
output_dir="images"

def run_fetcher(contents, stdout_file, output_directory):
    if not os.path.exists('images'):
        os.makedirs('images')

    if not os.path.exists('images/{}'.format(output_directory)):
        os.makedirs('images/{}'.format(output_directory))

    print('Fetching Images...')
    fetched_images = fetch_images(contents)
    frontier = []

    count = 0
    total = len(fetched_images)
    for (img, attr_type) in fetched_images:
        inline_progress(count, total, 'FETCH IN PROGRESS')
        long_img = img
        short_img = get_short_image_link(img)
        if img not in frontier:
            fetch_img = long_img
            frontier += [img]
            if url_length == 'short':
                fetch_img = short_img
            output_filename = get_image_name(short_img, count, file_extension)
            image_path = '{}/images/{}/{}'.format(current_dir, output_directory, output_filename)

            terminal_fetch_remote_asset(fetch_img, stdout_file, image_path)
        count += 1
    total_bar = progress_bar(100, 100)
    sys.stdout.flush()
    sys.stdout.write('[{}] {}% ...{}\n'.format(total_bar, 100.0, 'IMAGES SAVED! (SUCCESS)'))
    return

def get_link_meta(url):
    if not url:
        return 'template'
    return urlparse(url).hostname

def generate_filename():
    random_chars = [choice(au + al + digits) for _ in range(16)]
    filename = ''.join(random_chars)
    return '{}.txt'.format(filename)

def setup_fetcher(url):
    if not os.path.exists('logs'):
        os.makedirs('logs')

    hostname = get_link_meta(url)
    log_file = '{}__{}'.format(hostname, generate_filename())

    stdout_file = '{}/logs/{}'.format(current_dir, log_file)

    if not url:
        files = get_html_files()
        for file_path in files:
            print('Opening File with Path:', file_path)
            with open(file_path) as html_file:
                file_info = html_file.read()
                file_basename = os.path.basename(file_path)
                new_output_dir = file_basename.split('.')[0]
                run_fetcher(file_info, stdout_file, new_output_dir)
        return

    _, response = http_client.request(url)
    return run_fetcher(response, stdout_file, output_dir)
