#!/usr/bin/python

import datetime
import os
import re
from multiprocessing import Pool

import click
import requests
from bs4 import BeautifulSoup

from lib.download_utils import download_file


class JPLData(requests.Session):
    BASE_URL = 'https://snow-data.jpl.nasa.gov'
    ARCHIVE_PATH = '-historic'
    TYPES = {
        'MODSCAG': BASE_URL + '/modscag',
        'MODDRFS': BASE_URL + '/moddrfs'
    }

    FILE_BASE_REGEX = 'MOD09GA[.]A[\d]{7}[.]'

    def __init__(self, username, password):
        super(JPLData, self).__init__()
        self.auth = requests.auth.HTTPDigestAuth(username, password)

    def get_index(self, source):
        self.get(self.TYPES[source])

    def requested_files_regex(self, tiles, file_types):
        regex = '(' + '|'.join(tiles) + ').*(' + '|'.join(file_types) + ')$'
        return re.compile(self.FILE_BASE_REGEX + regex, re.IGNORECASE)

    def __get_index_url(self, types, year, day):
        url = self.TYPES[types]
        if year < 2015:
            url += self.ARCHIVE_PATH
        url += '/' + str(year) + '/' + day + '/'

        return url

    def files_for_date_range(self, types, tiles, year, day_range, file_types):
        files = {}

        for day in day_range:
            print('Parsing download links for day: ' + str(day))
            day = str(day).rjust(3, '0')

            index_dir_url = self.__get_index_url(types, year, day)
            file_links = BeautifulSoup(
                self.get(index_dir_url).text, 'html.parser'
            ).find_all(
                'a', text=self.requested_files_regex(tiles, file_types)
            )

            [
                files.update({link.text: index_dir_url + link.attrs['href']})
                for link in file_links
            ]

        return files


def to_array(_ctx, _param, value):
    return value.split(',')


def validate_types(ctx, _param, value):
    if value not in JPLData.TYPES:
        print('Invalid data type')
        ctx.abort()
    else:
        return value


def get_from_jpl(username, password, name, url, download_folder):
    session = JPLData(username, password)
    return download_file(session, name, url, download_folder)


def parse_year(_ctx, _param, value):
    if value:
        return range(value, value + 1)
    else:
        return range(2000, datetime.date.today().year + 1)


@click.command()
@click.option('--username',
              prompt='Your username',
              help='Your EarthData username')
@click.password_option(confirmation_prompt=False,
                       help='Your EarthData password')
@click.option('--download-folder',
              type=click.Path(exists=True),
              prompt=True,
              help='The destination folder to store the files')
@click.option('--year',
              prompt=False,
              type=int,
              callback=parse_year,
              help='The year (YYYY) to download data from.'
                   'Leave blank to download all data starting from 2000')
@click.option('--day-from',
              prompt=True,
              type=int,
              help='The starting day to download data from')
@click.option('--day-to',
              prompt=True,
              type=int,
              help='The ending day to download data from')
@click.option('--types',
              prompt='The type of data - MODSCAG or MODDRFS',
              callback=validate_types,
              help='The type of data - MODSCAG or MODDRFS')
@click.option('--tiles',
              prompt=True,
              callback=to_array,
              help='List of tiles separated by comma (,)')
@click.option('--file-names',
              prompt=True,
              callback=to_array,
              help='Pattern of file to look for')
def data_download(**kwargs):
    session = JPLData(kwargs['username'], kwargs['password'])
    # To authenticate for the session
    session.get_index(kwargs['types'])

    days = range(kwargs['day_from'], kwargs['day_to'] + 1)

    for year in kwargs['year']:
        download_folder = os.path.join(kwargs['download_folder'], str(year))

        print('Downloading files: ' + ', '.join(kwargs['file_names']) +
              ' for tiles: ' + ', '.join(kwargs['tiles']) +
              ' in day range ' + str(kwargs['day_from']) +
              ' to ' + str(kwargs['day_to']) +
              ' for year ' + str(year))

        file_list = session.files_for_date_range(
            kwargs['types'],
            kwargs['tiles'],
            year,
            days,
            kwargs['file_names'],
        )

        print('Found ' + str(len(file_list)) + ' files to download')
        p = Pool(4)
        p_res = [
            p.apply_async(
                get_from_jpl,
                (kwargs['username'], kwargs['password'], name, url, download_folder)
            ) for name, url in file_list.items()
        ]
        [res.get() for res in p_res]


if __name__ == '__main__':
    data_download()
