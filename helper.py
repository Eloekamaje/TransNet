#! /home/mpiuser/fassoo/fassoo-virtual-env/venv/bin/python3
# -*- coding: utf-8 -*-

from bson.json_util import dumps
import datetime
import os
from yaml import load
import time
from pymongo import MongoClient
import logging
from logging.handlers import TimedRotatingFileHandler
from threading import Thread

"""
 if you're running Linux there's a way to speed up Yaml parsing. By default, Python's yaml uses the Python parser.
 You have to tell it that you want to use PyYaml C parser. To install use following .. 
 $ apt-get install libyaml-dev
 $ pip install pyyaml --upgrade --force
"""
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


def get_time_sec(time_code):
    """
    Get time from time code to seconde

    :param time_code: Time in time code format
    :return: Time in seconde
    """
    if time_code == "":
        return 0.0
    else:
        hour = float(time_code.split(":")[0])
        min = float(time_code.split(":")[1])
        sec = float(time_code.split(":")[2])
        return (3600 * hour) + (60 * min) + sec


class Singleton(type):
    """
    singleton class for log manger.

    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances.keys():
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class LoggerManager(object):
    """
    LogManger is logging class.

    """
    __metaclass__ = Singleton

    #LOG_FILENAME = '/var/log/pama/keyframes_extraction.log'
    LOG_FILENAME ='/var/log/fassoo/kf/kf.log'
    mylogger = logging.getLogger(__name__)
    mylogger.setLevel(logging.DEBUG)  # DEBUG
    handler = logging.handlers.TimedRotatingFileHandler(LOG_FILENAME, 'midnight')
    handler.setLevel(logging.DEBUG)  # DEBUG
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    mylogger.addHandler(handler)

    @classmethod
    def error(cls, line):
        cls.mylogger.error("{}".format(line))

    @classmethod
    def info(cls, line):
        cls.mylogger.info("{}".format(line))

    @classmethod
    def debug(cls, line):
        cls.mylogger.debug("{}".format(line))


class YamlReader(object):
    """
    Yaml reader class

    """
    stream = open('config.yaml', 'r')
    data = load(stream, Loader=Loader)
    stream.close()
    database_servers = data['database_servers']
    replicaset = str((data['ca_certificate']['replica_set']))
    ca_certificate = os.path.join(str(data['ca_certificate']['path']),
                                  str(data['ca_certificate']['file']))
    db_user = str(data['ca_certificate']['db_user'])
    db_password = str(data['ca_certificate']['db_password'])
    auth_source_db = str(data['ca_certificate']['auth_source_db'])

    db_name = str(data['working_datebase']['db_name'])
    db_coll_images = str(data['working_datebase']['images_db_collection'])
    db_coll_profile = str(data['working_datebase']['profile_db_collection'])
    db_coll_user = str(data['working_datebase']['user_db_collection'])

    ssl_enabled = str(data['working_datebase']['ssl_enabled'])

    gpu_fraction = float(data['gpu']['fraction'])
    video_extensions = str(data['video']['extensions'])
    delta_time = float(data['delta_time'])
    insert_profile = int(data['insert_profile'])
    move_images_to_fassoo = int(data['move_images_to_fassoo'])
    customer_id = str(data['customers_id'])
    customer_name = str(data['customers_name'])

    location_incoming = str(data['location']['incoming'])
    location_outgoing = str(data['location']['outgoing'])
    location_fassoo = str(data['location']['fassoo'])

    es_index = str(data['es_index'])
    es_type = str(data['es_type'])

    '''api_host = str(data['api']['host'])
    api_port = str(data['api']['port'])'''

    techno = data['techno']
    pi_config = data['pi_config']

    '''sftp_customer = str(data['sftp']['customer'])
    sftp_host = str(data['sftp']['host'])
    sftp_port = int(data['sftp']['port'])
    sftp_directory = str(data['sftp']['directory'])
    sftp_user = str(data['sftp']['user'])
    sftp_pwd = str(data['sftp']['pwd'])'''

    def __init__(self):
        pass

    def get_databases(self):
        """
        get list of databases from YMAL

        :return: list of databases and its details
        """
        list_of_databases = []
        for db in self.database_servers:
            file = str(db['file'])
            path = str(db['path'])
            file = os.path.join(path, file)
            ip = str(db['ip'])
            port = str(db['port'])
            #LoggerManager.debug("{} {} {}".format(file, ip, port))
            list_of_databases.append([ip, port, file])
        #LoggerManager.debug("{}".format(list_of_databases))
        return list_of_databases

    def get_customer_names(self):
        """
        returns list of customer names

        :return: list of customer names
        """
        customer_list = []
        for customer in self.data['customers']:
            customer_list.append(customer['name'])
        #LoggerManager.debug("{}".format(customer_list))
        return customer_list

