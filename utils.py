import os
from os import path
import cv2
from time import sleep
from helper import LoggerManager as mylogger
from pymongo import MongoClient
from datetime import datetime

def get_timecode(sec, precision=3, use_rounding=True):
    """
    Get time from second to time code

    :param sec: Time in seconde
    :param precision: The number of digit after the comma on the seconde number
    :param use_rounding: If true round the seconde number else don't
    :return: Time in time code format
    """

    # Compute hours and minutes based off of seconds, and update seconds.
    secs = sec
    base = 60.0 * 60.0
    hrs = int(secs / base)
    secs -= (hrs * base)
    base = 60.0
    mins = int(secs / base)
    secs -= (mins * base)
    # Convert seconds into string based on required precision.
    if precision > 0:
        if use_rounding:
            secs = round(secs, precision)
        msec = format(secs, '.%df' % precision)[-precision:]
        secs = '%02d.%s' % (int(secs), msec)
    else:
        secs = '%02d' % int(round(secs, 0)) if use_rounding else '%02d' % int(secs)
    # Return hours, minutes, and seconds as a formatted timecode string.
    return '%02d:%02d:%s' % (hrs, mins, secs)

def create_dir(location, dir_name):
    """
    Create a directory in a specific location

    :param location: The location where to create the new folder
    :param dir_name: the name of folder to create
    :return: The path of the new directory created and the name of the new directory
    """
    directory = path.join(location, dir_name)
    try:
        if not path.exists("{}".format(directory)):
            #mylogger.debug("{} path don't exists.. creating.. ".format(directory))
            os.mkdir("{}".format(directory))  # , 0o777)
            # pathlib.Path("{}".format(directory)).mkdir(parents=True, exist_ok=True)
        else:
            date = str(datetime.now().date().strftime('%Y-%m-%d'))
            time = str(datetime.now().time().strftime('%H-%M-%S-%f'))
            directory = str(directory) + "_" + date + "_" + time
            dir_name = dir_name + "_" + date + "_" + time
            #mylogger.debug("{} path don't exists.. creating.. ".format(directory))
            os.mkdir("{}".format(directory))  # , 0o777)
            # pathlib.Path("{}".format(directory)).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        mylogger.error("file creation: {}".format(e))
    return directory, dir_name
        
def check_folder_exist(folder):
    return path.exists(folder)
    
    
def get_fps(video_file):
    fps = 0.0
    video = cv2.VideoCapture(video_file)
    fps = video.get(cv2.CAP_PROP_FPS)
    video.release()
    return fps
    
def get_local_db_connection(collection_name):
    client = MongoClient('localhost', 27017)
    db = client.pamaArchive_dbb
    collection = db.collection_name
    return client, collection
    
def get_db_connection(yaml, collection_name):
    """
    Get the client and the collection from Mongo DB server
    :return: The client and the profile collection
    """
    try:
        hosts = ""
        for i in yaml.get_databases():
            if hosts != "": hosts += ","
            hosts = hosts + str(i[0]) + ":" + str(i[1])
        client = MongoClient(hosts,
                             ssl=True,
                             ssl_ca_certs=yaml.ca_certificate,
                             ssl_match_hostname=False,
                             authSource=yaml.auth_source_db,
                             username=yaml.db_user,
                             password=yaml.db_password,
                             authMechanism='SCRAM-SHA-1')
        mylogger.debug("{}".format(client.server_info()))
        db = client['{}'.format(yaml.db_name)]
        collection = db['{}'.format(collection_name)]

        return client, collection
    except Exception as e:
        mylogger.error("{}".format(e))
        
def insert_profile(customer_id, client, profile_collection, yaml):

    techno = []
    pi_config = []

    if yaml.techno[0]['name'] != 'null':

        for tech in yaml.techno:
            d = {"name": str(tech["name"]), "steps": ["3"]}
            techno.append(d)

    if yaml.pi_config[0]['name'] != 'null':

        for pi in yaml.pi_config:
            pi_config.append(str(pi["name"]))

    for _ in range(20):
        try:
            profile = {
                "customer_id": customer_id,
                "directory": "",
                "pi_config": pi_config,
                "techno": techno,
                "arena":""
            }
            mylogger.info("Profile for video {}: {}".format(customer_id, profile))
            profile_collection.insert_one(profile)
            mylogger.debug("Inserted new customer_id: {}".format(customer_id))
            break
        except Exception as e:
            mylogger.error("Mongodb AutoReconnect: {}".format(e))
            sleep(20)

    client.close()
    
def save_frame_doc(doc, client, image_collection):
    inserted = False
    try:
        for _ in range(10):
            try:
                collection.insert_one(doc)
                inserted = True
                break
            except Exception as e:
                mylogger.info("{} MongoDB retry to save {} in the database".format(e, doc))
                sleep(20)
    except Exception as e:
        mylogger.error("{}".format(e))
    return inserted
