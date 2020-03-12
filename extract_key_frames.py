import ffmpeg
import numpy as np
import tensorflow as tf

from transnet import TransNetParams, TransNet
from transnet_utils import draw_video_with_predictions, scenes_from_predictions
from timeit import default_timer as timer
from datetime import timedelta, datetime
from utils import create_dir, get_timecode, check_folder_exist, get_fps, save_frame_doc, insert_profile, get_db_connection

from helper import YamlReader
import glob, os, sys
import json
from os import makedirs, lstat
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

import cv2


def load_model():
    # initialize the network
    params = TransNetParams()
    params.CHECKPOINT_PATH = "./model/transnet_model-F16_L3_S2_D256"
    net = TransNet(params)
    return net, params

def predict(net, params, video_file):
    scenes = None
    # export video into numpy array using ffmpeg
    start = timer()
    video_stream, err = (
        ffmpeg
        .input(video_file)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(params.INPUT_WIDTH, params.INPUT_HEIGHT))
        .global_args('-loglevel', 'error')
        .global_args('-y')
        .run(capture_stdout=True)
    )
    video = np.frombuffer(video_stream, np.uint8).reshape([-1, params.INPUT_HEIGHT, params.INPUT_WIDTH, 3])
    print('Loading video took (hh:mm:ss.ms) {}'.format(timedelta(seconds=timer()-start)))
    # predict transitions using the neural network
    start = timer()
    predictions = net.predict_video(video)
    print('Prediction took (hh:mm:ss.ms) {}'.format(timedelta(seconds=timer()-start)))
    # Generate list of scenes from predictions, returns tuples of (start frame, end frame)
    scenes = scenes_from_predictions(predictions, threshold=0.1)

    return scenes
    
def print_progress(iteration, total, prefix='', suffix='', decimals=3, bar_length=100):
    """
    Call in a loop to create standard out progress bar
    :param iteration: current iteration
    :param total: total iterations
    :param prefix: prefix string
    :param suffix: suffix string
    :param decimals: positive number of decimals in percent complete
    :param bar_length: character length of bar
    :return: None
    """

    format_str = "{0:." + str(decimals) + "f}"  # format the % done number string
    percents = format_str.format(100 * (iteration / float(total)))  # calculate the % done
    filled_length = int(round(bar_length * iteration / float(total)))  # calculate the filled bar length
    bar = '#' * filled_length + '-' * (bar_length - filled_length)  # generate the bar string
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),  # write out the bar
    sys.stdout.flush()  # flush to stdout


def extract_frames(video_path, frames_dir, mapping, scenes, isTransitions, yaml, 
                   origin, overwrite=False, start=-1, end=-1, every=1):
    """
    Extract frames from a video using OpenCVs VideoCapture
    :param video_path: path of the video
    :param frames_dir: the directory to save the frames
    :param overwrite: to overwrite frames that already exist?
    :param start: start frame
    :param end: end frame
    :param every: frame spacing
    :return: count of images saved
    """

    video_path = os.path.normpath(video_path)  # make the paths OS (Windows) compatible
    frames_dir = os.path.normpath(frames_dir)  # make the paths OS (Windows) compatible

    video_dir, video_filename = os.path.split(video_path)  # get the video path and filename from the path

    assert os.path.exists(video_path)  # assert the video file exists

    capture = cv2.VideoCapture(video_path)  # open the video using OpenCV

    if start < 0:  # if start isn't specified lets assume 0
        start = 0
    if end < 0:  # if end isn't specified assume the end of the video
        end = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    capture.set(1, start)  # set the starting frame of the capture
    frame = start  # keep track of which frame we are up to, starting from start
    while_safety = 0  # a safety counter to ensure we don't enter an infinite while loop (hopefully we won't need it)
    saved_count = 0  # a count of how many frames we have saved
    docs = list()

    while frame < end:  # lets loop through the frames until the end

        _, image = capture.read()  # read an image from the capture

        if while_safety > 500:  # break the while if our safety maxs out at 500
            break

        # sometimes OpenCV reads None's during a video, in which case we want to just skip
        if image is None:  # if we get a bad return flag or the image we read is None, lets not save
            while_safety += 1  # add 1 to our while safety, since we skip before incrementing our frame variable
            #print("bad frame after {} frames".format(saved_count))
            continue  # skip
        isTransition = frame in isTransitions.keys() and isTransitions[frame]
        if frame % every == 0 or isTransition:  # if this is a frame we want to write out based on the 'every' argument
            while_safety = 0  # reset the safety count
            name = "img{:06d}.jpg".format(mapping[frame])
            save_path = os.path.join(frames_dir, name)  # create the save path
            writeStatus = cv2.imwrite(save_path, image)  # save the extracted image
            if writeStatus:
                saved_count += 1  # increment our counter by one
                image = cv2.imread(save_path)
                new_height, new_width = image.shape[:2]
                image_info = lstat(save_path)
                image_time = str(image_info.st_mtime)
                image_size = str(image_info.st_size)
                start = get_timecode(yaml.delta_time * mapping[frame])
                end_doc = get_timecode(yaml.delta_time * (mapping[frame] + 1))
                download_time = str(datetime.now().time().strftime("%H:%M:%S"))
                download_date = str(datetime.now().date().strftime('%Y-%m-%d'))
                doc = {
                             "name": name,
                             "position": mapping[frame],
                             "origin": origin,
                             "videoExtension": yaml.video_extensions,
                             "remote": "false",
                             "customerName": yaml.customer_name,
                             "customerId": origin,
                             "start": start,
                             "end": end_doc,
                             "timestamp": "",
                             "sceneNumber": scenes[frame],
                             "sceneTrueStart": "",
                             "sceneHardStart": isTransitions[frame],
                             "sceneLogoStart": "",
                             "sceneMotionStart": "",
                             "size": str(image_size),
                             "status": "unprocessed",
                             "technologies": ["NLP", "LD", "PI"],
                             "downloadTime": download_time,
                             "downloadDate": download_date,
                             "processingDate": "",
                             "processingTime": "", "time": image_time,
                             "metadata": {},
                             "height": new_height,
                             "jobs": [],
                             "width": new_width,
                             "dataset": "",
                             "datasource": "",
                             "orientation": "",
                             "personComposition": {"personFilters": "", "pose": ""},
                             "date": {"insertion": datetime.utcnow(), "pamaArchiving": datetime.utcnow()},
                             "style": {"outdoor": "", "color": ""},
                             "boundingBoxes": [],
                             "tag": [],
                            "exportStatus": {"validated": "false", "exported": "false"}
                             } 
                docs.append(doc)      

        frame += 1  # increment our frame count

    capture.release()  # after the while has finished close the capture

    return saved_count, docs  # and return the count of the images we saved
    
def infos_calculation(total, transitions, every):
    mapping = dict()
    scenes = dict()
    isTransitions = dict()
    isTransition = False
    scene_index = -1
    frame_index = 0
    for i in range(total):
        if i % every == 0 or i in transitions:
            mapping[i] = frame_index
            if i in transitions:
                isTransition = True
                scene_index += 1
            else:
                isTransition = False
            scenes[i] = scene_index
            isTransitions[i] = isTransition
            frame_index += 1
    return mapping, scenes, isTransitions
    
def video_to_frames(video_path, frames_dir, transitions, config, origin, overwrite=False, every=1, chunk_size=1000):
    """
    Extracts the frames from a video using multiprocessing
    :param video_path: path to the video
    :param frames_dir: directory to save the frames
    :param overwrite: overwrite frames if they exist?
    :param every: extract every this many frames
    :param chunk_size: how many frames to split into chunks (one chunk per cpu core process)
    :return: path to the directory where the frames were saved, or None if fails
    """

    video_path = os.path.normpath(video_path)  # make the paths OS (Windows) compatible
    frames_dir = os.path.normpath(frames_dir)  # make the paths OS (Windows) compatible

    video_dir, video_filename = os.path.split(video_path)  # get the video path and filename from the path

    # make directory to save frames, its a sub dir in the frames_dir with the video name
    #os.makedirs(os.path.join(frames_dir, video_filename), exist_ok=True)

    capture = cv2.VideoCapture(video_path)  # load the video
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))  # get its total frame count
    capture.release()  # release the capture straight away
    
    mapping, scenes, isTransitions = infos_calculation(total, transitions, every)
    
    print(len(mapping.keys()), len(scenes.keys()), len(isTransitions.keys()))

    if total < 1:  # if video has no frames, might be and opencv error
        print("Video has no frames. Check your OpenCV + ffmpeg installation, can't read videos!!!\n"
              "You may need to install OpenCV by source not pip")
        return None  # return None

    frame_chunks = [[i, i+chunk_size] for i in range(0, total, chunk_size)]  # split the frames into chunk lists
    frame_chunks[-1][-1] = min(frame_chunks[-1][-1], total-1)  # make sure last chunk has correct end frame

    prefix_str = "Extracting frames from {}".format(video_filename)  # a prefix string to be printed in progress bar
    
    docs = list()
    print("Start saving")
    start = timer()
    # execute across multiple cpu cores to speed up processing, get the count automatically
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count(), ) as executor:

        futures = [executor.submit(extract_frames, video_path, frames_dir, mapping, scenes, 
                   isTransitions, config, origin, overwrite, f[0], f[1], every)
                   for f in frame_chunks]  # submit the processes: extract_frames(...)

        for i, f in enumerate(as_completed(futures)):  # as each process completes
            saved, partial_doc= f.result()
            docs.extend(partial_doc)
            #print_progress(i, len(frame_chunks)-1, prefix=prefix_str, suffix='Complete')  # print it's progress
    print('saving video took (hh:mm:ss.ms) {}'.format(timedelta(seconds=timer()-start)))
    #save documents in json file
    with open('{}.json'.format(origin), 'w') as outfile:
                json.dump(docs, outfile, indent=2, sort_keys=True, default=str)

    return os.path.join(frames_dir, video_filename)  # when done return the directory containing the frames

def main(yaml):
    
    #origine = customer_id = nom de la vidéo

    #outgoing: /opt/PRODUCTION_DATA/FTP_data/incomming/customer_id/origin/UNDONE,DONE,ONGOING

    net, params = load_model()
    videos_files = glob.iglob(os.path.join(yaml.location_incoming , "*.{}".format(yaml.video_extensions)))
    frequency = yaml.delta_time
    for video_file in videos_files:
        path, video_name = os.path.split(video_file)
        video_folder_name = video_name.split('.')[0]
        # Setup destination folders
        dest_path, video_folder_name = create_dir(yaml.location_outgoing, video_folder_name)
        dest_path,_ = create_dir(dest_path, video_folder_name)
        final_dest_path,_ = create_dir(dest_path, "UNDONE")
        create_dir(dest_path, "ONGOING")
        create_dir(dest_path, "DONE")        
        # Get frame rate of the video
        fps = get_fps(video_file)
        step_size = int(fps * frequency)
        print(frequency, step_size)
        #Make scene prediction
        scenes = predict(net, params, video_file)
        #Extracts the frames from a video using multiprocessing
        if scenes is not None :
            transitions = set(scenes[:,0])
            video_to_frames(video_path=video_file, frames_dir=final_dest_path, transitions=transitions,
                             config=yaml, origin=video_folder_name, overwrite=True, every=step_size, chunk_size=1000)
    
if __name__ == "__main__":

    yaml = YamlReader()
    main(yaml)
