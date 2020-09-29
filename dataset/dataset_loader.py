# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Classes to load KITTI and Cityscapes data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import json
import os
import re
from absl import logging
import numpy as np
import scipy.misc
import imageio

CITYSCAPES_CROP_BOTTOM = True  # Crop bottom 25% to remove the car hood.
CITYSCAPES_CROP_PCT = 0.75
CITYSCAPES_SAMPLE_EVERY = 2  # Sample every 2 frames to match KITTI frame rate.
BIKE_SAMPLE_EVERY = 6  # 5fps, since the bike's motion is slower.

class KittiRaw(object):
  """Reads KITTI raw data files."""

  def __init__(self,
               dataset_dir,
               split,
               load_pose=False,
               img_height=128,
               img_width=416,
               seq_length=3):
    static_frames_file = 'dataset/kitti/static_frames.txt'
    test_scene_file = 'dataset/kitti/test_scenes_' + split + '.txt'
    with open(test_scene_file, 'r') as f:
      test_scenes = f.readlines()
    self.test_scenes = [t[:-1] for t in test_scenes]
    self.dataset_dir = dataset_dir
    self.img_height = img_height
    self.img_width = img_width
    self.seq_length = seq_length
    self.load_pose = load_pose
    self.cam_ids = ['02']
    self.date_list = [
        '2011_10_03'
    ]
    self.collect_static_frames(static_frames_file)
    self.collect_train_frames()

  def collect_static_frames(self, static_frames_file):
    with open(static_frames_file, 'r') as f:
      frames = f.readlines()
    self.static_frames = []
    for fr in frames:
      if fr == '\n':
        continue
      unused_date, drive, frame_id = fr.split(' ')
      fid = '%.10d' % (np.int(frame_id[:-1]))
      for cam_id in self.cam_ids:
        self.static_frames.append(drive + ' ' + cam_id + ' ' + fid) 
      logging.info('\n all static frames [array] : %s', self.static_frames)  

  def collect_train_frames(self):
    """Creates a list of training frames."""
    all_frames = []
    for date in self.date_list:
      date_dir = os.path.join(self.dataset_dir, date)
      drive_set = os.listdir(date_dir)
      for dr in drive_set:
        drive_dir = os.path.join(date_dir, dr)
        logging.info('drive _dir : %s', drive_dir)
        if os.path.isdir(drive_dir):
          if dr[:-5] in self.test_scenes:
            continue
          for cam in self.cam_ids:
            img_dir = os.path.join(drive_dir, 'image_' + cam, 'data')
            num_frames = len(glob.glob(img_dir + '/*[0-9].png'))
            for i in range(num_frames):
              frame_id = '%.10d' % i
              all_frames.append(dr + ' ' + cam + ' ' + frame_id)
            logging.info('\n all train frames [array] : %s', all_frames)
    
    for s in self.static_frames:
      try:
        all_frames.remove(s)
      except ValueError:
        pass

    self.train_frames = all_frames
    self.num_train = len(self.train_frames)

  def is_valid_sample(self, frames, target_index):
    """Checks whether we can find a valid sequence around this frame."""
    num_frames = len(frames)
    target_drive, cam_id, _ = frames[target_index].split(' ')
    start_index, end_index = get_seq_start_end(target_index, self.seq_length)
    if start_index < 0 or end_index >= num_frames:
      return False
    start_drive, start_cam_id, _ = frames[start_index].split(' ')
    end_drive, end_cam_id, _ = frames[end_index].split(' ')
    if (target_drive == start_drive and target_drive == end_drive and
        cam_id == start_cam_id and cam_id == end_cam_id):
      return True
    return False

  def get_example_with_index(self, target_index):
    if not self.is_valid_sample(self.train_frames, target_index):
      return False
    example = self.load_example(self.train_frames, target_index)
    return example

  def load_image_sequence(self, frames, target_index):
    """Returns a sequence with requested target frame."""
    start_index, end_index = get_seq_start_end(target_index, self.seq_length)
    image_seq = []
    for index in range(start_index, end_index + 1):
      drive, cam_id, frame_id = frames[index].split(' ')
      img = self.load_image_raw(drive, cam_id, frame_id)
      if index == target_index:
        zoom_y = self.img_height / img.shape[0]
        zoom_x = self.img_width / img.shape[1]
      img = scipy.misc.imresize(img, (self.img_height, self.img_width))
      image_seq.append(img)
    return image_seq, zoom_x, zoom_y

  def load_pose_sequence(self, frames, target_index):
    """Returns a sequence of pose vectors for frames around the target frame."""
    target_drive, _, target_frame_id = frames[target_index].split(' ')
    target_pose = self.load_pose_raw(target_drive, target_frame_id)
    start_index, end_index = get_seq_start_end(target_frame_id, self.seq_length)
    pose_seq = []
    for index in range(start_index, end_index + 1):
      if index == target_frame_id:
        continue
      drive, _, frame_id = frames[index].split(' ')
      pose = self.load_pose_raw(drive, frame_id)
      # From target to index.
      pose = np.dot(np.linalg.inv(pose), target_pose)
      pose_seq.append(pose)
    return pose_seq

  def load_example(self, frames, target_index):
    """Returns a sequence with requested target frame."""
    image_seq, zoom_x, zoom_y = self.load_image_sequence(frames, target_index)
    target_drive, target_cam_id, target_frame_id = (
        frames[target_index].split(' '))
    intrinsics = self.load_intrinsics_raw(target_drive, target_cam_id)
    intrinsics = self.scale_intrinsics(intrinsics, zoom_x, zoom_y)
    example = {}
    example['intrinsics'] = intrinsics
    example['image_seq'] = image_seq
    example['folder_name'] = target_drive + '_' + target_cam_id + '/'
    example['file_name'] = target_frame_id
    if self.load_pose:
      pose_seq = self.load_pose_sequence(frames, target_index)
      example['pose_seq'] = pose_seq
    return example

  def load_pose_raw(self, drive, frame_id):
    date = drive[:10]
    pose_file = os.path.join(self.dataset_dir, date, drive, 'poses',
                             frame_id + '.txt')
    with open(pose_file, 'r') as f:
      pose = f.readline()
    pose = np.array(pose.split(' ')).astype(np.float32).reshape(3, 4)
    pose = np.vstack((pose, np.array([0, 0, 0, 1]).reshape((1, 4))))
    return pose

  def load_image_raw(self, drive, cam_id, frame_id):
    date = drive[:10]
    img_file = os.path.join(self.dataset_dir, date, drive, 'image_' + cam_id,
                            'data', frame_id + '.png')
    img = imageio.imread(img_file)
    return img

  def load_intrinsics_raw(self, drive, cam_id):
    date = drive[:10]
    calib_file = os.path.join(self.dataset_dir, date, 'calib_cam_to_cam.txt')
    filedata = self.read_raw_calib_file(calib_file)
    p_rect = np.reshape(filedata['P_rect_' + cam_id], (3, 4))
    intrinsics = p_rect[:3, :3]
    return intrinsics

  # From https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
  def read_raw_calib_file(self, filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}
    with open(filepath, 'r') as f:
      for line in f:
        key, value = line.split(':', 1)
        # The only non-float values in these files are dates, which we don't
        # care about.
        try:
          data[key] = np.array([float(x) for x in value.split()])
        except ValueError:
          pass
    return data

  def scale_intrinsics(self, mat, sx, sy):
    out = np.copy(mat)
    out[0, 0] *= sx
    out[0, 2] *= sx
    out[1, 1] *= sy
    out[1, 2] *= sy
    return out


"""def get_resource_path(relative_path):
  print("\n the relative path is = %s ", relative_path)
  return relative_path
"""

def get_seq_start_end(target_index, seq_length, sample_every=1):
  """Returns absolute seq start and end indices for a given target frame."""
  half_offset = int((seq_length - 1) / 2) * sample_every
  end_index = target_index + half_offset
  start_index = end_index - (seq_length - 1) * sample_every
  return start_index, end_index


def atoi(text):
  return int(text) if text.isdigit() else text


def natural_keys(text):
  return [atoi(c) for c in re.split(r'(\d+)', text)]
