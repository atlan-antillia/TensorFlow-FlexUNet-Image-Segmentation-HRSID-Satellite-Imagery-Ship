# Copyright 2025 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ImageMaskDatasetGeneraotor.py
# 2025/12/27

import os
import numpy as np
import json
import traceback
import shutil
import cv2
import traceback


class ImageMaskDatasetGeneraotor:
  def __init__(self, down_scale=True):
    self.down_scale = down_scale
    self.RESIZE = (512, 512)
    self.RGB_COLORS ={0:(0,0,0), 1:(0,255,0),2:(0,0,255),3:(255,0,255),4:(255,255,0),5:(0,255,255),6:(128,128,128)}
    pass

  def generate(self, json_file, images_dir, output_dir, output_images_dir):
    with open(json_file) as f:
      data = json.load(f)
      images = data["images"]
      annotations = data["annotations"]

      for image in images:
        #input("----")
        filename = image["file_name"]
        image_filepath = os.path.join(images_dir, filename)
        img = cv2.imread(image_filepath)
        if self.down_scale:
          img = cv2.resize(img, self.RESIZE)
        output_imagefilepath = os.path.join(output_images_dir, filename)
        cv2.imwrite(output_imagefilepath, img)
        print("---Saved {}".format(image_filepath))
        filename = filename.replace(".jpg", ".png")
        id = image["id"]
        h  = image["height"]
        w  = image["width"]
        mask = np.zeros((h, w, 3), dtype=np.uint8)
        bgr_color =(0,255,0)

        for annotation in annotations:
          image_id    = annotation["image_id"]
          if id == image_id:
            category_id = annotation["category_id"]
            rgb_color = (0,0,0)
            try:
              rgb_color = self.RGB_COLORS[category_id]
            except:
              traceback.print_exc()
              input("Input any key")

            (r, g, b) = rgb_color
            bgr_color = (b, g, r)            
            #print("categoyru_id {} color {}".format(category_id, rgb_color))
            segmentations = annotation["segmentation"]
            for segmentation in segmentations:
              l = len(segmentation)
              i = 0
              pt = []
              while i<l-1:
                x = int(segmentation[i])
                y = int(segmentation[i+1])
                pt.append ([x,y])
                i += 2 

              pts = np.array(pt)

            cv2.fillConvexPoly(mask, points =pts, color=(bgr_color))
            print("fill color {}".format(bgr_color))
        if self.down_scale:
          mask = cv2.resize(mask, self.RESIZE)
        output_mask_filepath = os.path.join(output_dir, filename)
        cv2.imwrite(output_mask_filepath, mask)
        print("---Saved {}".format(output_mask_filepath))



  
if __name__ == "__main__":
  try:
    json_file  = "./annotations/train2017.json"
    images_dir = "./JPEGImages/"
    output_dir = "./HRSID_train_master/"
    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    output_images_dir = "./HRSID_train_master/images/"
    output_masks_dir  = "./HRSID_train_master/masks/"
    os.makedirs(output_masks_dir)
    os.makedirs(output_images_dir)
 
    generator = ImageMaskDatasetGeneraotor(down_scale=False)

    generator.generate(json_file, images_dir, output_masks_dir, output_images_dir)
  except:
    traceback.print_exc()
