from copy import deepcopy

import cv2
import numpy as np
import os
import json
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import patches
from typing import Optional, Dict, List,Tuple
from pytesseract import image_to_data, image_to_string

import pytesseract

import logging

from client_code.Structure_Doc_Api.src.main.logs import setup_logging
logger = setup_logging(log_filename='app', log_level=logging.DEBUG, logs_dir='client_code/Structure_Doc_Api/logs', overwrite_existing=True)

def terminate():
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def draw_bounding_box(image: object, labels_list: list, saved_to:str=None):
    i= 0
    for item in labels_list:
        x1, y1, x2, y2= item
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if saved_to is not None:
        cv2.imwrite(os.path.join(saved_to, "main_printed_text_seg"+".jpg"),image)
    else:
        cv2.imshow("image",image)
        terminate()
        pass

def line_detection(ROI: List[int], image_path:str) -> List[List[int]]:
    '''
    Performs line detection on a region of interest (ROI) within an image.

    Parameters:
    - ROI (list): A list containing the coordinates [x1, y1, x2, y2] of the region of interest.
    - image_path (str): The path to the input image.

    Returns:
    - list: A list containing coordinates [x1, y1, x2, y2] of the detected lines within the ROI.
    '''

     # Parameter Type Checking
    assert isinstance(ROI, list) and len(ROI) == 4, "ROI must be a list of 4 integers."
    assert all(isinstance(coord, int) for coord in ROI), "All elements of ROI must be integers."
    assert isinstance(image_path, str), "image_path must be a string."

    assert os.path.isfile(image_path), f"Image file not found at path: {image_path}"
    image = cv2.imread(image_path)
    assert image is not None, f"Failed to read image at path: {image_path}"
    x1, y1, x2, y2= ROI
    image= image[y1:y2, x1:x2]
    result = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Detect horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
    detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # i=0
    line_seg_dict = {}
    line_seg_id = 0
    detected_lines = []
    for i, c in enumerate(cnts):
        x, y, w, h = cv2.boundingRect(c)
        if 100< w * h <1000:
            detected_lines.append([x, y, x + w, y + h])
            cv2.drawContours(result, [c], -1, (36, 255, 12), 2)
    # cv2.imshow('image', result)
    # terminate()
    return detected_lines


def line_detection_for_text_seg_block(ROI: List[int], image_path:str) -> List[List[int]]: 
    '''
    Performs line detection specifically for text segmentation within a region of interest (ROI) in an image.

    Parameters:
    - ROI (list): A list containing the coordinates [x1, y1, x2, y2] of the region of interest.
    - image_path (str): The path to the input image.

    Returns:
    - list: A list containing coordinates [x1, y1, x2, y2] of the detected lines within the text segmentation block.
    '''
    assert os.path.isfile(image_path), f"Image file not found at path: {image_path}"
    image = cv2.imread(image_path)
    assert image is not None, f"Failed to read image at path: {image_path}"

    x1, y1, x2, y2= ROI
    image= image[y1:y2, x1:x2]
    result = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Detect horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
    detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    line_seg_dict = {}
    line_seg_id = 0
    detected_lines = []
    for i, c in enumerate(cnts):
        x, y, w, h = cv2.boundingRect(c)
        if 100< w * h <1500:
            detected_lines.append([x, y, x + w, y + h])
            cv2.drawContours(result, [c], -1, (36, 255, 12), 2)
    # cv2.imshow('image', result)
    # terminate()
    return detected_lines


    
def line_segmentation(image_path: str, box_saved_path:Optional[str] = None, block_coords:Optional[List[int]]= None, closest_box: Optional[List[List[int]]]=None) -> Dict[str, List[int]]:
    '''
    Performs line segmentation within a given image.

    Parameters:
    - image_path (str): The path to the input image.
    - box_saved_path (str): The path to save the line segment images (default is None).
    - block_coords (list): The coordinates [x1, y1, x2, y2] of the block in the image (default is None).
    - closest_box (list): The coordinates [x1, y1, x2, y2] of the closest box (default is None).

    Returns:
    - dict: A dictionary containing line segment IDs as keys and their corresponding coordinates [x1, y1, x2, y2].
    '''
    # Parameter Type Checking
    assert isinstance(image_path, str), "image_path must be a string."
    assert box_saved_path is None or isinstance(box_saved_path, str), "box_saved_path must be a string or None."
    assert block_coords is None or isinstance(block_coords, list) and len(block_coords) == 4, "block_coords must be a list of 4 integers or None."
    assert closest_box is None or (isinstance(closest_box, list) and
                                   all(isinstance(coord, list) and len(coord) == 4 for coord in closest_box)), "closest_box must be a list of lists with each inner list containing 4 integers or None."

    assert os.path.isfile(image_path), f"Image file not found at path: {image_path}"
    image = cv2.imread(image_path)
    assert image is not None, f"Failed to read image at path: {image_path}"
    result = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Detect horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # i=0
    line_seg_dict = {}
    line_seg_id = 0
    current_line = []
    logger.info(f"block coords in line segmenation: {block_coords}")
    block_width= block_coords[2]- block_coords[0]
    if any(closest_box):
        ''' Block has main printed text '''
        for i, c in enumerate(cnts):
            x, y, w, h = cv2.boundingRect(c)
            if w * h > 2000 and block_width-5<= w<= block_width+5:
                current_line.append([x, y, x + w, y + h])
                if len(current_line) == 2:
                    line_seg_id += 1
                    line_seg_dict.update({"line_sig" + str(line_seg_id): [val for val in current_line]})
                    last_ele = current_line[1]
                    current_line.clear()
                    current_line.append(last_ele)
                    cv2.drawContours(result, [c], -1, (36, 255, 12), 2)
    else:
        '''block don't have main printed text'''
        logger.info("block don't have main printed text")
        for i, c in enumerate(cnts):
            x, y, w, h = cv2.boundingRect(c)
            if w * h > 1000 and block_width-5<= w<= block_width+5:                                           #and block_width-5<= w<= block_width+5
                current_line.append([x, y, x + w, y + h])
                if len(current_line) == 2:
                    line_seg_id += 1
                    line_seg_dict.update({"line_sig" + str(line_seg_id): [val for val in current_line]})
                    last_ele = current_line[1]
                    current_line.clear()
                    current_line.append(last_ele)
                    cv2.drawContours(result, [c], -1, (36, 255, 12), 2)

    logger.info(f"final line segmentation: {line_seg_dict}")
    assert line_seg_dict, "No line segments detected."
    line_seg_coords_dict= {}
    for line_seg_key in list(line_seg_dict.keys()):
        line_seg = line_seg_dict[line_seg_key]
        line1, line2 = line_seg[0], line_seg[1]
        # Find intersection point
        intersection_x = (line1[2] + line2[0]) // 2
        intersection_y = (line1[3] + line2[1]) // 2
        # Calculate the bounding box coordinates
        x_min = min(line1[0], line1[2], line2[0], line2[2])
        y_min = min(line1[1], line1[3], line2[1], line2[3])
        x_max = max(line1[0], line1[2], line2[0], line2[2])
        y_max = max(line1[1], line1[3], line2[1], line2[3])
        line_seg_coords_dict[line_seg_key]= [x_min, y_min, x_max, y_max]

    assert line_seg_coords_dict, "No line segment coordinates found."

    line_seg_path= os.path.join(box_saved_path, "line_seg")
    os.makedirs(line_seg_path, exist_ok= True)

    for line_seg_cords_key in list(line_seg_coords_dict.keys()):
        x1, y1, x2, y2 = line_seg_coords_dict[line_seg_cords_key]
        seg_roi= image[y1:y2, x1:x2]
        cv2.imwrite(os.path.join(line_seg_path, line_seg_cords_key+'.jpg'), seg_roi)
    return line_seg_coords_dict

def merge_boxes(boxes):
    if len(boxes) == 0:
        return []

    x_min = min(box[0] for box in boxes)
    y_min = min(box[1] for box in boxes)
    x_max = max(box[2] for box in boxes)
    y_max = max(box[3] for box in boxes)

    return [x_min, y_min, x_max, y_max]


def detecting_text_seg(word_coordinates:List[Dict[str, any]], box_value_lst:Optional[List[List[int]]]= None) -> List[List[int]]:
    '''
    Detects and segments text based on word coordinates, excluding words within specified boxes.

    Parameters:
    - word_coordinates (List[Dict[str, any]]): A list of dictionaries containing word coordinates.
    - box_value_lst (List[List[int]]): A list of box coordinates to exclude words (default is None).

    Returns:
    - List[List[int]]: A list containing coordinates [x1, y1, x2, y2] of segmented text lines.
    '''
     # Parameter Type Checking
    assert isinstance(word_coordinates, list), "word_coordinates must be a list."
    assert all(isinstance(word, dict) and 'coordinates' in word for word in word_coordinates), "Each element in word_coordinates must be a dictionary with 'coordinates'."
    assert box_value_lst is None or (isinstance(box_value_lst, list) and all(isinstance(coord, list) and len(coord) == 4 for coord in box_value_lst)), "box_value_lst must be a list of lists with each inner list containing 4 integers or None."

    reference_top=0
    line_seg= []
    text_seg=[]
    for word in word_coordinates:
        text_cords= word['coordinates']
        count=0
        if box_value_lst is not None:
            for box in box_value_lst:
                "Eliminating words from blocks"
                if get_intersection_percentage(box,text_cords)>0.5:
                    count+=1
        if count==0:
            #Assigened reference coordinate
            if reference_top==0:
                print('******************************* Reference coordinates ************************')
                reference_top= word['top']
                reference_cords= word['coordinates']
                line_seg.append(word['coordinates'])
            elif reference_top-5<= word['top']<= reference_top+5 and abs(reference_cords[2]- word['coordinates'][0])<=100:
                print("********************** distance between words ***********************************")
                print(reference_cords)
                print(word)
                print(abs(reference_cords[2]- word['coordinates'][0]))
                line_seg.append(word['coordinates'])
                reference_cords= word['coordinates']
            else:
                print(line_seg)
                text_seg.append(merge_boxes(line_seg))
                reference_top= word['top']
                reference_cords= word['coordinates']
                line_seg.clear()
                line_seg.append(word['coordinates'])
    return text_seg



def detecting_main_printed_text(box_coords: list, img: object, saved_to: str, word_coordinates:list):
    '''
    Detects and extracts main printed text in the image.

    Parameters:
    - box_coords (list): List of bounding boxes in the image.
    - img (object): The input image object.
    - saved_to (str): The path to save the main printed text images.
    - word_coordinates (list): List of word coordinates.

    Returns:
    - list: List of coordinates of the main printed text.
    '''
     # Parameter Type Checking
    assert isinstance(box_coords, list), "box_coords must be a list."
    assert all(isinstance(box_dict, dict) and len(box_dict) == 1 for box_dict in box_coords), "Each element in box_coords must be a dictionary with a single key."
    assert isinstance(img, np.ndarray), "img must be a NumPy array."
    assert isinstance(saved_to, str), "saved_to must be a string."
    assert isinstance(word_coordinates, list), "word_coordinates must be a list."

    main_printed_text_group= {}
    box_value_lst= [list(box_dict.values())[0] for box_dict in box_coords]
    print(box_value_lst)
    for group_id,box_dict in enumerate(box_coords):
        box_val= list(box_dict.values())[0]
        for _, box in enumerate(box_value_lst):
            if box!=box_val and box[1]==box_val[1]:
                # more than one box in parallel
                if group_id not in list(main_printed_text_group.keys()):
                    main_printed_text_group[group_id]=[]
                    main_printed_text_group[group_id].append(box)
                else:
                    main_printed_text_group[group_id].append(box)

            else:
                main_printed_text_group[group_id]=[]
                main_printed_text_group[group_id].append(box_val)
    
    print('*'*20, "main printed text",'*'*20)
    print(main_printed_text_group)
    text_seg= detecting_text_seg(word_coordinates, box_value_lst)

    draw_bounding_box(img, text_seg, saved_to)
    return text_seg





def get_iou_new(bb1, bb2):
	"""
	Calculate the Intersection over Union (IoU) of two bounding boxes.

	Parameters
	----------
	bb1 : dict
		Keys: {0, '2', 1, '3'}
		The (x1, 1) position is at the top left corner,
		the (2, 3) position is at the bottom right corner
	bb2 : dict
		Keys: {0, '2', 1, '3'}
		The (x, y) position is at the top left corner,
		the (2, 3) position is at the bottom right corner

	Returns
	-------
	float
		in [0, 1]
	"""
	assert bb1[0] < bb1[2]
	assert bb1[1] < bb1[3]
	assert bb2[0] < bb2[2]
	assert bb2[1] < bb2[3]

	# determine the coordinates of the intersection rectangle
	x_left = max(bb1[0], bb2[0])
	y_top = max(bb1[1], bb2[1])
	x_right = min(bb1[2], bb2[2])
	y_bottom = min(bb1[3], bb2[3])

	if x_right < x_left or y_bottom < y_top:
		return 0.0

	# The intersection of two axis-aligned bounding boxes is always an
	# axis-aligned bounding box
	intersection_area = (x_right - x_left) * (y_bottom - y_top)

	# compute the area of both AABBs
	bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
	bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
	assert iou >= 0.0
	assert iou <= 1.0
	return iou


def get_intersection_percentage(bb1, bb2):
    # Calculate the percentage of vertical intersection between two bounding boxes
    x1_bb1, y1_bb1, x2_bb1, y2_bb1 = bb1
    x1_bb2, y1_bb2, x2_bb2, y2_bb2 = bb2

    x_left = max(x1_bb1, x1_bb2)
    y_top = max(y1_bb1, y1_bb2)
    x_right = min(x2_bb1, x2_bb2)
    y_bottom = min(y2_bb1, y2_bb2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = max(0, y_bottom - y_top)
    bb1_area = y2_bb1 - y1_bb1
    bb2_area = y2_bb2 - y1_bb2

    return intersection_area / min(bb1_area, bb2_area)




def draw_coords(image: object, coords_dict: list):
    print('Entere into draw coords +++++++++++++++++++++++++++++')
    # coords_keys= [coords_key.keys() for coords_key in coords_dict]
    for dict_item in coords_dict:
        print(f'keys traversed: {list(dict_item.keys())[0]}')
        if list(dict_item.keys())[0] == "text_8":
            coords = list(dict_item.values())[0]
            # print(coords)
            # exit('++++++++++++++')
            cv2.rectangle(image, (coords[0], coords[1]), (coords[2], coords[3]), (0, 255, 0), 2)
    cv2.imshow(f'image', image)
    terminate()


def draw_roi(thresh, box: dict):
    for value in box.values():
        x1, y1, x2, y2 = value[0], value[1], value[2], value[3]
        print(x1, y1, x2, y2)
        # exit('++++++++++++++')
        ROI = thresh[y1:y2, x1:x2]
        data = pytesseract.image_to_string(ROI, lang='eng', config='--psm 3')
        print(data)
        # cv2.imshow('thresh', thresh)
        # cv2.imshow('ROI', ROI)
        cv2.imwrite('data/output/roi.jpg', ROI)


def calculate_intersection(box1, box2):
    '''
    Calculate the percentage of vertical intersection between two bounding boxes.

    Parameters:
    - bb1 (list): Coordinates [x1, y1, x2, y2] of the first bounding box.
    - bb2 (list): Coordinates [x1, y1, x2, y2] of the second bounding box.

    Returns:
    - float: Percentage of vertical intersection between the two bounding boxes.
    '''
    # Calculate the intersection area
    x_overlap = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0]))
    y_overlap = max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
    intersection_area = x_overlap * y_overlap

    # Calculate the area of each bounding box
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate the intersection percentage
    intersection_percentage = intersection_area / min(area_box1, area_box2)

    return intersection_percentage


def calculate_area(box):
    return (box[2] - box[0]) * (box[3] - box[1])

def merge_block_boxes(bounding_boxes):
    '''
    Merges overlapping bounding boxes and returns the final list of non-overlapping bounding boxes.

    Parameters:
    - bounding_boxes (List[Dict[str, int]]): List of bounding boxes, where each bounding box is represented as a dictionary.

    Returns:
    - Tuple[List[Dict[str, int]], List[int]]: A tuple containing:
        - List[Dict[str, int]]: Final non-overlapping bounding boxes.
        - List[int]: List of indices to remove from the original list of bounding boxes.
    '''
    # Parameter Type Checking
    assert isinstance(bounding_boxes, list), "bounding_boxes must be a list."
    # assert all(isinstance(box, dict) and set(box.keys()) == {'box_'+str(i) for i in range(4)} for box in bounding_boxes), "Each element in bounding_boxes must be a dictionary with keys: 'box_0', 'box_1', 'box_2', 'box_3'."
    print(f'entered into merge boxes +++++++++++++++++++++++++++++')
    remove_index_lst= []
    final_boxes = []
    for i in range(len(bounding_boxes)):
      overlap_box= []
      for j in range(i+1, len(bounding_boxes)):

          box1 = bounding_boxes[i]
          box2 = bounding_boxes[j]

          box1_coords= list(box1.values())[0]
          box2_coords= list(box2.values())[0]
          print(box1_coords)

          intersection_percentage = calculate_intersection(box1_coords, box2_coords)
          print(f"intersection percentage: {intersection_percentage}")
          # If there is an overlap, choose the box with the larger area
          if intersection_percentage > 0.4:
              if calculate_area(box1_coords) >= calculate_area(box2_coords):
                  overlap_box.append(box1)
                  remove_index_lst.append(j)
              else:
                  overlap_box.append(box2)
                  remove_index_lst.append(i)
          # else:
          #     # If no overlap, append both boxes to the final list
          #     overlap_box.append(box1)
      print("overlapping boxes +++++++++++++++++")
      print(overlap_box)
      if len(overlap_box)!=0:
        [final_boxes.append(val) for val in overlap_box]
    print(f'index to remove at end: ++++++++++++++++++++++=')
    print(remove_index_lst)
    for index in remove_index_lst:
      del bounding_boxes[index]
    return bounding_boxes, remove_index_lst



def __crop_image__(img: object, block_coords_lst: list, save_to):
    pil_image = Image.fromarray(np.uint8(img))
    for block in block_coords_lst:
        for block_id, block_cords in block.items():
            x1, y1, x2, y2 = block_cords
            img2 = pil_image.crop((x1, y1, x2, y2))
            # saving in outputs/crop/each_block
            os.makedirs(os.path.join(save_to, block_id),exist_ok=True)
            img2.save(os.path.join(save_to,block_id, block_id+'.jpg'))





def block_detection(img_path, file_name)-> Tuple[List[Dict[str, int]], List[List[int]], str]:
    '''
    Detects and extracts blocks from an image, saving the cropped blocks and the overall block image.

    Parameters:
    - img_path (str): The path to the input image.
    - file_name (str): The name to use for saving the cropped blocks and the overall block image.

    Returns:
    - Tuple[List[Dict[str, int]], List[List[int]], str]: A tuple containing:
        - List[Dict[str, int]]: Coordinates of individual blocks.
        - List[List[int]]: Hierarchy information for each block.
        - str: The path where the cropped blocks and the overall block image are saved.
    '''

     # Parameter Type Checking
    assert isinstance(img_path, str), "img_path must be a string."
    assert isinstance(file_name, str), "file_name must be a string."

    assert os.path.isfile(img_path), f"Image file not found at path: {img_path}"
    img = cv2.imread(img_path)
    assert img is not None, f"Failed to read image at path: {img_path}"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # Blur the image
    blur = cv2.GaussianBlur(thresh_inv, (1, 1), 0)

    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # find contours
    contours, hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    print(hierachy)
    hierarchy = hierachy[0]

    mask = np.ones(img.shape[:2], dtype="uint8") * 255
    i = 0
    box_coords = []
    hirerachy= []
    parent_block=  []
    parent_hirerachy= []
    for compt in zip(contours, hierarchy):
        c= compt[0]
        hirechy= compt[1]
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(c)
        if (w * h > 1500 and w>1000 and h>120):
          if hirechy[1]<0 and hirechy[3]< 0:
            parent_block.append({"box_" + str(i): [x, y, x + w, y + h]})
            print('hirechy in parent block')
            print(hirechy)
            parent_hirerachy.append(hirechy)
            i+=1
          else:
            if hirechy[3]< 0 or not any(element < 0 for element in hirechy):
                hirerachy.append(hirechy)
                print('hirechy in child block')
                print(hirechy)
                box_coords.append({"box_" + str(i): [x, y, x + w, y + h]})
                cv2.rectangle(mask, (x, y), (x + w, y + h), (0, 0, 255), -1)
                i += 1
    print(f'initial block coords: {box_coords}')
    print(f'parent block ++++++++++++++++++')
    print(parent_block)
    if len(box_coords)==0:
      box_coords= parent_block
      hirerachy= parent_hirerachy

    # remove ovelapping block boxes

    box_coords, remove_index_lst= merge_block_boxes(box_coords)

    for index in remove_index_lst:
      del hirerachy[index]

    res_final = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask))

    print(box_coords)
    save_to = os.path.join("data/output", 'crop', file_name)
    os.makedirs(save_to, exist_ok=True)
    __crop_image__(img, box_coords, save_to)
    cv2.imwrite(os.path.join(save_to,"block"+".jpg"), res_final)
    return box_coords, hirerachy,save_to


def block_detection_on_test_image(img: object, file_name: str, image_saving_path:str):
    '''
    Detects and extracts blocks in a test image.

    Parameters:
    - img (object): The input image object.
    - file_name (str): The name of the image file.
    - image_saving_path (str): The path to save the block images.

    Returns:
    - Tuple[List[Dict[str, int]], List[int], str]: A tuple containing:
        - List[Dict[str, int]]: Final non-overlapping bounding boxes.
        - List[int]: List of indices to remove from the original list of bounding boxes.
        - str: Path where block images are saved.
    '''
    # Parameter Type Checking
    assert isinstance(img, np.ndarray), "img must be a NumPy array."
    assert isinstance(file_name, str), "file_name must be a string."
    assert isinstance(image_saving_path, str), "image_saving_path must be a string."

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # Blur the image
    blur = cv2.GaussianBlur(thresh_inv, (1, 1), 0)

    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # find contours
    contours, hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    hierarchy = hierachy[0]

    mask = np.ones(img.shape[:2], dtype="uint8") * 255
    i = 0
    box_coords = []
    hirerachy= []
    parent_block=  []
    parent_hirerachy= []
    for compt in zip(contours, hierarchy):
        c= compt[0]
        hirechy= compt[1]
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(c)
        if (w * h > 1500 and w>1500 and h>120):
          if hirechy[1]<0 and hirechy[3]< 0:
            parent_block.append({"box_" + str(i): [x, y, x + w, y + h]})
            parent_hirerachy.append(hirechy)
            i+=1
          else:
            if hirechy[3]< 0 or not any(element < 0 for element in hirechy):
                hirerachy.append(hirechy)
                box_coords.append({"box_" + str(i): [x, y, x + w, y + h]})
                cv2.rectangle(mask, (x, y), (x + w, y + h), (0, 0, 255), -1)
                i += 1
    logger.info(f'initial block coords: {box_coords}')
    logger.info(f'parent block ++++++++++++++++++')
    print(parent_block)
    if len(box_coords)==0:
      box_coords= parent_block
      hirerachy= parent_hirerachy

    # remove ovelapping block boxes

    box_coords, remove_index_lst= merge_block_boxes(box_coords)

    for index in remove_index_lst:
      del hirerachy[index]

    res_final = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask))

    logger.info(f"final block coords: {box_coords}")
    os.makedirs(image_saving_path, exist_ok=True)
    __crop_image__(img, box_coords, image_saving_path)
    cv2.imwrite(os.path.join(image_saving_path,"block"+".jpg"), res_final)
    return box_coords, hirerachy,image_saving_path



def  text_blank_Field_detection(img_path:str, box_folder_path:str= None):
    '''
    Detects blank spaces and text fields in a block.

    Parameters:
    - img_path (str): The path to the input image.
    - box_folder_path (str): The path to save the visualized image (default is None).

    Returns:
    - tuple: A tuple containing two lists - blank_space_coords and text_coords.
    '''
    # Parameter Type Checking
    assert isinstance(img_path, str), "img_path must be a string."
    assert box_folder_path is None or isinstance(box_folder_path, str), "box_folder_path must be a string or None."

    img= cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,230,255,0)
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    ROIs=[]
    coords=[]
    for cnt in contours:
        x1,y1 = cnt[0][0]
        approx = cv2.approxPolyDP(cnt, 0.005*cv2.arcLength(cnt, True), True)
        x, y, w, h = cv2.boundingRect(cnt)
        if len(approx) == 4:
            if w>50 and h>30:
                ROI = img[y:y+h, x:x+w]
                ROIs.append(ROI)
                coords.append([x,y, x+w, y+h])
                ratio = float(w)/h
                if ratio >= 0.9 and ratio <= 1.1:
                    img = cv2.drawContours(img, [cnt], -1, (0,255,255), 3)
                else:
                    img = cv2.drawContours(img, [cnt], -1, (0,255,0), 3)
        elif w>50 and h>30:
          """This is exceptional case where the text inside the rectangle/square box are overlapping in this case 
          will more than 4 sides and make sure it is not considered text as rectangle"""
          print(f'value of approx in this situation : {len(approx)}')
          x, y, w, h = cv2.boundingRect(cnt)
          ROI = img[y:y+h, x:x+w]
          ROIs.append(ROI)
          coords.append([x,y, x+w, y+h])
          if ratio >= 0.9 and ratio <= 1.1:
                    img = cv2.drawContours(img, [cnt], -1, (0,255,255), 3)
          else:
                    img = cv2.drawContours(img, [cnt], -1, (0,255,0), 3)
    blank_space_coords=[]
    text_coords=[]
    count=0
    text_id=0
    for id, i in enumerate(ROIs):
        roi = i
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray,230,255,cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        if len(contours) > 1:
            text_coords.append({'text_'+str(text_id): coords[id]})
            text_id+=1
            x1, y1,x2,y2= coords[id]
            cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)

            print('DO OCR HERE')
        else:
            print(f'len of contour: {len(contours)}')
            blank_space_coords.append({"blank_"+str(count):coords[id]})
            x1, y1,x2,y2= coords[id]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
            print('BLANK SPACE')
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 2)
            count+=1
    print(f'blank space count: {count}')
    print(blank_space_coords)
    print(f'text block count: {len(text_coords)}')
    print(text_coords)
    blank_text_path= os.path.join(box_folder_path, "blank_space_text")
    os.makedirs(blank_text_path, exist_ok=True)
    cv2.imwrite(os.path.join(blank_text_path, "blank_text_vis.jpg"),img)
    return blank_space_coords, text_coords


def  text_blank_Field_detection_on_test_image(img_path:str, box_folder_path:str= None):
    '''
    Perform text and blank field detection on a test image.

    Parameters:
    - img_path (str): The path to the input image.
    - box_folder_path (str): The path to the folder where the output images will be saved (default is None).

    Returns:
    - tuple: A tuple containing two lists - blank_space_coords and text_coords.
    '''
    # Parameter Type Checking
    assert isinstance(img_path, str), "img_path must be a string."
    assert box_folder_path is None or isinstance(box_folder_path, str), "box_folder_path must be a string or None."

    img= cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,230,255,0)
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    ROIs=[]
    coords=[]
    for cnt in contours:
        x1,y1 = cnt[0][0]
        approx = cv2.approxPolyDP(cnt, 0.005*cv2.arcLength(cnt, True), True)
        x, y, w, h = cv2.boundingRect(cnt)
        if len(approx) == 4:
            if w>50 and h>30:
                ROI = img[y:y+h, x:x+w]
                ROIs.append(ROI)
                coords.append([x,y, x+w, y+h])
                ratio = float(w)/h
                if ratio >= 0.9 and ratio <= 1.1:
                    img = cv2.drawContours(img, [cnt], -1, (0,255,255), 3)
                else:
                    img = cv2.drawContours(img, [cnt], -1, (0,255,0), 3)
        elif w>50 and h>30:
          """This is exceptional case where the text inside the rectangle/square box are overlapping in this case 
          will more than 4 sides and make sure it is not considered text as rectangle"""
          print(f'value of approx in this situation : {len(approx)}')
          x, y, w, h = cv2.boundingRect(cnt)
          ROI = img[y:y+h, x:x+w]
          ROIs.append(ROI)
          coords.append([x,y, x+w, y+h])
          if ratio >= 0.9 and ratio <= 1.1:
                    img = cv2.drawContours(img, [cnt], -1, (0,255,255), 3)
          else:
                    img = cv2.drawContours(img, [cnt], -1, (0,255,0), 3)
    blank_space_coords=[]
    text_coords=[]
    count=0
    text_id=0
    for id, i in enumerate(ROIs):
        roi = i
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray,230,255,cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        if len(contours) > 1:
            text_coords.append({'text_'+str(text_id): coords[id]})
            text_id+=1
            x1, y1,x2,y2= coords[id]
            cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)

            logger.info('DO OCR HERE')
        else:
            logger.info(f'len of contour: {len(contours)}')
            blank_space_coords.append({"blank_"+str(count):coords[id]})
            x1, y1,x2,y2= coords[id]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
            logger.info('BLANK SPACE')
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 2)
            count+=1
    logger.info(f'blank space count: {count}')
    logger.info(f'text block count: {len(text_coords)}')
    return blank_space_coords, text_coords



def  text_blank_Field_detection_roi(img: object, roi: list):
    x1, y1, x2, y2 = roi
    img= img[y1:y2, x1:x2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,230,255,0)
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    ROIs=[]
    coords=[]
    for cnt in contours:
        x1,y1 = cnt[0][0]
        approx = cv2.approxPolyDP(cnt, 0.005*cv2.arcLength(cnt, True), True)
        x, y, w, h = cv2.boundingRect(cnt)
        if len(approx) == 4:
            if w>50 and h>30:
                ROI = img[y:y+h, x:x+w]
                ROIs.append(ROI)
                coords.append([x,y, x+w, y+h])
                ratio = float(w)/h
                if ratio >= 0.9 and ratio <= 1.1:
                    img = cv2.drawContours(img, [cnt], -1, (0,255,255), 3)
                else:
                    img = cv2.drawContours(img, [cnt], -1, (0,255,0), 3)
        elif w>50 and h>30:
          """This is exceptional case where the text inside the rectangle/square box are overlapping in this case 
          will more than 4 sides and make sure it is not considered text as rectangle"""
          print(f'value of approx in this situation : {len(approx)}')
          x, y, w, h = cv2.boundingRect(cnt)
          ROI = img[y:y+h, x:x+w]
          ROIs.append(ROI)
          coords.append([x,y, x+w, y+h])
          if ratio >= 0.9 and ratio <= 1.1:
                    img = cv2.drawContours(img, [cnt], -1, (0,255,255), 3)
          else:
                    img = cv2.drawContours(img, [cnt], -1, (0,255,0), 3)
    blank_space_coords=[]
    text_coords=[]
    all_coords= {}
    count=0
    text_id=0
    for id, i in enumerate(ROIs):
        roi = i
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray,230,255,cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        if len(contours) > 1:
            text_coords.append({'text_'+str(text_id): coords[id]})
            all_coords['text_'+str(text_id)]=  coords[id] 
            text_id+=1
            x1, y1,x2,y2= coords[id]
            cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)

            print('DO OCR HERE')
        else:
            print(f'len of contour: {len(contours)}')
            blank_space_coords.append({"blank_"+str(count):coords[id]})
            all_coords['blank_'+str(count)]=  coords[id] 
            x1, y1,x2,y2= coords[id]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
            print('BLANK SPACE')
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 2)
            count+=1
    print(f'blank space count: {count}')
    print(blank_space_coords)
    print(f'text block count: {len(text_coords)}')
    print(text_coords)
    return blank_space_coords, text_coords, all_coords, img

   


def get_ocr_tesseract(img):
    '''
    Performs OCR using Tesseract on an image.

    Parameters:
    - img: The image on which OCR is performed.

    Returns:
    - tuple: A tuple containing two elements - word_coordinates (list) and all_text (str).
    '''
    # Parameter Type Checking
    assert isinstance(img, np.ndarray), "img must be a NumPy array."

    logger.info("called OCR tesseract...")
    # img = cv2.imread(image)
    # hImg,wImg,_ = img.shape
    d = image_to_data(img, lang='eng', config='--psm 3')
    all_text = image_to_string(img, lang='eng', config= "--psm 3")
    word_coordinates = []
    for i, b in enumerate(d.splitlines()):
        if i != 0:
            b = b.split()
            if len(b) == 12:
                word = b[11]
                x, y, w, h = int(b[6]), int(b[7]), int(b[8]), int(b[9])
                dic = {
                    "word": word,
                    "left": x,
                    "top": y,
                    "width": w,
                    "height": h,
                    "x1": x,
                    "y1": y,
                    "x2": x + w,
                    "y2": y + h,
                    "coordinates": [x, y, x + w, y + h],
                    "confidence": float(b[10])
                }
                if dic not in word_coordinates:
                    word_coordinates.append(dic)
                else:
                    continue
    return word_coordinates, all_text

def get_ocr_using_roi(ROI: list, image_path:str):
    '''
    Performs OCR using Tesseract on a region of interest (ROI) within an image.

    Parameters:
    - ROI (list): A list containing the coordinates [x1, y1, x2, y2] of the region of interest.
    - image_path (str): The path to the input image.

    Returns:
    - tuple: A tuple containing two elements - word_coordinates (list) and all_text (str).
    '''
     # Parameter Type Checking
    assert isinstance(ROI, list) and len(ROI) == 4, "ROI must be a list of 4 integers."
    assert all(isinstance(coord, int) for coord in ROI), "All elements in ROI must be integers."
    assert isinstance(image_path, str), "image_path must be a string."

    img= cv2.imread(image_path)
    x1,y1,x2,y2= ROI
    img = img[y1:y2, x1:x2]
    d = image_to_data(img, lang='eng', config='--psm 3')
    all_text = image_to_string(img, lang='eng', config= "--psm 3")
    print(all_text)
    word_coordinates = []
    for i, b in enumerate(d.splitlines()):
        if i != 0:
            b = b.split()
            if len(b) == 12:
                word = b[11]
                x, y, w, h = int(b[6]), int(b[7]), int(b[8]), int(b[9])
                dic = {
                    "word": word,
                    "left": x,
                    "top": y,
                    "width": w,
                    "height": h,
                    "x1": x,
                    "y1": y,
                    "x2": x + w,
                    "y2": y + h,
                    "coordinates": [x, y, x + w, y + h],
                    "confidence": float(b[10])
                }
                if dic not in word_coordinates:
                    word_coordinates.append(dic)
                else:
                    continue
    return word_coordinates, all_text

def plot_bounding_boxes(bb1=None, bb2=None, img=None, top_percentage=None, color=None, top_bottom: list = None):
    if img is not None:
        plt.figure(figsize=(10, 10))
        if bb1 is not None:
            draw_rectangle(img, bb1, color if color is not None else 'g')
        if bb2 is not None:
            draw_rectangle(img, bb2, color if color is not None else 'r')
        if top_percentage is not None:
            height, width, _ = img.shape
            top_height = int(height * (top_percentage / 100))
            top_25_image = img[:top_height, :]
        if top_bottom is not None:
            height, width, _ = img.shape
            top_height = int(height * (top_bottom[0] / 100))
            bottom_height = int(height * ((100 - top_bottom[1]) / 100))
            top_25_image = img[top_height:bottom_height, :]
        else:
            top_25_image = img
        plt.imshow(top_25_image)
        plt.show()
    else:
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot bbox1
        bbox1 = patches.Rectangle((bb1[0], bb1[1]), bb1[2] - bb1[0], bb1[3] - bb1[1], edgecolor='blue',
                                  facecolor='none', label='bbox1')
        ax.add_patch(bbox1)

        # Plot bbox2
        bbox2 = patches.Rectangle((bb2[0], bb2[1]), bb2[2] - bb2[0], bb2[3] - bb2[1], edgecolor='green',
                                  facecolor='none', label='bbox2')
        ax.add_patch(bbox2)

        ax.scatter(bb1[0], bb1[1], color='purple', label='y1')
        ax.scatter(bb1[0], bb2[1], color='orange', label='y2')

        # Set labels and legend
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.legend(loc='best', bbox_to_anchor=(-0.5, 0.5, 0.5, 0.5))
        # Set plot limits with some padding

        ## to plot based on OCR coordinates.
        ##first quadrant
        ax.set_xlim(0, max(bb1[2], bb2[2]) + 100)

        ax.set_ylim(max(bb1[3], bb2[3]) + 100, 0)

        plt.show()


def get_iou(bb1, bb2, img=None):
    # print(f"bb1 is {bb1}")
    # print(f"bb2 is {bb2}")
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {0, '2', 1, '3'}
        The (x1, 1) position is at the top left corner,
        the (2, 3) position is at the bottom right corner
    bb2 : dict
        Keys: {0, '2', 1, '3'}
        The (x, y) position is at the top left corner,
        the (2, 3) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """

    # determine the coordinates of the intersection rectangle
    x1min = min(bb1[0], bb2[0])
    x1max = max(bb1[0], bb2[0])
    x2min = min(bb1[2], bb2[2])
    x2max = max(bb1[2], bb2[2])
    y1min = min(bb1[1], bb2[1])
    y1max = max(bb1[1], bb2[1])
    y2min = min(bb1[3], bb2[3])
    y2max = max(bb1[3], bb2[3])

    isOverlapping = (x1min < x2max and x2min > x1max and y1min < y2max and y2min > y1max)
    return isOverlapping


def detect_box(image, line_min_width=15, threshold=150, maxval=255,
               kernal_x=3, kernal_y=3, dilate_iterations=1, connectivity=8, min_len=10, max_len=50):
    gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    th1, img_bin = cv2.threshold(gray_scale, threshold, maxval, cv2.THRESH_BINARY)
    kernal6h = np.ones((1, line_min_width), np.uint8)
    kernal6v = np.ones((line_min_width, 1), np.uint8)
    img_bin_h = cv2.morphologyEx(~img_bin, cv2.MORPH_OPEN, kernal6h)
    img_bin_v = cv2.morphologyEx(~img_bin, cv2.MORPH_OPEN, kernal6v)
    img_bin_final = img_bin_h | img_bin_v
    final_kernel = np.ones((kernal_x, kernal_y), np.uint8)
    img_bin_final = cv2.dilate(img_bin_final, final_kernel, iterations=dilate_iterations)
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(~img_bin_final, connectivity=connectivity,
                                                                     ltype=cv2.CV_32S)
    ## for checkboxes

    filtered_stats = [stat for stat in stats if min_len < stat[2] < max_len and min_len < stat[3] < max_len]

    return filtered_stats, labels


def imshow_components(labels, heu_value=179):
    ### creating a hsv image, with a unique hue value for each label
    label_hue = np.uint8(heu_value * labels / np.max(labels))
    ### making saturation and volume to be 255
    empty_channel = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, empty_channel, empty_channel])
    ### converting the hsv image to BGR image
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    labeled_img[label_hue == 0] = 0
    ### returning the color image for visualising Connected Componenets
    return labeled_img


def draw_rectangle(image, bbox, color='r',output_path=None):
    if len(bbox) == 4:
        x1, y1, x2, y2 = bbox
    else:
        x1, y1, x2, y2, _ = bbox

    if color == 'r':
        rect_color = (0, 0, 255)  # BGR values for red
    elif color == 'g':
        rect_color = (0, 255, 0)  # BGR values for green
    elif color == 'b':
        rect_color = (255, 0, 0)  # BGR values for blue
    elif color == 'bl':
        rect_color = (255, 255, 255)  # BGR values for blue
    elif color == 'y':
        rect_color = (0, 255, 255)  # BGR values for blue
    else:
        rect_color = color
    cv2.rectangle(image, (x1, y1), (x2, y2), color=rect_color, thickness=2)
    if output_path is not None:
        cv2.imwrite(os.path.join(output_path,"temp.png"), image)


def get_ocr_util(img, json_path=None, text_path=None, dump_data=False):
    os.makedirs("ocr_save_data",exist_ok=True)
    if json_path is None:
        json_path = "ocr_save_data/word_coordinates.json"
    if text_path is None:
        text_path = "ocr_save_data/all_text.txt"
    """
    Perform OCR using Tesseract on the given image.

    Parameters:
    - img: The input image for OCR.
    - json_path: The path to save word coordinates in JSON format.
    - text_path: The path to save all extracted text in a text file.

    Returns:
    - word_coordinates: A dictionary containing word coordinates.
    - all_text: The extracted text from the image.
    """
    if dump_data:
        word_coordinates, all_text = get_ocr_tesseract(img)
        with open(json_path, 'w') as f:
            json.dump(word_coordinates, f, indent=4)
        with open(text_path, 'w') as f:
            f.write(all_text)
        return word_coordinates, all_text

    if os.path.exists(json_path) and os.path.exists(text_path):
        with open(json_path, 'r') as f:
            word_coordinates = json.load(f)
        with open(text_path, 'r') as f:
            all_text = f.read()
    else:
        word_coordinates, all_text = get_ocr_tesseract(img)
        with open(json_path, 'w') as f:
            json.dump(word_coordinates, f, indent=4)
        with open(text_path, 'w') as f:
            f.write(all_text)

    return word_coordinates, all_text


def custom_sort_boxes(boxes_dict):
    sorted_keys = sorted(boxes_dict.keys(), key=lambda k: (boxes_dict[k][3], boxes_dict[k][0]))
    sorted_boxes = {k: boxes_dict[k] for k in sorted_keys}
    result = {key: sorted_boxes[key] for key in sorted_keys}
    return result


def shift_word_coordinates(word_coordinates, roi,reversed = False):
    x_offset, y_offset = roi[0], roi[1]  # x_offset is x1 and y_offset is y1

    shifted_coordinates = []
    if isinstance(word_coordinates, list) and len(word_coordinates)==4 and isinstance(word_coordinates[0],int):
        coords = word_coordinates
        return  [coords[0] + x_offset if reversed  else coords[0]-x_offset,  # x1 - y_offset
                          coords[1] + y_offset if reversed else coords[1] - y_offset,  # x2 - x_offset
                          coords[2] + x_offset if reversed else coords[2] - x_offset,  # y1 - y_offset
                          coords[3] + y_offset if reversed else coords[3] - y_offset]  # y2 - x_offset

    if isinstance(word_coordinates,dict):
        word_coordinates = word_coordinates.get("data",word_coordinates)
    for word_info in word_coordinates:
        word = word_info["word"]
        coords = word_info["word_coordinates"]
        # Shift the coordinates based on the ROI
        shifted_coords = [coords[0] + x_offset if reversed  else coords[0]-x_offset,  # x1 - y_offset
                          coords[1] + y_offset if reversed else coords[1] - y_offset,  # x2 - x_offset
                          coords[2] + x_offset if reversed else coords[2] - x_offset,  # y1 - y_offset
                          coords[3] + y_offset if reversed else coords[3] - y_offset]  # y2 - x_offset

        shifted_word_info = {
            "word": word,
            "confidence": word_info["confidence"],
            "word_coordinates": shifted_coords,
            "x1": shifted_coords[0],
            "y1": shifted_coords[1],
            "x2": shifted_coords[2],
            "y2": shifted_coords[3],
            "top": shifted_coords[0],
            "left": shifted_coords[1]
        }

        shifted_coordinates.append(shifted_word_info)

    return shifted_coordinates

def draw(imggg_copy,point):
    imggg_copy = imggg_copy.copy()
    x1,y1,x2,y2 = point
    cv2.rectangle(imggg_copy, (x1, y1), (x2, y2), (255, 0, 255), 2)
    cv2.imshow("image", imggg_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def check_box_pp(img, data, threshold_distance=25, threshold_distance_y=5,word_coordinates = None, roi = False):

    filtered_check_boxes = {}
    failed_check_boxes = {}
    filtered_list = []
    from ast import literal_eval
    data_keys_list = list(data.keys())
    logger.info("OCR Process Start")
    if not word_coordinates:
        word_coordinates, _ = get_ocr_util(img, dump_data=True)
    else:
        word_coordinates = shift_word_coordinates(word_coordinates,roi)
    # with open('/home/ntlpt59/MAIN/tf_final_tarun_nt/cb_s/shifterd_word_coordinates.json', 'w') as f:
    #     json.dump(word_coordinates, f, indent=4)
    imggg_copy = img.copy()

    # for w in word_coordinates:
    #     print(w['word'])
    #     x1, y1, x2, y2 = w.get("word_coordinates")
    #     if all([x>0 for x in w['word_coordinates']]):
    #         print(x1, y1, x2, y2)
    #         # cv2.rectangle(imggg_copy,(x1,y1),(x2,y2),(255,0,255),2)
    #         # cv2.rectangle(imggg_copy,(5,5),(12,29),(255,0,255),2)
    #         cv2.rectangle(imggg_copy,(302,5),(326,33),(255,0,255),2)
    # x1,y1,x2,y2 = [94,7,110,23]
    # x1,y1,x2,y2 = [5, 5, 12, 29]
    # x1,y1,x2,y2 = [
    #     276,
    #     6,
    #     293,
    #     23
    # ]

    if len(word_coordinates):
        print(f"\nOCR Done successfully total words: {len(word_coordinates)}")
    checkbox_index = 0
    index1 = 0
    word_coordinates = sorted(word_coordinates,key = lambda x: ((x['word_coordinates'][0]+x['word_coordinates'][1])/2,(x['word_coordinates'][2]+x['word_coordinates'][3])/2))
    # with open('/home/ntlpt59/MAIN/tf_final_tarun_nt/cb_s/sorted_shifted_word_coordinates.json', 'w') as f:
    #     json.dump(word_coordinates, f, indent=4)

    while (index1 < len(word_coordinates)):
        ocr_word_coord = word_coordinates[index1]
        # if index1 in [347, 120, 106]:
        #     index1 += 1
        #     continue
        image_copy = img.copy()
        ocr_word_coordinate = ocr_word_coord.get('coordinates',ocr_word_coord.get('word_coordinates'))
        if checkbox_index + 1 > len(data_keys_list):
            break
        current_check_box = data[data_keys_list[checkbox_index]]
        is_overlapping = get_iou(ocr_word_coordinate, current_check_box)
        w = abs(current_check_box[2] - current_check_box[0])
        h = abs(current_check_box[3] - current_check_box[1])
        x1, y1, x2, y2 = ocr_word_coordinate
        x3, y3, x4, y4 = current_check_box
        # print('################################################################')
        # print("index:",index1)
        # print("ocr_word_coordinate",ocr_word_coordinate)
        # draw(imggg_copy,ocr_word_coordinate)
        # print("current_check_box",current_check_box)
        # draw(imggg_copy,current_check_box)
        # print(is_overlapping)
        # print(
        #     f"point: x1:{x1}, y1:{y1},x2:{x2},y2:{y2} , thresDistance: {threshold_distance}, thresY: {threshold_distance_y}")
        # print(f"point: x3:{x3}, y3:{y3},x4:{x4},y4:{y4} , thresDistance: {threshold_distance}, thresY: {threshold_distance_y}")
        # print(f"w: {w}, h: {h}, w/h: {w / h}, x1-x4: {(x1 - x4)}, x3-x2: {(x3 - x2)}, y3-y2:{(y3 - y2)},y1-y4: {(y1 - y4)}")
        # print(f"x1-x4: {(x1 - x4)}, x3-x2: {(x3 - x2)}, y3-y2:{(y3 - y2)},y1-y4: {(y1 - y4)}")
        is_overlapping = False ## Ignore overlapping case
        x_max = max((x1 - x4),(x3 - x2))
        y_max = max((y3 - y2),(y1 - y4))

        # data[data_keys_list[checkbox_index]] = shift_word_coordinates(data[data_keys_list[checkbox_index]],roi=roi,reversed = True)
        # ocr_word_coord = shift_word_coordinates(ocr_word_coord,roi=roi,reversed = True)
        if is_overlapping:
            # print("##### Is overlapping so fail #####")
            # breakpoint()
            failed_check_boxes.update({data_keys_list[checkbox_index]: {
                "checkbox_coord": data[data_keys_list[checkbox_index]], "ocr_coord": ocr_word_coord}})
            checkbox_index += 1
        elif (not is_overlapping) and (w / h >= 0.8) and (x_max<0 or (x_max<= threshold_distance)) and (y_max<0 or y_max<= threshold_distance_y):
            # print("##### Not overlapping #####")
            # print(f"point: x1:{x1}, y1:{y1},x2:{x2},y2:{y2} , thresDistance: {threshold_distance}, thresY: {threshold_distance_y}")
            # print(f"w: {w}, h: {h}, w/h: {w / h}, x1-x4: {(x1 - x4)}, x3-x2: {(x3 - x2)}, y3-y2:{(y3 - y2)},y1-y4: {(y1 - y4)}")
            # breakpoint()
            filtered_check_boxes.update({data_keys_list[checkbox_index]: data[data_keys_list[checkbox_index]]})
            filtered_list.append(current_check_box)
            failed_check_boxes.update({})
            checkbox_index += 1
        elif y1 >= y3 or ((not is_overlapping) and (
                (x_max<0 or x_max > threshold_distance) and (x1 >= x3 and y1 >= y3)) and (y_max<0 or y_max <= threshold_distance_y)):
            # print("##### Last Condition #####")
            # print(
            #     f"point: x1:{x1}, y1:{y1},x2:{x2},y2:{y2} , thresDistance: {threshold_distance}, thresY: {threshold_distance_y}")
            # print(f"x1-x4: {(x1 - x4)}, x3-x2: {(x3 - x2)}, y3-y2:{(y3 - y2)},y1-y4: {(y1 - y4)}")
            # breakpoint()
            filtered_check_boxes.update({data_keys_list[checkbox_index]: data[data_keys_list[checkbox_index]]})
            filtered_list.append(current_check_box)
            checkbox_index += 1
            index1 = index1 - 1
        else:
            # print("Inside Else")
            # print(f"checkbox index: {checkbox_index},datakeysList: {data_keys_list}")
            # breakpoint()
            pass
        if checkbox_index < len(data_keys_list):
            pass
        index1 += 1
    return filtered_check_boxes, filtered_list, failed_check_boxes


def save_results(__file_name,out_folder, imag_copy=None, data=None,flag = True):
    if not flag: return
    if not os.path.exists(out_folder):
        os.makedirs(out_folder, exist_ok=True)
    if __file_name[-4:] == "json":
        with open(os.path.join(out_folder, __file_name), 'w') as f:
            json.dump(data, f, indent=4)
    elif __file_name[-3:] in ['png', 'jpg', 'jpeg']:
        cv2.imwrite(os.path.join(out_folder, __file_name), imag_copy)
    elif __file_name[-3:] == "txt":
        with open(os.path.join(out_folder, __file_name, 'w')) as f:
            f.write(data)


def boxes_data_creation(stats, imag_copy):
    data = {}
    i = 0
    for x, y, w, h, area in stats:
        y2 = int(h) + int(y)
        x2 = int(w) + int(x)
        data.update({f"box_{i}": [int(x), int(y), x2, y2]})
        i = i + 1
    return data

def is_checked_box(img,bbox_coord,contour_area_threshold=100,roi_thres = 230):
    x1,y1,x2,y2= bbox_coord
    roi = img[y1:y2,x1:x2]
    gray = cv2.cvtColor( roi,cv2.COLOR_BGR2GRAY)
    _,binary_roi = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(f"len of contours: {len(contours)}")
    # print(f"roi mean is :{roi.mean()}")
    # brighPC = (roi>230).sum()
    # print(f"bpc: {brighPC}")
    # print(f"roisize: {roi.size}")
    # print(f"roisize: {0.5*roi.size}")
    is_checked = roi.mean()<roi_thres
    if is_checked:  # Adjust the threshold as needed
        print("Checkbox is  checked")
    else:
        print("Checkbox is  not checked")
    return is_checked


def checkbox_passorfail(img,data:dict):
    print(data)
    filtered_data = {}
    for box_id,box_coord in data.items():
        print(f"box id : {box_id}")
        img_cp = img.copy()
        # plot_bounding_boxes(bb1=box_coord,img = img_cp,color=(255,0,0),top_bottom=[0,50])
        if is_checked_box(img,box_coord):
            filtered_data.update({box_id:{'ischecked':True,'coordinates':box_coord}})
        else:
            filtered_data.update({box_id:{'ischecked':False,'coordinates':box_coord}})
    return filtered_data

def draw_put_text(img,stats_dict:dict):
    for k,v in stats_dict.items():
        x1,y1,x2,y2 = v
        cv2.putText(img, f"{k}", (x1+5, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
def checkbox_detection_roi(seg_id: str=None, ROI:list=None,image_path:list= None,file_name:str=None,img=None,ocr_coordinates = None,
                           params = {}):
    data_final ={}
    print("############### CHECKBOX DETECTION START ################")


    #parameters
    dilation_kernal = params.get('dilation_kernel', (2, 2))
    erode_iterations = params.get('erode_iterations', 1)
    line_min_width = params.get('line_min_width', 15)
    box_threshold = params.get('box_threshold', 135)
    min_box_len = params.get('min_box_len', 10)
    max_box_len = params.get('max_box_len', 28)
    save_images = params.get('save_images', False)

    output_folder_path = "outputs_checkbox"
    if save_images:
        os.makedirs(output_folder_path, exist_ok=True)

    if img is None:
        image = cv2.imread(image_path)
        x1, y1, x2, y2= ROI
        image= image[y1:y2, x1:x2]
    else:
        image  = img
        x1, y1, x2, y2= ROI
        image= image[y1:y2, x1:x2]


    if  not image.any():
        data_final.update({"check_boxes_dict": {}})
        data_final.update({"check_boxes_list": {}})
        data_final.update({"failed_check_boxes": {}})
        data_final.update({"checkboxes_is_check_dict": {}})
        return data_final

    imag_copy = image.copy()


    ## erosion of image
    dilation_kernel = np.ones(dilation_kernal, np.uint8)
    image = cv2.erode(image, dilation_kernel, iterations= erode_iterations)

    stats, labels = detect_box(image,
                               line_min_width=line_min_width,
                               threshold=box_threshold,
                               min_len=min_box_len,
                               max_len=max_box_len)

    cc_out = imshow_components(labels)

    data = boxes_data_creation(stats, imag_copy)

    data_final.update({"all_boxes":data})
    data = custom_sort_boxes(data)
    img_checkboxes = image.copy()
    for box_id,box_coord in data.items():
        draw_rectangle(img_checkboxes,box_coord,'r')
    save_results("checkboxes_before_pp.jpg",imag_copy=img_checkboxes,data=data,out_folder=output_folder_path,
                 flag= save_images)
    data_before_pp = data.copy()
    data_before = data.copy()

    data, filtered_list, failed_check_boxes = check_box_pp(imag_copy, data,word_coordinates=ocr_coordinates,roi = ROI)  ## post processing
    data_final.update({"check_boxes_dict":data})
    data_final.update({"check_boxes_list":filtered_list})
    data_final.update({"failed_check_boxes":failed_check_boxes})

    for fil_coord in filtered_list:
        draw_rectangle(imag_copy,fil_coord,'g')
    save_results(__file_name="checkboxes_failed_coord.json",data=failed_check_boxes,out_folder=output_folder_path,flag= save_images)
    save_results(__file_name="checkboxes_failed_coord.json",data=failed_check_boxes,out_folder=output_folder_path,flag= save_images)
    save_results(__file_name = "checkboxes_final.json",data=data,out_folder=output_folder_path,flag= save_images)
    save_results(__file_name= "checkbox_after_pp.jpg",imag_copy=imag_copy,out_folder=output_folder_path,flag= save_images)
    img_fail_cp = image.copy()
    for failed_coord in failed_check_boxes.values():
        draw_rectangle(image=imag_copy,bbox=failed_coord['checkbox_coord'],color='r')
        draw_rectangle(image=img_fail_cp,bbox=failed_coord['checkbox_coord'],color='r')
        draw_rectangle(image=img_fail_cp,bbox=failed_coord['ocr_coord'].get('word_coordinates',failed_coord['ocr_coord'].get("coordinates")),color='y')

    save_results(__file_name= "checkbox_fail.jpg",imag_copy=img_fail_cp,out_folder=output_folder_path,flag= save_images)
    save_results(__file_name= "checkbox_fail_and_pass.jpg",imag_copy=imag_copy,out_folder=output_folder_path,flag= save_images)


    is_check_image = image.copy()
    data = checkbox_passorfail(is_check_image,data=data)
    data_final.update({"checkboxes_is_check_dict":data})
    print("################# CHECKBOX DETECTION DONE ###################")

    # print("BeFORE SHIFTING CHECKBOXES ")
    # breakpoint()
    # print(data_final)
    # breakpoint()
    # print(f'RIOI:{ROI}')
    # breakpoint()
    data_final = shift_coordinates_back_original_image(data_final,ROI)
    # print("AFTER shifting checkboxes")
    # breakpoint()
    # print(data_final)
    # breakpoint()
    data = data_final["checkboxes_is_check_dict"]
    return data,data_final

def add_roi_to_coordinate(coordinate,roi):
    x1,y1,x2,y2 = coordinate
    x1,y1,x2,y2 = x1+roi[0],y1+roi[1],x2+roi[0],y2+roi[1]
    return [x1,y1,x2,y2]
def shift_coordinates_back_original_image(response, point):
    updated_response: dict = {}
    traverse_response = deepcopy(response)
    for key,value in traverse_response.items():
        match key:
            case 'all_boxes' | 'check_boxes_dict' | 'failed_check_boxes':
                updated_response[key] = {k:add_roi_to_coordinate(v,point)for k,v in traverse_response[key].items()}
            case 'check_boxes_list':
                updated_response[key] = [add_roi_to_coordinate(_,point) for _ in traverse_response[key]]
            case 'checkboxes_is_check_dict':
                updated_response[key] = {k:{'ischecked':v['ischecked'],'coordinates':add_roi_to_coordinate(v['coordinates'],point)} for k,v in traverse_response[key].items()}
    return updated_response

def checkbox_post_processing(img,data,padding= 2):
    data_pp, data_fail = {},{}
    for box_id,box_info in data.items():
        import copy 
        img_copy = copy.deepcopy(img)
        x1,y1,x2,y2 = box_info["coordinates"]
        try:
            roi = img_copy[y1-padding:y2+padding,x1-padding:x2+padding]
            ocr_string = pytesseract.image_to_string(roi)
        except Exception as e:
            roi = img_copy[y1:y2,x1:x2]
            ocr_string = pytesseract.image_to_string(roi)
        if not ocr_string:
            data_pp.update({box_id:box_info})
        else:
            data_fail.update({box_id:box_info}) 

    return data_pp,data_fail


# def checkbox_detection_roi(seg_id: str=None, ROI:list=None,image_path:list= None,file_name:str=None,img=None):
#     data_final ={}
#     print("############### CHECKBOX DETECTION START ################")
#     output_folder_path = "outputs_checkbox"
#     os.makedirs(output_folder_path,exist_ok=True)
#
#     if img is None:
#         image = cv2.imread(image_path)
#         x1, y1, x2, y2= ROI
#         image= image[y1:y2, x1:x2]
#         cv2.imshow(f"image in checkbox", image)
#         terminate()
#     else:
#         image  = img
#         x1, y1, x2, y2= ROI
#         image= image[y1:y2, x1:x2]
#
#     imag_copy = image.copy()
#
#     ## erosion of image
#     dilation_kernel = np.ones((3, 3), np.uint8)
#     image = cv2.erode(image, dilation_kernel, iterations=1)
#     stats, labels = detect_box(image,
#                                line_min_width=15,
#                                threshold=135,
#                                min_len=10,
#                                max_len=25)
#
#     data = boxes_data_creation(stats, imag_copy)
#
#     # before sorting
#
#     data_final.update({"all_boxes":data})
#     data = custom_sort_boxes(data)
#
#     img_checkboxes = image.copy()
#     data_json = {}
#     for box_id,box_coord in data.items():
#         draw_rectangle(img_checkboxes,box_coord,'r')
#         data_json.update({box_id:{"is_checked":None,"coordinates":box_coord}})
#
#
#     save_results("checkboxes_before_pp.jpg",imag_copy=img_checkboxes,data=data,out_folder=output_folder_path)
#     save_results("checkboxes_coords_before_pp.json",output_folder_path,data = data_json)
#
#     data_json_pp,data_failed = checkbox_post_processing(image,data_json,padding=10)
#
#     img_fail_cp = image.copy()
#
#     for box_id in data.keys():
#         if box_id in data_json_pp:
#             draw_rectangle(image=img_fail_cp, bbox=data_json_pp[box_id]["coordinates"], color='g')
#         else:
#             draw_rectangle(image=img_fail_cp, bbox=data_failed[box_id], color='r')
#     save_results(__file_name= "checkbox_discarded_red.jpg",imag_copy=img_fail_cp,out_folder=output_folder_path)
#
#     for k,v in data_json_pp.items():
#         data[k] = v['coordinates']
#
#     is_check_image = image.copy()
#     data = checkbox_passorfail(is_check_image, data=data)
#     data_final.update({"checkboxes_is_check_dict": data})
#     save_results("checkbox_is_check_final.json", output_folder_path, data=data)
#
#     img_fail_cp = image.copy()
#     for box_id in data.keys():
#         print(data)
#         if data[box_id]["ischecked"]:
#             draw_rectangle(image=img_fail_cp, bbox=data[box_id]["coordinates"], color='g')
#         else:
#             draw_rectangle(image=img_fail_cp, bbox=data[box_id]["coordinates"], color='r')
#     save_results(__file_name="checkbox_is_checked.jpg", imag_copy=img_fail_cp, out_folder=output_folder_path)
#
#     save_results("checkbox_master_json.json", output_folder_path, data=data_final)
#
#     print("################# DONE ###################")
#
#     return data,data_final
#


def ocr_extraction():
    pass


if __name__ == "__main__":
    # os.makedirs("cb_s",exist_ok=True)
    # img = cv2.imread('/home/ntlpt59/MAIN/tf_final_tarun_nt/cb_s/image.jpg')
    # with open('/home/ntlpt59/MAIN/tf_final_tarun_nt/cb_s/data.json', 'r') as f:
    #     data = json.load(f)
    # with open('/home/ntlpt59/MAIN/tf_final_tarun_nt/cb_s/word_coordinates.json', 'r') as f:
    #     word_coordinates = json.load(f)
    # with open('cb_s/roi.txt','w') as f:
    # roi = [490, 551, 1652, 605]
    # from client_code.trade_finance_structure_document.src.main.definitions import Tesseract
    # updated_word_coordinates = Tesseract().filter_coordinates(word_coordinates["data"],((roi[0],roi[1]),(roi[2],roi[3])))
    # with open('/home/ntlpt59/MAIN/tf_final_tarun_nt/cb_s/filtered_coords.json', 'w') as f:
    #      json.dump(updated_word_coordinates, f,indent=1)
    # a,b,c =  check_box_pp(img=img,data = data,word_coordinates=updated_word_coordinates,roi=roi)
    # print(a)
    exit('+++++++++++++++')
    file_name= "LC_export0"
    image_path= "data/LC_export0.jpg"
    block_detection(image_path, file_name)
    exit('++++++++++++++')
    img_path = 'data/output/crop/LC_export0/box_5.jpg'
    img = cv2.imread('data/output/crop/LC_export0/box_5.jpg')
    # image= img.copy()
    # boxes_dict = {'text_12': [3, 47, 305, 91], 'text_13': [658, 2, 860, 91], 'text_14': [3, 2, 305, 45], 'blank_8': [307, 47, 656, 91], 'blank_9': [862, 2, 1564, 91], 'blank_10': [307, 2, 656, 45]}
    # sorted_dict = custom_sort_boxes(boxes_dict)
    # print('*'*20)
    # print(boxes_dict)
    # print('*'*20)
    # print(sorted_dict)
    # img = cv2.imread('/home/ntlpt59/MAIN/trade_finance_codes/structure_documents/structure_document_manikanta_ruppa_repo/src/main/data/output/crop/LC_export0/box_5.jpg')

    # draw_put_text(img,sorted_dict)
    # plt.imshow(img)
    # plt.show()
    # exit('++++')
    # img = cv2.imread('data/output/crop/LC_export0/box_5.jpg')
    # image = img.copy()
    blank_space_coords, text_coords= text_blank_Field_detection(img_path)
    # print(f'text boxes count: {len(text_coords)}')
    # print(text_coords)
    # draw_coords(image, text_coords)
    line_segmentation(img_path)
    # exit('+++++++++++++++++++')
    exit('+++++++++++++++++++')
    # line_segmentation(img)
    file_name = "LC_export0_check_box.png"
    # image_path = "data/LC_export0.jpg"
    image_path = os.path.join("../data/Export_app1.jpg")
    output_folder_path = "../data"
    # block_detection(image_path, file_name)
    data = checkbox_detection(image_path,file_name)
    print(data)
    exit('+++++++++=')
    img  = cv2.imread(image_path)
    with open('/home/ntlpt59/MAIN/trade_finance_codes/structure_documents/structure_document_manikanta_ruppa_repo/src/main/data/checkboxes_final.json','r') as f:
        data = json.load(f)
    for v in data.values():
        img_d = img.copy()
        cx = is_checked_box(img_d,v)
        plot_bounding_boxes(img=img_d,bb1=v,color=(255,0,0),top_bottom=[0,25])
        print(f"checkes : {cx}")















