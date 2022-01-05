"""
Character Detection

The goal of this task is to implement an optical character recognition system consisting of Enrollment, Detection and Recognition sub tasks

Please complete all the functions that are labelled with '# TODO'. When implementing the functions,
comment the lines 'raise NotImplementedError' instead of deleting them.

Do NOT modify the code provided.
Please follow the guidelines mentioned in the project1.pdf
Do NOT import any library (function, module, etc.).
"""


import argparse
import json
import os
import glob
import cv2
import numpy as np


def read_image(img_path, show=False):
    """Reads an image into memory as a grayscale array.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if show:
        show_image(img)

    return img


def show_image(img, delay=1000):
    """Shows an image.
    """
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('image', img)
    cv2.waitKey(delay)
    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--test_img", type=str, default="./data/test_img.jpg",
        help="path to the image used for character detection (do not change this arg)")
    parser.add_argument(
        "--character_folder_path", type=str, default="./data/characters",
        help="path to the characters folder")
    parser.add_argument(
        "--result_saving_directory", dest="rs_directory", type=str, default="./",
        help="directory to which results are saved (do not change this arg)")
    args = parser.parse_args()
    return args


def ocr(test_img, characters):
    """Step 1 : Enroll a set of characters. Also, you may store features in an intermediate file.
       Step 2 : Use connected component labeling to detect various characters in an test_img.
       Step 3 : Taking each of the character detected from previous step,
         and your features for each of the enrolled characters, you are required to a recognition or matching.

    Args:
        test_img : image that contains character to be detected.
        characters_list: list of characters along with name for each character.

    Returns:
    a nested list, where each element is a dictionary with {"bbox" : (x(int), y (int), w (int), h (int)), "name" : (string)},
        x: row that the character appears (starts from 0).
        y: column that the character appears (starts from 0).
        w: width of the detected character.
        h: height of the detected character.
        name: name of character provided or "UNKNOWN".
        Note : the order of detected characters should follow english text reading pattern, i.e.,
            list should start from top left, then move from left to right. After finishing the first line, go to the next line and continue.

    """
    # TODO Add your code here. Do not modify the return and input arguments

    enrollment(characters)

    detection(test_img)

    return recognition(test_img, characters)

    # raise NotImplementedError


def enrollment(characters):
    """ Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 1 : Your Enrollment code should go here.

    # Initialzing placeholder for Harris corner features for enrolled images
    temp_list = []

    # Looping over all enrolled images
    for i in characters:

        # Get the image data for i'th image
        image_i = i[1]

        # Calculated the Harris corner
        corners = cv2.cornerHarris(image_i, 2, 3, 0.06)
        # Resize to a common size
        corners = cv2.resize(corners, (30, 30))

        # Append to the placeholder temp_list
        temp = [i[0], corners]
        temp = np.array(temp)
        temp_list.append(temp)

    # Save features for future use
    temp_list = np.array(temp_list)
    np.save('./data/features/features', temp_list)


def detection(test_img):
    """
    Use connected component labeling to detect various characters in an test_img.
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """

    # TODO: Step 2 : Your Detection code should go here.

    # Making a copy of input test_image
    output = test_img.copy()

    # Shape of test_image
    img_x, img_y = output.shape

    # Classes counter (Starting from class 1)
    cls = 1

    # Initializing placeholder for conflicts during 1st Pass
    # Will implement Union-Find via using this array to store Parent-Child link
    # Initialized to maximum possible 747*328 classes but most of it will be sparse
    conf = np.arange(747*328)

    # Initializing placeholder for labeled points
    lp = np.zeros((img_x, img_y))

    # For loop to convert all pixels below a threshold of 235 to 1 & otherwise 0.
    for y in range(0, img_y):
        for x in range(0, img_x):
            if output[x][y] > 235:
                output[x][y] = 0
            else:
                output[x][y] = 1

    # Implementing First Pass: Each pixel is checked for Background, west & north neighbour.
    # The pixel is then labled according to minimum of the possibilities.
    # The conflict is noted as a link between parent(smaller value) -> child(larger value) link in 'conf' list for the index of parent's & child's class.
    # 'output' : Points after thresholding to 0 OR 1
    # 'lp' : labeled points
    for x in range(0, img_x):
        for y in range(0, img_y):

            # Background
            if output[x][y] == 0:
                pass
            else:
                # Check north, west labels
                if output[x][y-1] == 0 and output[x-1][y] == 0:
                    lp[x, y] = cls
                    cls += 1
                elif output[x][y-1] == 0 and output[x-1][y] != 0:
                    lp[x, y] = int(lp[x-1, y])
                elif output[x][y-1] != 0 and output[x-1][y] == 0:
                    lp[x, y] = int(lp[x, y-1])
                elif output[x-1][y] != 0 and output[x][y-1] != 0:
                    if lp[x-1, y] == lp[x, y-1]:
                        lp[x, y] = int(lp[x-1, y])
                    else:
                        # Note down the conflicts in  'conf'
                        if lp[x-1, y] < lp[x, y-1]:
                            lp[x, y] = int(lp[x-1, y])
                            conf[int(lp[x, y-1])] = conf[int(lp[x-1, y])]
                        elif lp[x-1, y] > lp[x, y-1]:
                            lp[x, y] = int(lp[x, y-1])
                            conf[int(lp[x-1, y])] = conf[int(lp[x, y-1])]

    # Implementing Second Pass: to change all conflicted lables to the value of their parent node
    # Parent node is found using the 'conf' list
    for x in range(0, img_x):
        for y in range(0, img_y):
            lp[x, y] = conf[int(lp[x, y])]

    # Initializing placeholder for future data
    data = []

    # Iterate over all classes 'max_cls'
    max_cls = int(np.max(lp))
    for i in range(max_cls):
        # Initializing the bounding parameters for the blob class to 0
        x_i = 0
        y_i = 0
        w_i = 0
        h_i = 0
        # counter to count the number of pixels in blob
        cnt = 0

        # iterate over the image & keep on incrementing  or decrementing based on pixels coordinates
        # so that we can get minimum & maximum of the blob
        for x in range(0, img_x):
            for y in range(0, img_y):
                if int(lp[x, y]) == i:
                    cnt += 1
                    if x_i == 0:
                        x_i = x
                        w_i = x
                    if y_i == 0:
                        y_i = y
                        h_i = y
                    if x_i > x:
                        x_i = x
                    if w_i < x:
                        w_i = x
                    if y_i > y:
                        y_i = y
                    if h_i < y:
                        h_i = y
                    pass
                pass
            pass

        # Offsetting x,y coordinates by one pixel each
        w_i = w_i - x_i
        h_i = h_i - y_i

        x_i = x_i + 1
        y_i = y_i + 1

        # Append the blob only of it's parameters like coordinates & number of pixels are above OR below a threshold
        if x_i != 0 and y_i != 0 and w_i >= 4 and w_i <= 40 and h_i >= 4 and h_i <= 40 and cnt > 20:

            # Append the blob to the placeholder list 'data' in the mentioned form as per assignment
            data.append(
                {"bbox": [int(y_i), int(x_i), int(h_i), int(w_i)], "name": "UNKNOWN"})

    # Saving the data (detected character positions) as for future use
    with open('./data/results.json', 'w') as f:
        json_object = json.dump(data, f)

    # raise NotImplementedError


def recognition(test_img, characters):
    """
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 3 : Your Recognition code should go here.

    # Load the character position data that was created in 'detection()'
    with open('./data/results.json', 'r') as f:
        data = json.load(f)

    # Loading the features data that was calculated in 'enrollment()'
    features = np.load('./data/features/features.npy',  allow_pickle=True)

    # Maximum number of blobs in data
    max_cls = len(data)

    # Initializing placeholder for final data to be returned (i.e. character positions with recognized points)
    data_return = []

    # Copying 'test_image'
    output = test_img.copy()

    # counter
    cntr = 0

    # For loop to iterate over each blob to recognize
    for cntr in range(max_cls):

        # i'th character positions
        i = data[cntr]

        # increment counter
        cntr += 1

        # crop the blob from the image
        crop_img = output[i["bbox"][1]: i["bbox"][1]+i["bbox"]
                          [3], i["bbox"][0]: i["bbox"][0]+i["bbox"][2]]

        # Blob's Harris corner data
        i_corners = cv2.cornerHarris(crop_img, 2, 3, 0.04)
        i_corners = cv2.resize(i_corners, (30, 30))

        # Initializing temporary list as placeholer
        temp_list = []

        # for loop traversing enrolled characters and comparing harris score with each one of it
        for i_f in features:
            i_ft = i_f[1]

            # Compare histograms using 'Cross-Corelation' b/w enrolled character & blob
            cc = cv2.compareHist(i_corners, i_ft, 0)
            temp_list.append(cc)

        # Index of the best candidate
        pred_index = np.argmax(temp_list)

        # Value of the best candidate
        pred_char = features[pred_index][0]

        # If score is less than threshold than change value to "UNKNOWN"
        if temp_list[pred_index] < 0.175:
            pred_char = "UNKNOWN"

        # Appending the data in mentioned format
        data_return.append({"bbox": [int(i["bbox"][0]), int(i["bbox"][1]), int(
            i["bbox"][2]), int(i["bbox"][3])], "name": pred_char})

    return data_return

    # raise NotImplementedError


def save_results(coordinates, rs_directory):
    """
    Donot modify this code
    """
    with open(os.path.join(rs_directory, 'results.json'), "w") as file:
        json.dump(coordinates, file)


def main():
    args = parse_args()

    characters = []

    all_character_imgs = glob.glob(args.character_folder_path + "/*")

    for each_character in all_character_imgs:
        character_name = "{}".format(
            os.path.split(each_character)[-1].split('.')[0])
        characters.append(
            [character_name, read_image(each_character, show=False)])

    test_img = read_image(args.test_img)

    results = ocr(test_img, characters)

    save_results(results, args.rs_directory)


if __name__ == "__main__":
    main()
