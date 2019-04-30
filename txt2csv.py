import sys
import os
import numpy as np
import pandas as pd
import argparse


# Save csv file to output path
def array2csv(merge_array, csvfile):
    # Add header to final csv file
    header = ['class', 's1', 'prob1', 'frame1', 's2', 'prob2', 'frame2', 's3', 'prob3', 'frame3',
              's4', 'prob4', 'frame4', 's5', 'prob5', 'frame5', 's6', 'prob6', 'frame6',
              's7', 'prob7', 'frame7', 's8', 'prob8', 'frame8']

    # Save final array matrix to csv file with header defined
    pd.DataFrame(merge_array).to_csv(csvfile, header=header, index=None)


# Return merged array with label attached
def merge_test(matrix_in, matrix_out, matrix_out2):
    # Convert list[list] to array matrix
    merge_3 = matrix_in + matrix_out + matrix_out2
    merge_array = np.row_stack(merge_3)

    # Replace the first column (speaker_id) with label ('member'/'non-member')
    for i in range(len(merge_array)):
        if i < len(matrix_in):
            merge_array[i, 0] = 'member'
        else:
            merge_array[i, 0] = 'nonmember'

    return merge_array


# Return merged array with label attached
def merge_train(matrix1_in, matrix2_in, matrix1_out, matrix1_out2, matrix2_out, matrix2_out2):
    # Merge list[list]s and convert the merged list[list] to array matrix
    merge_6 = matrix1_in + matrix2_in + matrix1_out + matrix1_out2 + matrix2_out + matrix2_out2
    merge_array = np.row_stack(merge_6)

    # Replace the first column (speaker_id) with label ('member'/'non-member')
    for i in range(len(merge_array)):
        if i < (len(matrix1_in)+len(matrix2_in)):
            merge_array[i, 0] = 'member'
        else:
            merge_array[i, 0] = 'nonmember'

    return merge_array


# Return merged array with label attached
def merge4array(matrix1, matrix2, matrix3, matrix4, label):
    # Merge list[list]s and convert the merged list[list] to array matrix
    merge_4 = matrix1 + matrix2 + matrix3 + matrix4
    merge_array = np.row_stack(merge_4)

    # Replace the first column (speaker_id) with label ('member'/'non-member')
    for i in range(len(merge_array)):
        merge_array[i, 0] = label

    return merge_array


# Return matrix which each row corresponding to one speaker
def matrix2spk(matrix, unique_list):
    matrix_spk = []     # Create an empty list for matrix_speaker
    spk_row = -1        # Row number for matrix_spk

    for uni_spk in unique_list:
        spk1_flg = 0            # Flag for each unique speaker's first record not found
        spk_row += 1            # Row number for matrix_spk
        matrix_spk.append([])   # Create a list for this speaker in matrix_spk list

        for row in range(len(matrix)):
            if uni_spk in matrix[row]:
                # If this unique speaker is found && its first record has not found
                if spk1_flg == 0:
                    spk1_flg = 1    # Flag for the unique speaker's 1st record has found

                    matrix_spk[spk_row].append(matrix[row][0])  # For the (spk_row)th speaker, add matched matrix(row).
                    matrix_spk[spk_row].append(matrix[row][1])
                    matrix_spk[spk_row].append(matrix[row][2])
                    matrix_spk[spk_row].append(matrix[row][3])

                elif spk1_flg == 1:
                    matrix_spk[spk_row].append(matrix[row][1])  # For this speaker, add match matrix[row] except [0] id
                    matrix_spk[spk_row].append(matrix[row][2])
                    matrix_spk[spk_row].append(matrix[row][3])

    if len(matrix_spk) == len(unique_list):
        print("Successfully merge each individual's multiple transcription recordings.")
        return matrix_spk
    else:
        print("Something wrong while merging each individual's transcription recordings.")
        sys.exit()


# Return unique list: refine unique item of the input list1
def unique(list1):
    # intilize a null list
    unique_list = []

    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)

    return unique_list


# Return matrix (list of list)
def txt2matrix(txtfile):
    # Open .txt file for operation
    with open(txtfile) as txt:
        row = 0
        nr = 0
        matrix = []
        for line in txt:
            word = line.split()
            spid = word[0].split("_")           # Split speaker_id & sentence_id

            if (row % 2) == 0:
                word[1] = " ".join(word[2:(len(word) - 2)])     # Merge all phonics to one string

                matrix.append([])               # Create a list in the list
                matrix[nr].append(spid[0])      # Append speaker_id to this list of the list
                matrix[nr].append(word[1])      # Append sentence(phx type) to this list of the list
            else:
                matrix[nr].append(word[2])      # Append probability to this list of the list
                matrix[nr].append(word[4])      # Append frame to this list of the list
                nr += 1
            row += 1

    return matrix


# Return matrix that each row corresponding to one speaker
def txt_matrix2spk(txtfile):
    matrix1 = txt2matrix(txtfile)

    multi_spk = []
    for row in matrix1:
        multi_spk.append(row[0])

    unique_list = unique(multi_spk)

    matrix_spk = matrix2spk(matrix1, unique_list)

    return matrix_spk


def get_arguments():
    parser = argparse.ArgumentParser(description='Description of your path of input and output files.')
    parser.add_argument('in1', type=, help='path to input in_file.txt')
    # parser.add_argument('in2', type=str, help='path to input in_file.txt')
    parser.add_argument('-out1', "--string", type=str, help='path to input out_file.txt')
    parser.add_argument('out2', type=str, help='path to input out_file.txt')
    # parser.add_argument('out3', type=str, help='path to input out_file.txt')
    # parser.add_argument('out4', type=str, help='path to input out_file.txt')
    parser.add_argument('csv', type=str, help='path of output file.csv')
    arguments = parser.parse_args()
    return arguments


if __name__ == '__main__':
    args = get_arguments()

    txt1_in1 = args.in1
    txt1_out1 = args.out1
    txt1_out2 = args.out2
    # txt2_in1 = args.in2
    # txt2_out1 = args.out3
    # txt2_out2 = args.out4
    csvfile = args.csv

    if not os.path.isfile(txt1_in1 | txt1_out1 | txt1_out2):
        print("File path {} or {} or {} does not exist. Exiting...".format(txt1_in1, txt1_out1, txt1_out2))
        sys.exit()

    in_spk1 = txt_matrix2spk(txt1_in1)
    out_spk1 = txt_matrix2spk(txt1_out1)
    out_spk2 = txt_matrix2spk(txt1_out2)
    # in2_spk1 = txt_matrix2spk(txt2_in1)
    # out2_spk1 = txt_matrix2spk(txt2_out1)
    # out2_spk2 = txt_matrix2spk(txt2_out2)

    array_merge = merge_test(in_spk1, out_spk1, out_spk2)
    # array_merge = merge_test(in_spk1, in2_spk1, out_spk1, out_spk2, out2_spk1, out2_spk2)

    array2csv(array_merge, csvfile)




