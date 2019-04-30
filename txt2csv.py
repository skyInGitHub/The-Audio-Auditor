import sys
import os
import argparse


def matrix2spk(matrix, unique_list):
    matrix_spk = []     # Create an empty list for matrix_speaker
    spk_row = -1        # Row number for matrix_spk
    # multi_list = []

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


def unique(list1):
    # intilize a null list
    unique_list = []

    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
            # print list
    return unique_list


def txt2matrix(txtfile):

    with open(txtfile) as txt:
        row = 0
        nr = 0
        matrix = []
        for line in txt:
            word = line.split()
            spid = word[0].split("_")

            if (row % 2) == 0:
                word[1] = " ".join(word[2:(len(word) - 2)])

                matrix.append([])
                matrix[nr].append(spid[0])
                matrix[nr].append(word[1])
            else:
                matrix[nr].append(word[2])
                matrix[nr].append(word[4])
                nr += 1
            row += 1

    return matrix


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('txt1', type=str, help='path to file.txt')
    parser.add_argument('-csv', type=str, help='path of file.csv')
    arguments = parser.parse_args()
    return arguments


if __name__ == '__main__':
    args = get_arguments()

    txtfile = args.txt
    csvfile = '{}'.format(args.csv)

    if not os.path.isfile(txtfile):
        print("File path {} does not exist. Exiting...".format(txtfile))
        sys.exit()


