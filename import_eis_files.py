from enum import Enum


class MPT_Mode(Enum):
   Unknown =0
   FirstLine =1
   NbHeader =2
   JumpToData = 3
   Units =4
   Data =5
   Junk =6
   CSVFormat = 7
   CollectData = 8

print_whole_file = False
unit_list = ['freq/Hz',
             'Re(Z)/Ohm',
             '-Im(Z)/Ohm',
             '|Z|/Ohm',
             'Phase(Z)/deg',
             'time/s',
             '<Ewe>/V']

import re
import numpy
import math
def import_eis_file(file, format=None):

    content_b = file.read()
    content = content_b.decode("utf-8", "ignore")
    content = content.replace('\r\n', '\n')
    all_lines = content.split('\n')
    if format==None or format in ['MPT', 'DAHN_TXT']:

        data = {}
        for u in unit_list:
            data[u] = []

        my_mode = MPT_Mode.FirstLine
        for i in range(len(all_lines)):
            new_line = all_lines[i]
            if my_mode == MPT_Mode.FirstLine:
                if 'EC-Lab ASCII FILE' in new_line:
                    my_mode = MPT_Mode.NbHeader
                    continue
                elif '"EC-Lab","ASCII","FILE"' in new_line:
                    my_mode = MPT_Mode.CSVFormat
                    break
                else:
                    split_line = new_line.split('\t')
                    expected = True
                    if len(unit_list) > len(split_line):
                        expected = False
                    else:
                        for index in range(len(unit_list)):
                            if not (unit_list[index] in split_line[index] or '#NAME?' in split_line[index]):
                                expected = False
                                break

                        if not expected:

                            return {}, False
                        else:
                            my_mode = MPT_Mode.CollectData
                            number_of_header_lines = 1
                            continue




            elif my_mode == MPT_Mode.NbHeader:
                matchObj = re.match(r'Nb header lines : (\d{1,})',
                                    new_line)

                if matchObj:
                    number_of_header_lines = int(matchObj.group(1))
                    my_mode = MPT_Mode.JumpToData
                    continue
                else:
                    continue

            elif my_mode == MPT_Mode.JumpToData:
                if i == number_of_header_lines - 1:
                    split_line = new_line.split('\t')
                    expected = True
                    if len(unit_list) > len(split_line):
                        expected = False
                    else:
                        for index in range(len(unit_list)):
                            if not (unit_list[index] in split_line[index] or '#NAME?' in split_line[index]):
                                expected = False
                                break

                        if not expected:
                            print(repr(new_line))
                        else:
                            my_mode = MPT_Mode.CollectData
                            continue

                else:
                    matchObj = re.match(r'Number of loops : (\d{1,})',
                                        new_line)

                    if matchObj:
                        number_of_loops = int(matchObj.group(1))
                        if not number_of_loops == 1:
                            clean_multiple_loops = True
                        continue
                    else:
                        continue
            elif my_mode == MPT_Mode.CollectData:
                if i >= number_of_header_lines:
                    split_line = new_line.split('\t')
                    if len(unit_list) > len(split_line):
                        continue
                    else:
                        vals = []
                        for index in range(len(unit_list)):
                            try:
                                val = float(split_line[index])
                            except ValueError:
                                if re.match(r'(\d{1,2}):(\d{1,2})\.(\d{1,4})', split_line[index]):
                                    val = 0.
                                else:

                                    break
                            vals.append(val)
                        if len(vals) < len(unit_list):
                            continue
                        for index in range(len(unit_list)):
                            data[unit_list[index]].append(vals[index])
                        continue

                else:
                    continue

        if my_mode == MPT_Mode.NbHeader:
            print("didn't find number of header lines.")
            return {}, False
        elif my_mode == MPT_Mode.CSVFormat:
            all_lines  = [line.split(',') for line in content.split('\n')]
            my_mode = MPT_Mode.FirstLine
            for i in range(len(all_lines)):
                new_line = all_lines[i]
                if my_mode == MPT_Mode.FirstLine:
                    if ['EC-Lab', 'ASCII', 'FILE'] == new_line:
                        my_mode = MPT_Mode.NbHeader
                        continue
                    else:
                        split_line = new_line
                        expected = True
                        if len(unit_list) > len(split_line):
                            expected = False
                        else:
                            for index in range(len(unit_list)):
                                if not (unit_list[index] in split_line[index] or '#NAME?' in split_line[index]):
                                    expected = False
                                    break

                            if not expected:
                                print(repr(new_line))
                                print("    didn't find.")
                                print(all_lines[:min(len(all_lines), 100)])
                                return {}, False
                            else:
                                my_mode = MPT_Mode.CollectData
                                continue

                elif my_mode == MPT_Mode.NbHeader:
                    if len(new_line) == 5 and new_line[0] == 'Nb' and \
                            new_line[1] == 'header' and new_line[2] == 'lines' and \
                            new_line[3] == ':':
                        number_of_header_lines = int(new_line[4])
                        my_mode = MPT_Mode.JumpToData
                        continue
                    else:
                        continue
                elif my_mode == MPT_Mode.JumpToData:
                    if i == number_of_header_lines - 1:
                        split_line = new_line
                        expected = True
                        if len(unit_list) > len(split_line):
                            expected = False
                        else:
                            for index in range(len(unit_list)):
                                if not (unit_list[index] in split_line[index] or '#NAME?' in split_line[index]):
                                    expected = False
                                    break

                            if not expected:
                                print(repr(new_line))
                            else:
                                my_mode = MPT_Mode.CollectData
                                continue
                    else:
                        continue

                elif my_mode == MPT_Mode.CollectData:
                    if i >= number_of_header_lines:
                        split_line = new_line
                        if len(unit_list) > len(split_line):
                            print('new line', new_line)
                            continue
                        else:
                            vals = []
                            for index in range(len(unit_list)):
                                try:
                                    val = float(split_line[index])
                                except ValueError:
                                    if re.match(r'(\d{1,2}):(\d{1,2})\.(\d{1,4})', split_line[index]):
                                        val = 0.
                                    else:
                                        print("newline: ", new_line)
                                        break
                                vals.append(val)

                            if len(vals) < len(unit_list):
                                continue
                            for index in range(len(unit_list)):
                                data[unit_list[index]].append(vals[index])
                            continue

                    else:
                        continue


                else:
                    continue
    elif format in ['COLUMBIA_TXT']:
        data = {}
        for u in unit_list:
            data[u] = []
        true_unit_list = unit_list[:3]

        for i in range(len(all_lines)):
            new_line = all_lines[i]
            if new_line.startswith('%'):
                continue
            split_line = re.split(r'\s', new_line)
            vals = []
            for tok in split_line:
                if tok == '':
                    continue
                try:
                    val = float(tok)
                except ValueError:

                    break
                vals.append(val)
            if len(vals) != len(true_unit_list):
                continue
            for index in range(len(true_unit_list)):
                data[true_unit_list[index]].append(vals[index])
        my_mode = MPT_Mode.CollectData

    if my_mode == MPT_Mode.CollectData:
        # all_datas.append(copy.deepcopy(data))
        # count += 1
        '''
        ['freq/Hz',
         'Re(Z)/Ohm',
         '-Im(Z)/Ohm',
        '|Z|/Ohm',
         'Phase(Z)/deg',
         'time/s',
        '<Ewe>/V']

        '''

        log_freq_ = numpy.log(2 * math.pi * numpy.array(data['freq/Hz']))
        re_z_ = numpy.array(data['Re(Z)/Ohm'])
        im_z_ = -numpy.array(data['-Im(Z)/Ohm'])
        if format in ['COLUMBIA_TXT']:
            log_freq_ = numpy.flip(log_freq_, 0)
            re_z_ = numpy.flip(re_z_, 0)
            im_z_ = numpy.flip(im_z_, 0)

        if not len(log_freq_) < 10:

            if True:  # clean_multiple_loops:
                turning_points = [0]
                for index in range(len(log_freq_) - 1):
                    if log_freq_[index] < log_freq_[index + 1]:
                        turning_points.append(index + 1)

                if len(turning_points) > 1:

                    return {}, False

                turning_points.append(len(log_freq_))
                log_freq__ = log_freq_
                re_z__ = re_z_
                im_z__ = im_z_
                for index in range(len(turning_points) - 1):
                    log_freq_ = log_freq__[turning_points[index]:turning_points[index + 1]]
                    re_z_ = re_z__[turning_points[index]:turning_points[index + 1]]
                    im_z_ = im_z__[turning_points[index]:turning_points[index + 1]]

                    if not len(log_freq_) < 10:
                        if log_freq_[0] > log_freq_[-1]:
                            log_freq = numpy.flip(log_freq_, axis=0)
                            re_z = numpy.flip(re_z_, axis=0)
                            im_z = numpy.flip(im_z_, axis=0)
                        else:
                            log_freq = log_freq_
                            re_z = re_z_
                            im_z = im_z_

                        negs = 0
                        for i in reversed(range(len(log_freq))):
                            if im_z[i] < 0.0:
                                break
                            else:
                                negs += 1

                        tails = 0
                        for i in list(reversed(range(len(log_freq))))[:-1]:
                            '''
                            we are looking for a pattern where as w -> infinity, -Im increases (Im decreases)
                            '''

                            if im_z[i] > im_z[i - 1]:
                                break
                            else:
                                tails += 1

                        actual_voltage = None
                        if len(data['<Ewe>/V']) != 0:
                            voltages = numpy.array(sorted(numpy.array(data['<Ewe>/V'])))
                            if len(voltages) > 30:
                                voltages = voltages[3:-3]
                            actual_voltage = numpy.mean(voltages)

                        test_1 = numpy.max(numpy.abs(re_z) + numpy.abs(im_z)) > 1e6

                        mean_re_z = numpy.mean(re_z)
                        mean_im_z = numpy.mean(im_z)
                        mean_mag = math.sqrt(mean_re_z ** 2 + mean_im_z ** 2)
                        length = int((len(re_z) - 1) / 3)
                        mean_dev = numpy.mean(
                            numpy.sort(
                                numpy.sqrt((re_z[1:] - re_z[:-1]) ** 2 + (im_z[1:] - im_z[:-1]) ** 2))[
                            -length:])

                        test_2 = mean_dev >= mean_mag

                        mean_re_z_ = re_z - numpy.mean(re_z)
                        mean_im_z_ = im_z - numpy.mean(im_z)
                        mean_mag_ = numpy.mean(numpy.sqrt(mean_re_z_ ** 2 + mean_im_z_ ** 2))

                        test_3 = 2. * mean_dev >= mean_mag_

                        test = test_1 or test_2 or test_3


                        record = {'original_spectrum': (log_freq, re_z, im_z),
                                      'freqs_with_negative_im_z': negs,
                                      'freqs_with_tails_im_z': tails,
                                      'measured_voltage': actual_voltage,
                                      'bad_measurement':test
                                }


                        return record, True





    return {}, False







