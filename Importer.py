import matplotlib.pyplot as plt
import os
import pickle
import math
import datetime
import argparse

import csv
import re
from enum import Enum
class Tally():
    def __init__(self):
        self.data = {}

    def record(self, e):
        if e in self.data.keys():
            self.data[e] += 1
        else:
            self.data[e] = 1

    def sorted_list(self):
        return sorted(self.data.items(),key=lambda x: x[1],reverse= True)


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





import numpy
import copy

def is_number(string, call=float):
   try:
      call(string)
      return True
   except ValueError:
      return False



class Mode(Enum):
   Unknown =0
   Header =1
   Tags =2
   Units =3
   Data =4
   Junk =5

known_tags = ["FREQUENCY SWEEP","USER","ID","DATE","TIME","CALIBRATION FILE","VOLTAGE","CYCLE"]
known_header = ["AC IMPEDANCE TEST"]
known_units = ["TIME (s)", "FREQ (Hz)","Z(Re) (Ohm)", "-Z(Im) (Ohm)","VOLTS (V)"]
retained_data_cols = {1:"FREQ", 2:"Re[Z]",3:"Im[Z]",4:"VOLTS"}
# for now we assume that Im[Z] was given with a negative sign

def parse_fra_file(filename):
   my_header = []
   my_tags = {}
   my_units = {}
   my_data = []
   my_junk = []
   with open(filename,'r') as myfile:
      header_content = ""
      mode = Mode.Unknown
      redo = False
      for i in range(10000):
         if redo:
            redo = False
         else:
            this_line = myfile.readline()

         if i < 10:
            header_content+= this_line

         if this_line == '':
            break


         if mode == Mode.Unknown:
            # test for header
            some_header = [this_line.startswith(head) for head in known_header]
            found_header = False
            for some_head in some_header:
               found_header = found_header or some_head

            if found_header:
               mode = Mode.Header
               redo = True
               continue
            
            # test for tags
            x = this_line.split(':')
            if len(x) > 1:
               if x[0] in known_tags:
                  mode = Mode.Tags
                  redo = True
                  continue

            # test for units
            some_units = [unit in this_line for unit in known_units]
            found_all_units = True
            for some_unit in some_units:
               found_all_units = found_all_units and some_unit

            if found_all_units:
               mode = Mode.Units
               redo = True
               continue

            # test for data
            x = this_line.split('\n')[0].split('\t')
            some_data = [is_number(x_i) for x_i in x]
            all_numbers = True
            for some_datum in some_data:
               all_numbers = all_numbers and some_datum

            if all_numbers:
               mode = Mode.Data
               redo = True
               continue

            mode = Mode.Junk
            redo = True
            continue



         if mode == Mode.Header:
            
            for head in known_header:
               if this_line.startswith(head):
                  my_header.append(head)

            mode= Mode.Unknown
            continue

         if mode == Mode.Tags:
            x = this_line.split('\n')[0].split(':', maxsplit=1)
            current_tag = known_tags.index(x[0])
            content = x[1]
            if known_tags[current_tag] in my_tags.keys():
               mode = Mode.Junk
               redo = True
               continue
            clean_content = content.replace('\t','').replace(' ','')
            if known_tags[current_tag] == 'DATE':
               if not all([is_number(d,call=int) for d in clean_content.split('/')]):
                  mode = Mode.Junk
                  redo = True
                  continue

               parsed_date = [int(d) for d in clean_content.split('/')] # month\day\year
               my_tags[known_tags[current_tag]] = copy.deepcopy(parsed_date)
            elif known_tags[current_tag] == 'TIME':
               if not all([is_number(d,call=int) for d in clean_content.split(':')]):
                  mode = Mode.Junk
                  redo = True
                  continue
               
               parsed_time = [int(d) for d in clean_content.split(':')] # hour:minute:second
               my_tags[known_tags[current_tag]] = copy.deepcopy(parsed_time)
            elif known_tags[current_tag] == 'VOLTAGE':
               if not is_number(clean_content):
                  mode = Mode.Junk
                  redo = True
                  continue
               
               parsed_voltage = float(clean_content)
               my_tags[known_tags[current_tag]] = copy.deepcopy(parsed_voltage)
            elif known_tags[current_tag] == 'CYCLE':
               if not is_number(clean_content,call=int):
                  mode = Mode.Junk
                  redo = True
                  continue
               
               parsed_cycle = int(clean_content)
               my_tags[known_tags[current_tag]] = copy.deepcopy(parsed_cycle)
                  
            else:
               my_tags[known_tags[current_tag]] = content.replace('\t','').replace(' ','')
            
            
            mode= Mode.Unknown
            continue

         if mode == Mode.Units:
            x = this_line.split('\n')[0].split('\t')
            x = [x_i.replace('\t','') for x_i in x]
            # map where the units are
            unit_positions = [known_units.index(x_i) for x_i in x]
            for retained_data_col in retained_data_cols.keys():
               my_units[retained_data_cols[retained_data_col]] = unit_positions[retained_data_col]

            mode = Mode.Unknown
            continue
         
         if mode == Mode.Data:
            x = this_line.split('\n')[0].split('\t')
            some_data = [float(x_i) for x_i in x]
            my_data.append( copy.deepcopy(some_data))

            mode = Mode.Unknown
            continue

         if mode == Mode.Junk:
            my_junk += copy.deepcopy([this_line])
            mode = Mode.Unknown
            continue
            
      my_data = numpy.array(my_data)
      my_data_shape = numpy.shape(my_data)
      my_split_data = {}
      if not 'Im[Z]' in my_units.keys():
         if len(my_units) > 0 or len(my_data) >0:
             print("my file:, ", filename)
             print("my units: ", my_units)
             print("my data: ", my_data)
         my_units = {'TIME':0, 'FREQ':1, 'Re[Z]':2, 'Im[Z]':3, 'VOLTS':4}


      if not my_data == [] and len(my_data_shape) > 1:


         for i in range(len(my_data)):
            my_data[i,my_units["Im[Z]"]] = -my_data[i,my_units["Im[Z]"]]


         for key in my_units.keys():
            if my_units[key] >= my_data_shape[1]:
               if len(my_data) >0:
                   print("my file:, ", filename)
                   print("my_data: ", my_data)
            else:
               my_split_data[key] = my_data[:, my_units[key]]
      else:
         if len(my_data) > 0:
             print("my file:, ", filename)
             print("my_data: ", my_data)

      parsed_data = {"header":copy.deepcopy(my_header), "tags":copy.deepcopy(my_tags), "data":copy.deepcopy(my_split_data), "junk":copy.deepcopy(my_junk)}

         
      return parsed_data



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--file_types', choices=['fra', 'eis'], default='fra')
    #parser.add_argument('--finetuned', type=bool, default=False)
    #parser.add_argument('--finetune', dest='finetuned', action='store_true')
    #parser.add_argument('--no-finetune', dest='finetuned', action='store_false')
    #parser.set_defaults(finetuned=True)

    #File paths
    parser.add_argument('--data_dir', default='RealData')
    #parser.add_argument('--fra_no_finetune_file', default="results_of_inverse_model.file")
    #parser.add_argument('--fra_finetune_file', default="results_fine_tuned_with_adam_{}.file")
    #parser.add_argument('--eis_no_finetune_file', default="results_of_inverse_model_eis.file")
    #parser.add_argument('--eis_finetune_file', default="results_eis_fine_tuned_with_adam_{}.file")
    #parser.add_argument('--steps', type=int, default=1000)

    args = parser.parse_args()




    path_to_spectra = os.path.join(".", args.data_dir)

    # tally the file extentions
    tally_extensions = Tally()
    for root, dirs, filenames in os.walk(path_to_spectra):
       for file in filenames:
          extension = file.split('.')[-1]
          tally_extensions.record(extension)


    print(tally_extensions.sorted_list())


    all_mpt_filenames = []

    all_fra_filenames = []
    for root, dirs, filenames in os.walk(path_to_spectra):
       for file in filenames:
          if file.endswith('.FRA'):
             all_fra_filenames.append(os.path.join(root, file))

          if file.endswith('.mpt') or file.endswith('.txt'):
              all_mpt_filenames.append(os.path.join(root, file))

    print('Number of fra files {}.'.format(len(all_fra_filenames)))



    print('Number of mpt files {}.'.format(len(all_mpt_filenames)))
    with open(os.path.join('.', 'Bad_File_Examples.txt'), 'r')  as f:
        bad_files = [x.split('\n')[0] for x in f.readlines()]



    print_whole_file = False
    unit_list = ['freq/Hz',
                 'Re(Z)/Ohm',
                 '-Im(Z)/Ohm',
                '|Z|/Ohm',
                 'Phase(Z)/deg',
                 'time/s',
                '<Ewe>/V']
    count = 0
    database_eis = {}
    all_datas = []
    data_length_tally = Tally()
    for repeat_index in range(2):
        for filename in all_mpt_filenames:
            clean_multiple_loops = False
            with open(filename, 'r') as f:
                data = {}
                for u in unit_list:
                    data[u] = []

                if print_whole_file:
                    print(f.read())
                else:
                    all_lines = f.readlines()
                    my_mode = MPT_Mode.FirstLine
                    for i in range(len(all_lines)):
                        new_line = all_lines[i]
                        if my_mode == MPT_Mode.FirstLine:
                            if 'EC-Lab ASCII FILE' in new_line:
                                my_mode = MPT_Mode.NbHeader
                                continue
                            elif '"EC-Lab","ASCII","FILE"\n' in new_line:
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
                                        print(repr(new_line))
                                        print(all_lines[:min(len(all_lines), 100)])
                                        break
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
                            if i == number_of_header_lines-1:
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
                                        my_mode=MPT_Mode.CollectData
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





                if my_mode == MPT_Mode.NbHeader:
                    print("didn't find number of header lines.")
                    print(all_lines[:min(len(all_lines), 100)])
                elif my_mode == MPT_Mode.CSVFormat:
                    with open(filename, newline='') as f:
                        all_lines= list(csv.reader(f))
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
                                            break
                                        else:
                                            my_mode = MPT_Mode.CollectData
                                            continue

                            elif my_mode == MPT_Mode.NbHeader:
                                if len(new_line) == 5 and new_line[0]=='Nb' and \
                                        new_line[1]=='header' and new_line[2]=='lines' and \
                                        new_line[3]==':':
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

                if my_mode == MPT_Mode.CollectData:

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


                    if not len(log_freq_) < 10:

                        if True:#clean_multiple_loops:
                            turning_points = [0]
                            for index in range(len(log_freq_) - 1):
                                if log_freq_[index] < log_freq_[index+1]:
                                    turning_points.append(index+1)

                            if len(turning_points) > 1 and repeat_index == 0:
                                continue

                            turning_points.append(len(log_freq_))
                            log_freq__ = log_freq_
                            re_z__ = re_z_
                            im_z__ = im_z_
                            for index in range(len(turning_points)-1):
                                log_freq_ = log_freq__[turning_points[index]:turning_points[index+1]]
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

                                        if im_z[i] > im_z[i-1]:
                                            break
                                        else:
                                            tails += 1

                                    if True:
                                        if len(turning_points) == 2:
                                            new_filename = filename
                                        else:
                                            new_filename = filename.split('.mpt')[0] + '_loop{}.mpt'.format(index+1)
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
                                            numpy.sort(numpy.sqrt((re_z[1:] - re_z[:-1]) ** 2 + (im_z[1:] - im_z[:-1]) ** 2))[
                                            -length:])

                                        test_2 =  mean_dev >= mean_mag

                                        mean_re_z_ = re_z - numpy.mean(re_z)
                                        mean_im_z_ = im_z - numpy.mean(im_z)
                                        mean_mag_ = numpy.mean(numpy.sqrt(mean_re_z_ ** 2 + mean_im_z_ ** 2))

                                        test_3 = 2.*mean_dev >= mean_mag_

                                        test = test_1 or test_2 or test_3

                                        if not test:
                                            record = {'original_spectrum': (log_freq, re_z, im_z),
                                                                          'freqs_with_negative_im_z': negs,
                                                                          'freqs_with_tails_im_z': tails,
                                                                          'actual_voltage': actual_voltage,
                                                                          'recognized_metadata': False}

                                            not_already_recorded = True
                                            for already_recorded in database_eis.keys():
                                                comp_record = database_eis[already_recorded]
                                                if (not record['freqs_with_negative_im_z'] == comp_record['freqs_with_negative_im_z'] or
                                                        not record['freqs_with_tails_im_z'] == comp_record[
                                                            'freqs_with_tails_im_z']):
                                                   continue

                                                if not len(record['original_spectrum'][0]) == len(comp_record['original_spectrum'][0]):
                                                    continue

                                                continue_q = False
                                                for index_i in range(len(log_freq)):
                                                    if ((record['original_spectrum'][0][index_i] - comp_record['original_spectrum'][0][index_i]) > 1e-10 or
                                                            ( record['original_spectrum'][1][index_i] -
                                                                comp_record['original_spectrum'][1][index_i]) > 1e-10 or
                                                            (record['original_spectrum'][2][index_i] -
                                                                comp_record['original_spectrum'][2][index_i]) > 1e-10) :
                                                       continue_q = True
                                                       break

                                                if continue_q:
                                                    continue
                                                else:
                                                    not_already_recorded = False
                                                    print('already recorded. was file {}'.format(already_recorded))
                                                    break

                                            if not_already_recorded:
                                                database_eis[new_filename] = record
                                                count += 1
                                                print('record added. was file {}'.format(new_filename))
                                            else:
                                                print('duplicate identified. was file {}'.format(new_filename))
                                                continue
                                        else:
                                            print('bad file: {}, t1:{}, t2:{}, t3:{}'.format( new_filename, test_1,test_2,test_3))
                                            continue


                                else:
                                    print('bad file: ', new_filename)
                                    continue




                    else:
                        print('bad file: ', filename)
                        continue

    print('number of properly processed eis files: {}'.format(count))

    with open(os.path.join(".", args.data_dir, "database_eis.file"), 'wb') as f:
        pickle.dump(database_eis, f, pickle.HIGHEST_PROTOCOL)




    database = {}



    count = 0
    empty_parse = 0
    print('Number of files {}.'.format(len(all_fra_filenames)))
    for filename in all_fra_filenames:
        dats = parse_fra_file(filename)

        if 'data' in dats.keys() and 'FREQ' in dats['data'].keys() and 'Re[Z]' in dats['data'].keys() and 'Im[Z]' in dats['data'].keys() :
           log_freq_ = numpy.log(2* math.pi * numpy.array(dats['data']['FREQ']))
           re_z_ = numpy.array(dats['data']['Re[Z]'])
           im_z_ = numpy.array(dats['data']['Im[Z]'])

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

           if not (any([x < 0.0 for x in re_z])):
               my_tags = dats['tags']
               if ('VOLTAGE' in my_tags.keys()) and ('CYCLE' in my_tags.keys()) and ('DATE' in my_tags.keys()) and ('TIME' in my_tags.keys()):
                   last_filename = filename.split('\\')[-1]
                   decomposed = last_filename.split('_')

                   if len(decomposed) >= 8:
                       valid = True
                       matchObj1 = re.match(r'(FRA)',
                                            decomposed[1])
                       if valid and not matchObj1:
                           valid = False

                       matchObj1 = re.match(r'0(\d{5,5})',
                                            decomposed[2])

                       matchObj2 = re.match(r'(\d{5,5})',
                                            decomposed[2])

                       matchObj3 = re.match(r'(\d{2,5}[A-Z])',
                                            decomposed[2])

                       if valid and matchObj1:
                           cell_id = matchObj1.group(1)
                       elif valid and matchObj2:
                           cell_id = matchObj2.group(1)
                       elif valid and matchObj3:
                           cell_id = matchObj3.group(1)
                       else:
                           valid = False

                       matchObj1 = re.match(r'(NEWARE)', decomposed[3])
                       matchObj2 = re.match(r'(Nw)', decomposed[3])
                       if matchObj1 or matchObj2:
                           cycle_index = 4
                       else:
                           cycle_index = 3

                       matchObj1 = re.match(r'c(\d{1,7})',
                                            decomposed[cycle_index])
                       if valid and matchObj1:
                           cycle_offset = int(matchObj1.group(1))
                       else:
                           valid = False

                       if valid:
                           pre_record = {'original_spectrum': (log_freq, re_z, im_z),
                                                 'freqs_with_negative_im_z': negs, 'cell_id': cell_id, 'cycle': (cycle_offset + my_tags['CYCLE']),
                                              'nominal_voltage': my_tags['VOLTAGE'], 'complete':True}

                       else:
                           print('bad file: ', filename)
                           continue
                   else:
                       matchObj1 = re.match(r'(.*)-EIS(\d{4,4}).FRA',
                                            last_filename)
                       if matchObj1:
                           pre_record = {'original_spectrum': (log_freq, re_z, im_z),
                                                 'freqs_with_negative_im_z': negs,'cell_id': matchObj1.group(1), 'cycle': my_tags['CYCLE'],
                                              'nominal_voltage': my_tags['VOLTAGE'], 'complete': False}

                       else:
                           print('bad file: ', filename)
                           continue



                   test_1 = numpy.max(numpy.abs(re_z) + numpy.abs(im_z)) > 1e6

                   mean_re_z = numpy.mean(re_z)
                   mean_im_z = numpy.mean(im_z)
                   mean_mag = math.sqrt(mean_re_z ** 2 + mean_im_z ** 2)
                   length = int((len(re_z) - 1) / 3)
                   mean_dev = numpy.mean(
                       numpy.sort(numpy.sqrt((re_z[1:] - re_z[:-1]) ** 2 + (im_z[1:] - im_z[:-1]) ** 2))[
                       -length:])

                   test_2 = mean_dev >= mean_mag

                   mean_re_z_ = re_z - numpy.mean(re_z)
                   mean_im_z_ = im_z - numpy.mean(im_z)
                   mean_mag_ = numpy.mean(numpy.sqrt(mean_re_z_ ** 2 + mean_im_z_ ** 2))

                   test_3 = 2. * mean_dev >= mean_mag_

                   test = test_1 or test_2 or test_3

                   if not test:
                       record = pre_record
                       not_already_recorded = True
                       for already_recorded in database.keys():
                           comp_record = database[already_recorded]
                           if (not record['freqs_with_negative_im_z'] == comp_record['freqs_with_negative_im_z'] ):
                               continue

                           if not len(record['original_spectrum'][0]) == len(comp_record['original_spectrum'][0]):
                               continue

                           continue_q = False
                           for index_i in range(len(log_freq)):
                               if ((record['original_spectrum'][0][index_i] - comp_record['original_spectrum'][0][
                                   index_i]) > 1e-10 or
                                       (record['original_spectrum'][1][index_i] -
                                        comp_record['original_spectrum'][1][index_i]) > 1e-10 or
                                       (record['original_spectrum'][2][index_i] -
                                        comp_record['original_spectrum'][2][index_i]) > 1e-10):
                                   continue_q = True
                                   break

                           if continue_q:
                               continue
                           else:
                               if record['cell_id'] == comp_record['cell_id']:

                                   matchObj = re.match(r'.*-EIS(\d{4,4})\.FRA',
                                            filename)
                                   matchObj2 = re.match(r'.*EIS(\d{4,4})\.FRA',
                                            already_recorded)

                                   if matchObj and matchObj2:
                                       if matchObj.group(1) == matchObj2.group(1):
                                           not_already_recorded = False
                                           print('found duplicate at {}'.format(already_recorded))
                                           break
                                       else:
                                           continue
                                   else:
                                       print('something went wrong.')
                                       break

                       if not_already_recorded:
                           database[filename] = record
                           count += 1
                           print('record added. was file {}'.format(filename))
                       else:
                           print('duplicate identified. was file {}'.format(filename))
                           continue
                   else:
                       print('bad file: {}, t1:{}, t2:{}, t3:{}'.format(filename, test_1, test_2, test_3))
                       continue

                   int_month = my_tags['DATE'][0]
                   int_day = my_tags['DATE'][1]
                   int_year = my_tags['DATE'][2]


                   int_hour = my_tags['TIME'][0]
                   int_minute = my_tags['TIME'][1]
                   int_second = my_tags['TIME'][2]


                   start_time = datetime.datetime(int_year, int_month, int_day, hour=int_hour, minute=int_minute,
                                                  second=int_second)

                   if 'data' in dats.keys() and 'VOLTS' in dats['data'].keys():
                       voltages = numpy.array(sorted(dats['data']['VOLTS']))
                       if len(voltages) > 30:
                           voltages=voltages[3:-3]

                       actual_voltage = numpy.mean(voltages)
                   else:
                       actual_voltage = database[filename]['nominal_voltage']

                   database[filename]['time'] = start_time
                   database[filename]['actual_voltage'] = actual_voltage

        else:
           empty_parse += 1


    print('successfully processed {} fra files.'.format(count))

    metadata_groups = {}

    for file_id in all_fra_filenames:
        if file_id in database.keys():

            meta = database[file_id]
            cell_id = meta['cell_id']
            if not cell_id in metadata_groups.keys():
                metadata_groups[cell_id]=[]
            metadata_groups[cell_id].append(file_id)


    for k in metadata_groups.keys():
        all_times = []
        for file_id in metadata_groups[k]:
            all_times.append(database[file_id]['time'])

        start_time = min(all_times)

        for file_id in metadata_groups[k]:
            database[file_id]['time'] = database[file_id]['time'] - start_time

        cycle_groups = {}

        for file_id in metadata_groups[k]:
            meta = database[file_id]
            cycle = meta['cycle']

            if not cycle in cycle_groups.keys():
                cycle_groups[cycle] = []
            cycle_groups[cycle].append({'file_id':file_id, 'time':meta['time'],'nominal_voltage':meta['nominal_voltage']})


        for cyc in cycle_groups.keys():
            cycle_groups[cyc] = sorted(cycle_groups[cyc], key=lambda x: x['time'])

            if len(cycle_groups[cyc]) == 1:
                database[cycle_groups[cyc][0]['file_id']]['charge'] = True
            else:
                for index in range(len(cycle_groups[cyc])):
                    if index == 0:
                        if cycle_groups[cyc][1]['nominal_voltage'] > cycle_groups[cyc][index]['nominal_voltage']:
                            database[cycle_groups[cyc][index]['file_id']]['charge'] = True
                        else:
                            database[cycle_groups[cyc][index]['file_id']]['charge'] = False

                    elif index == len(cycle_groups[cyc]) -1:
                        if cycle_groups[cyc][index]['nominal_voltage'] > cycle_groups[cyc][index-1]['nominal_voltage']:
                            database[cycle_groups[cyc][index]['file_id']]['charge'] = True
                        else:
                            database[cycle_groups[cyc][index]['file_id']]['charge'] = False

                    else:
                        if (cycle_groups[cyc][index]['nominal_voltage'] > cycle_groups[cyc][index - 1]['nominal_voltage']) and \
                            (cycle_groups[cyc][index+1]['nominal_voltage'] > cycle_groups[cyc][index]['nominal_voltage']):
                            database[cycle_groups[cyc][index]['file_id']]['charge'] = True
                        elif (cycle_groups[cyc][index]['nominal_voltage'] < cycle_groups[cyc][index - 1]['nominal_voltage']) and \
                             (cycle_groups[cyc][index+1]['nominal_voltage'] < cycle_groups[cyc][index]['nominal_voltage']):
                            database[cycle_groups[cyc][index]['file_id']]['charge'] = False

                        elif (cycle_groups[cyc][index]['nominal_voltage'] > cycle_groups[cyc][index - 1]['nominal_voltage']) and \
                             (cycle_groups[cyc][index+1]['nominal_voltage'] <= cycle_groups[cyc][index]['nominal_voltage']):
                            database[cycle_groups[cyc][index]['file_id']]['charge'] = True

                        elif (cycle_groups[cyc][index]['nominal_voltage'] < cycle_groups[cyc][index - 1]['nominal_voltage']) and \
                             (cycle_groups[cyc][index+1]['nominal_voltage'] >= cycle_groups[cyc][index]['nominal_voltage']):
                            database[cycle_groups[cyc][index]['file_id']]['charge'] = False

                        elif (cycle_groups[cyc][index]['nominal_voltage'] == cycle_groups[cyc][index - 1]['nominal_voltage']) and \
                             (cycle_groups[cyc][index+1]['nominal_voltage'] < cycle_groups[cyc][index]['nominal_voltage']):
                            database[cycle_groups[cyc][index]['file_id']]['charge'] = False

                        elif (cycle_groups[cyc][index]['nominal_voltage'] == cycle_groups[cyc][index - 1]['nominal_voltage']) and \
                             (cycle_groups[cyc][index+1]['nominal_voltage'] >= cycle_groups[cyc][index]['nominal_voltage']):
                            database[cycle_groups[cyc][index]['file_id']]['charge'] = True
                        else:
                            database[cycle_groups[cyc][index]['file_id']]['charge'] = True



    print('empty parses: {}.'.format(empty_parse))


    with open(os.path.join(".", args.data_dir, "database.file"), 'wb') as f:
        pickle.dump(database, f, pickle.HIGHEST_PROTOCOL)
