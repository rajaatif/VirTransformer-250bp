import sys
import time
import traceback
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
from tqdm.auto import tqdm
os.system("cls")

import warnings
warnings.filterwarnings("ignore")

class ConvertDNALabelEncoder(object):
    """
        convert dna sequence string csv file to dna label encoder csv file and viceverse
    """
    def __init__(self):
        pass

    @staticmethod
    def convert_dna_string_to_dna_labelencoder(dna_string_csv_path, seq_column,label_column):
        """
            convert dna sequence string csv file to dna label encoder csv file
        args:
            dna_string_csv_path (string): dna string csv file path
            dna_labelencoder_csv_path (string): dna label encoder csv file path
        returns:
            none
        """

        df_dna_string = pd.read_csv(filepath_or_buffer=dna_string_csv_path) #.iloc[:10,:]
        #df_dna_string.loc[df_dna_string['labels']!='Homo sapiens','labels'] ='Other Choredate Host'
        #df_dna_string['labels']=df_dna_string.loc[df_dna_string['labels']!='Homo sapiens','label'] #='Other Choredate Host'
        #print(len(df_dna_string))
        label_encoder = LabelEncoder()
        dna_string_list = []
        for row in tqdm(df_dna_string.itertuples()):
            #print(row[2])
            #dna_string_row = row[3] #.Sequence
            dna_string_row = getattr(row, seq_column)
            dna_string_row=dna_string_row.replace('S','').replace('W','').replace('Y','').replace('H','').replace('R','').replace('K','').replace('V','').replace('M','').replace('D','').replace('B','').replace('I','').replace('J','')
            dna_string_nparray = np.array(list(dna_string_row))
            sample=['A','T','C','G','N']
            label_encoder.fit(sample)

            dna_labelencoder_row = label_encoder.transform(dna_string_nparray)
            dna_string_list.append(dna_labelencoder_row.astype(np.int8))
        #df_dna_labelencoder = pd.DataFrame(dna_string_list)
        #df_dna_labelencoder.to_csv(path_or_buf=dna_labelencoder_csv_path, index=False, header=None)
        return dna_string_list, df_dna_string[label_column]

    @staticmethod
    def convert_dna_labelencoder_to_dna_string(dna_labelencoder_list):
    	"""
        	Convert DNA sequence label encoder CSV file to DNA string CSV file.
    	Args:
        	dna_labelencoder_csv_path (string): DNA label encoder CSV file path
        	dna_string_csv_path (string): DNA string CSV file path
    """
    	try:
        # Read the label-encoded DNA sequences
        		#df_dna_labelencoder = pd.read_csv(filepath_or_buffer=dna_labelencoder_csv_path, header=None)
        		dna_labelencoder_list = dna_labelencoder_list #df_dna_labelencoder.values.tolist()

        # Initialize the LabelEncoder and fit it to the DNA bases
       			label_encoder = LabelEncoder()
        		sample = ['A', 'T', 'C', 'G', 'N']  # DNA bases
        		label_encoder.fit(sample)

        # Use the inverse transform to decode label encodings back to DNA strings
        		dna_string_list = []
        		for encoded_sequence in dna_labelencoder_list:
            			encoded_array = np.array(encoded_sequence, dtype=np.int8)  # Ensure it's a NumPy array
            			decoded_sequence = label_encoder.inverse_transform(encoded_array)  # Decode back to DNA bases
            			dna_string_list.append(''.join(decoded_sequence))  # Join decoded bases into a string

        # Save the decoded DNA strings to a CSV file
        		df_dna_string = pd.DataFrame(dna_string_list)
        		#df_dna_string.to_csv(path_or_buf=dna_string_csv_path, index=False, header=None)

    	except Exception as e:
		        print("An error occurred. {}".format(ConvertDNALabelEncoder.get_exception_stack_trace()))

    @staticmethod
    def get_exception_stack_trace():
        """
            get exception stack trace
        args:
            none
        returns:
            exception_stack_trace (string): exception stack trace parameters
        """
        try:
            exception_type, exception_value, exception_traceback = sys.exc_info()
            file_name, line_number, procedure_name, line_code = traceback.extract_tb(exception_traceback)[-1]
            exception_stack_trace = ''.join('[Time Stamp]: ' + str(time.strftime('%d-%m-%Y %I:%M:%S %p')) + '' + '[File Name]: ' + str(file_name) + ' '
            + '[Procedure Name]: ' + str(procedure_name) + ' '
            + '[Error Message]: ' + str(exception_value) + ' '
            + '[Error Type]: ' + str(exception_type) + ' '
            + '[Line Number]: ' + str(line_number) + ' '
            + '[Line Code]: ' + str(line_code))
        except:
            print("An error occurred in {}".format("get_exception_stack_trace() function"))
        return exception_stack_trace

    @staticmethod
    def get_program_running(start_time):
        """
            calculate program running
        args:
            start_time (string): start time program runtime
        returns:
            none
        """
        try:
            end_time = time.time()
            diff_time = end_time - start_time
            result = time.strftime("%H:%M:%S", time.gmtime(diff_time))
            print("program runtime: {}".format(result))
        except:
            print("An error occurred. {}".format(ConvertDNALabelEncoder.get_exception_stack_trace()))