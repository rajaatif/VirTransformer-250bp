import numpy as np
import torch
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

class PreprocessLLMData:
    """
    A class to automate preprocessing for large language model input data.
    """

    def __init__(self, X,y, start_token=12, end_token=13):
        """
        Args:
            pickle_file_path (str): Path to the pickled data file.
            mask_shape (tuple): Shape of the mask (rows, sequence length).
            start_token (int): Token to prepend to the input sequences.
            end_token (int): Token to append to the input sequences.
        """
        #self.pickle_file_path = pickle_file_path
        self.X=X
        self.y= y

        self.start_token = start_token
        self.end_token = end_token

    def load_process_data(self):
        """
        Load the pickled data file and extract DNA label encodings and labels.
        Returns:
            inputs (np.array): DNA label-encoded sequences.
            y (np.array): Labels.
        """
        try:
            #with open(self.pickle_file_path, 'rb') as pickle_file:
             #   data = pickle.load(pickle_file)
            #X = np.array(self.X)
            y = np.array(self.y)
            def reads(seq):
                n = len(seq)
                reads = []
                for i in range(0, n, 250):
                    read = seq[i:i+250]
                    # Add zero padding if necessary
                    if len(read) < 250:
                        read = np.pad(read, (0, 250-len(read)), mode='constant')
                    reads.append(read)
                return reads
            X=reads(self.X)
            X = list(map(lambda t: list(t), X))
            #X=X[:-1]
            return X, y
        except FileNotFoundError:
            print(f"Pickle file not found at {self.pickle_file_path}")
            raise
        except KeyError as e:
            print(f"Missing key in the pickle file: {e}")
            raise

    def shuffle_data(self, inputs, y):
        """
        Shuffle inputs and labels using a random permutation.
        Args:
            inputs (np.array): DNA sequences.
            y (np.array): Labels.
        Returns:
            inputs (torch.Tensor): Shuffled input sequences as tensor.
            masks (torch.Tensor): Shuffled mask tensor.
            y (list): Shuffled labels.
        """
        # indices = torch.randperm(len(y))
        # inputs = torch.from_numpy(np.array(inputs)).type(torch.int8)[indices]
        # y = [y[i] for i in list(indices.numpy())]
        mask_shape = (len(inputs), 250)  


        # Generate masks of the appropriate shape
        masks = np.ones(mask_shape, dtype=np.int8)
        masks = torch.from_numpy(masks).type(torch.int8) #[indices]

        return inputs, masks, y

    def one_hot_encode_labels(self, y):
        """
        One-hot encode the labels using sklearn's OneHotEncoder.
        Args:
            y (list): Labels.
        Returns:
            y_ (np.array): One-hot encoded labels.
        """
        if(y=='Other Choredate Host'):
          y_=[1,0]
        elif(y=='Homo sapiens'):
          y_=[0,1]
        else:
          y_=[1,0]  
        return y_

    def add_tokens_and_adjust_masks(self, inputs, masks):
        """
        Add start and end tokens to the input sequences and adjust masks accordingly.
        Args:
            inputs (torch.Tensor): DNA sequences.
            masks (torch.Tensor): Mask tensor.
        Returns:
            inputs_ (np.array): Inputs with start and end tokens added.
            masks_ (np.array): Masks adjusted for the added tokens.
        """
        # Insert start tokens at the beginning and end tokens at the end of each sequence
        inputs_ = np.insert(inputs, 0, self.start_token, axis=1)
        inputs_ = np.insert(inputs_, inputs_.shape[1], self.end_token, axis=1)

        # Adjust masks for the added tokens
        masks_ = np.insert(masks, 0, 1, axis=1)  # Start token
        masks_ = np.insert(masks_, masks_.shape[1], 1, axis=1)  # End token

        return inputs_, masks_

    def preprocess(self):
        """
        Main method to preprocess the data for large language models.
        Returns:
            inputs_ (torch.Tensor): Final processed inputs.
            masks_ (torch.Tensor): Final processed masks.
            y_ (np.array): One-hot encoded labels.
        """
        # Step 1: Load data
        inputs, y = self.load_process_data()

        # Step 2: Shuffle data
        inputs, masks, y = self.shuffle_data(inputs, y)

        # Step 3: One-hot encode labels
        y_ = self.one_hot_encode_labels(y)
        y_=np.tile(y_,len(inputs)).reshape(-1,2)

        # Step 4: Add tokens and adjust masks
        inputs_, masks_ = self.add_tokens_and_adjust_masks(inputs, masks)
        test_labels = torch.tensor(y_).type(torch.FloatTensor)
        test_inputs = torch.tensor(inputs_).type(torch.LongTensor)
        test_masks = torch.tensor(masks_).type(torch.LongTensor)    
        test_data = TensorDataset(test_inputs, test_masks, test_labels)
        test_dataloader = DataLoader(test_data, batch_size=16)
        return torch.from_numpy(np.array(inputs_)), torch.from_numpy(np.array(masks_)), y_,test_dataloader
        


# Example usage
# if __name__ == "__main__":
#     # Define the path to the pickle file and mask shape
#     pickle_file_path = "/content/Viruses/numfeatures_spec.pkl"
#     mask_shape = (20003842, 250)  # Update this based on your data

#     # Initialize the PreprocessLLMData class
#     preprocessor = PreprocessLLMData(pickle_file_path, mask_shape)

#     # Preprocess the data
#     inputs_, masks_, y_ = preprocessor.preprocess()

#     # Save or use the preprocessed data
#     print(f"Processed Inputs Shape: {inputs_.shape}")
#     print(f"Processed Masks Shape: {masks_.shape}")
#     print(f"One-Hot Encoded Labels Shape: {y_.shape}")