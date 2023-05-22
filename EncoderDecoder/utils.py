import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import wandb
import random
import re
from torch.utils.data import Dataset
import pandas as pd


device_gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

eng_alphabets = 'abcdefghijklmnopqrstuvwxyz'
eng_alphabets+=eng_alphabets.upper()
pad_char = '-PAD-'

eng_alpha2index = {pad_char: 0}
for index, alpha in enumerate(eng_alphabets):
    eng_alpha2index[alpha] = index+1

# print(eng_alpha2index)

hindi_alphabets = [chr(alpha) for alpha in range(2304, 2432)]
hindi_alphabet_size = len(hindi_alphabets)

hindi_alpha2index = {pad_char: 0}
for index, alpha in enumerate(hindi_alphabets):
    hindi_alpha2index[alpha] = index+1

tensor_dict = {key: torch.tensor(value).to(device_gpu) for key, value in eng_alpha2index.items()}

non_eng_letters_regex = re.compile('[^a-zA-Z ]')

def cleanEnglishVocab(line):
    """
    Cleans the English vocabulary by removing non-letter characters.
    
    Args:
        line (str): Input line of text in English.
    
    Returns:
        list: List of cleaned English words.
    """
    line = line.replace('-', ' ').replace(',', ' ').upper()
    line = non_eng_letters_regex.sub('', line)
    return line.split()


def cleanHindiVocab(line):
    """
    Cleans the Hindi vocabulary by removing non-letter characters.
    
    Args:
        line (str): Input line of text in Hindi.
    
    Returns:
        list: List of cleaned Hindi words.
    """
    line = line.replace('-', ' ').replace(',', ' ')
    cleaned_line = ''
    for char in line:
        if char in hindi_alpha2index or char == ' ':
            cleaned_line += char
    return cleaned_line.split()

class TransliterationDataLoader(Dataset):
    """
    Dataset class for loading and processing transliteration data.
    """

    def __init__(self, filename):
        """
        Initializes the TransliterationDataLoader object.

        Args:
            filename (str): Path to the CSV file containing the transliteration data.
        """
        self.eng_words, self.hindi_words = self.readCsvDataset(filename, cleanHindiVocab)
        self.shuffle_indices = list(range(len(self.eng_words)))
        random.shuffle(self.shuffle_indices)
        self.shuffle_start_index = 0

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.eng_words)

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset based on the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: Tuple containing the English word and Hindi word.
        """
        return self.eng_words[idx], self.hindi_words[idx]

    def readCsvDataset(self, filename, lang_vocab_cleaner):
        """
        Reads the CSV dataset file and cleans the English and Hindi vocabulary.

        Args:
            filename (str): Path to the CSV file.
            lang_vocab_cleaner (function): Function to clean the Hindi vocabulary.

        Returns:
            tuple: Tuple containing the cleaned English words and cleaned Hindi words.
        """
        lang1_words = []
        lang2_words = []

        df = pd.read_csv(filename)

        for index, row in df.iterrows():
            lang1_word = cleanEnglishVocab(row[0])
            lang2_word = lang_vocab_cleaner(row[1])

            # Skip noisy data
            if len(lang1_word) != len(lang2_word):
                print('Skipping: ', row[0], ' - ', row[1])
                continue

            lang1_words.extend(lang1_word)
            lang2_words.extend(lang2_word)

        return lang1_words, lang2_words

    def get_random_sample(self):
        """
        Retrieves a random sample from the dataset.

        Returns:
            tuple: Tuple containing the English word and Hindi word.
        """
        return self.__getitem__(np.random.randint(len(self.eng_words)))

    def get_batch_from_array(self, batch_size, array):
        """
        Retrieves a batch of samples from the given array.

        Args:
            batch_size (int): Size of the batch to retrieve.
            array (list): List of samples.

        Returns:
            list: Batch of samples.
        """
        end = self.shuffle_start_index + batch_size
        batch = []
        if end >= len(self.eng_words):
            batch = [array[i] for i in self.shuffle_indices[0:end%len(self.eng_words)]]
            end = len(self.eng_words)
        return batch + [array[i] for i in self.shuffle_indices[self.shuffle_start_index : end]]

    def get_batch(self, batch_size, postprocess=True):
        """
        Retrieves a batch of samples from the dataset.

        Args:
            batch_size (int): Size of the batch to retrieve.
            postprocess (bool): Flag indicating whether to perform post-processing.

        Returns:
            tuple: Tuple containing the batch of English words and batch of Hindi words.
        """
        eng_batch = self.get_batch_from_array(batch_size, self.eng_words)
        hindi_batch = self.get_batch_from_array(batch_size, self.hindi_words)
        self.shuffle_start_index += batch_size + 1

        # Reshuffle if 1 epoch is complete
        if self.shuffle_start_index >= len(self.eng_words):
            random.shuffle(self.shuffle_indices)
            self.shuffle_start_index = 0

        return eng_batch, hindi_batch

    
def word_rep(word, letter2index, device='cpu'):
    """
    Converts a word into a one-hot representation.

    Args:
        word (str): Word to be converted.
        letter2index (dict): Mapping of letters to indices.
        device (str): Device to store the tensor on.

    Returns:
        torch.Tensor: One-hot representation of the word.
    """
    rep = torch.zeros(len(word) + 1, 1, len(letter2index)).to(device)
    for letter_index, letter in enumerate(word):
        pos = letter2index[letter]
        rep[letter_index][0][pos] = 1
    pad_pos = letter2index[pad_char]
    rep[letter_index + 1][0][pad_pos] = 1
    return rep


def gt_rep(word, letter2index, device='cpu'):
    """
    Converts a ground truth word into a representation.

    Args:
        word (str): Ground truth word to be converted.
        letter2index (dict): Mapping of letters to indices.
        device (str): Device to store the tensor on.

    Returns:
        torch.Tensor: Representation of the ground truth word.
    """
    gt_rep = torch.zeros([len(word) + 1, 1], dtype=torch.long).to(device)
    for letter_index, letter in enumerate(word):
        pos = letter2index[letter]
        gt_rep[letter_index][0] = pos
    gt_rep[letter_index + 1][0] = letter2index[pad_char]
    return gt_rep


def infer(net, eng_word, shape, device='cpu'):
    """
    Performs inference on the network given an English word.

    Args:
        net (nn.Module): Network model.
        eng_word (str): English word for inference.
        shape: Shape parameter (not specified in the code).
        device (str): Device to perform inference on.

    Returns:
        torch.Tensor: Outputs of the network.
    """
    net.to(device)
    input_ = word_rep(eng_word, tensor_dict, device)
    outputs = net(input_, shape, device)
    return outputs


MAX_OUTPUT_CHARS = 70

