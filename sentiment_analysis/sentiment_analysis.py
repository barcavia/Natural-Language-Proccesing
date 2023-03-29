import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.data_loader import *
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset, Dataset
import operator
import utils.data_loader as data_loader
import pickle
import tqdm
import matplotlib.pyplot as plt

# ------------------------------------------- Constants ----------------------------------------

SEQ_LEN = 52
W2V_EMBEDDING_DIM = 300

ONEHOT_AVERAGE = "onehot_average"
W2V_AVERAGE = "w2v_average"
W2V_SEQUENCE = "w2v_sequence"

TRAIN = "train"
VAL = "val"
TEST = "test"
NEGATED_POLARITY_SET = "neg"
RARE_SET = "rare"

N_EPOCHS = 20
BATCH_SIZE = 64
LR = 0.01
WEIGHT_DECAY = 0.001


# ------------------------------------------ Helper methods and classes --------------------------

def get_available_device():
    """
    Allows training on GPU if available. Can help with running things faster when a GPU with cuda is
    available but not a most...
    Given a device, one can use module.to(device)
    and criterion.to(device) so that all the computations will be done on the GPU.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_model(model, path, epoch, optimizer):
    """
    Utility function for saving checkpoint of a model, so training or evaluation can be executed later on.
    :param model: torch module representing the model
    :param optimizer: torch optimizer used for training the module
    :param path: path to save the checkpoint into
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()}, path)


def load(model, path, optimizer):
    """
    Loads the state (weights, paramters...) of a model which was saved with save_model
    :param model: should be the same model as the one which was saved in the path
    :param path: path to the saved checkpoint
    :param optimizer: should be the same optimizer as the one which was saved in the path
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch


# ------------------------------------------ Data utilities ----------------------------------------

def load_word2vec():
    """ Load Word2Vec Vectors
        Return:
            wv_from_bin: All 3 million embeddings, each lengh 300
    """
    import gensim.downloader as api
    wv_from_bin = api.load("word2vec-google-news-300")
    vocab = list(wv_from_bin.key_to_index.keys())
    print(wv_from_bin.key_to_index[vocab[0]])
    print("Loaded vocab size %i" % len(vocab))
    return wv_from_bin


def create_or_load_slim_w2v(words_list, cache_w2v=False):
    """
    returns word2vec dict only for words which appear in the dataset.
    :param words_list: list of words to use for the w2v dict
    :param cache_w2v: whether to save locally the small w2v dictionary
    :return: dictionary which maps the known words to their vectors
    """
    w2v_path = "w2v_dict.pkl"
    if not os.path.exists(w2v_path):
        full_w2v = load_word2vec()
        w2v_emb_dict = {k: full_w2v[k] for k in words_list if k in full_w2v}
        if cache_w2v:
            save_pickle(w2v_emb_dict, w2v_path)
    else:
        w2v_emb_dict = load_pickle(w2v_path)
    return w2v_emb_dict


def get_w2v_average(sent, word_to_vec, embedding_dim):
    """
    This method gets a sentence and returns the average word embedding of the words consisting
    the sentence.
    :param sent: the sentence object
    :param word_to_vec: a dictionary mapping words to their vector embeddings
    :param embedding_dim: the dimension of the word embedding vectors
    :return The average embedding vector as numpy ndarray.
    """
    w2v_average = np.zeros(embedding_dim)
    word_ind = 0
    for word in sent.text:
        if word in word_to_vec:
            w2v_average += word_to_vec[word]
            word_ind += 1
    # we averaging without the unknown words
    return w2v_average if word_ind == 0 else w2v_average / word_ind


def get_one_hot(size, ind):
    """
    this method returns a one-hot vector of the given size, where the 1 is placed in the ind entry.
    :param size: the size of the vector
    :param ind: the entry index to turn to 1
    :return: numpy ndarray which represents the one-hot vector
    """
    one_hot = np.zeros(size)
    one_hot[ind] = 1
    return one_hot


def average_one_hots(sent, word_to_ind):
    """
    this method gets a sentence, and a mapping between words to indices, and returns the average
    one-hot embedding of the tokens in the sentence.
    :param sent: a sentence object.
    :param word_to_ind: a mapping between words to indices
    :return:
    """
    embedding_size = len(word_to_ind)
    sentence_size = len(sent.text)
    avg_one_hot = np.zeros(embedding_size)
    for word in sent.text:
        avg_one_hot += get_one_hot(embedding_size, word_to_ind[word])
    return avg_one_hot / sentence_size


def get_word_to_ind(words_list):
    """
    this function gets a list of words, and returns a mapping between
    words to their index.
    :param words_list: a list of words
    :return: the dictionary mapping words to the index
    """
    word_to_ind = dict()
    word_ind = 0
    for word in words_list:
        if word not in word_to_ind:
            word_to_ind[word] = word_ind
            word_ind += 1
    return word_to_ind


def sentence_to_embedding(sent, word_to_vec, seq_len, embedding_dim=300):
    """
    this method gets a sentence and a word to vector mapping, and returns a list containing the
    words embeddings of the tokens in the sentence.
    :param sent: a sentence object
    :param word_to_vec: a word to vector mapping.
    :param seq_len: the fixed length for which the sentence will be mapped to.
    :param embedding_dim: the dimension of the w2v embedding
    :return: numpy ndarray of shape (seq_len, embedding_dim) with the representation of the sentence
    """
    result = []
    words_counter = 1
    for word in sent.text:
        word_vec = word_to_vec[word] if word in word_to_vec else np.zeros(embedding_dim)
        result.append(word_vec)
        if words_counter == seq_len:
            # if we got here, we have the first 52 word embeddings in our result
            return np.array(result)
        words_counter += 1
    # if we got here, the sentence length is smaller than 52, so we need to pad the rest of the result
    for _ in range(seq_len - len(result)):
        result.append(np.zeros(embedding_dim))
    return np.array(result)


class OnlineDataset(Dataset):
    """
    A pytorch dataset which generates model inputs on the fly from sentences of SentimentTreeBank
    """

    def __init__(self, sent_data, sent_func, sent_func_kwargs):
        """
        :param sent_data: list of sentences from SentimentTreeBank
        :param sent_func: Function which converts a sentence to an input datapoint
        :param sent_func_kwargs: fixed keyword arguments for the state_func
        """
        self.data = sent_data
        self.sent_func = sent_func
        self.sent_func_kwargs = sent_func_kwargs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sent = self.data[idx]
        sent_emb = self.sent_func(sent, **self.sent_func_kwargs)
        sent_label = sent.sentiment_class
        return sent_emb, sent_label


class DataManager():
    """
    Utility class for handling all data management task. Can be used to get iterators for training and
    evaluation.
    """

    def __init__(self, data_type=ONEHOT_AVERAGE, use_sub_phrases=True, dataset_path="stanfordSentimentTreebank",
                 batch_size=50,
                 embedding_dim=None):
        """
        builds the data manager used for training and evaluation.
        :param data_type: one of ONEHOT_AVERAGE, W2V_AVERAGE and W2V_SEQUENCE
        :param use_sub_phrases: if true, training data will include all sub-phrases plus the full sentences
        :param dataset_path: path to the dataset directory
        :param batch_size: number of examples per batch
        :param embedding_dim: relevant only for the W2V data types.
        """

        # load the dataset
        self.sentiment_dataset = data_loader.SentimentTreeBank(dataset_path, split_words=True)
        # map data splits to sentences lists
        self.sentences = {}
        if use_sub_phrases:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set_phrases()
        else:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set()

        self.sentences[VAL] = self.sentiment_dataset.get_validation_set()
        self.sentences[TEST] = self.sentiment_dataset.get_test_set()

        # getting the rare words, and negated polarity words
        self.sentences[RARE_SET] = [self.sentences[TEST][rare_ind] for rare_ind in
                                    get_rare_words_examples(self.sentences[TEST], self.sentiment_dataset)]
        self.sentences[NEGATED_POLARITY_SET] = [self.sentences[TEST][neg_ind] for neg_ind
                                                in get_negated_polarity_examples(self.sentences[TEST])]

        # map data splits to sentence input preperation functions
        words_list = list(self.sentiment_dataset.get_word_counts().keys())
        if data_type == ONEHOT_AVERAGE:
            self.sent_func = average_one_hots
            self.sent_func_kwargs = {"word_to_ind": get_word_to_ind(words_list)}
        elif data_type == W2V_SEQUENCE:
            self.sent_func = sentence_to_embedding

            self.sent_func_kwargs = {"seq_len": SEQ_LEN,
                                     "word_to_vec": create_or_load_slim_w2v(words_list),
                                     "embedding_dim": embedding_dim
                                     }
        elif data_type == W2V_AVERAGE:
            self.sent_func = get_w2v_average
            words_list = list(self.sentiment_dataset.get_word_counts().keys())
            self.sent_func_kwargs = {"word_to_vec": create_or_load_slim_w2v(words_list),
                                     "embedding_dim": embedding_dim
                                     }
        else:
            raise ValueError("invalid data_type: {}".format(data_type))
        # map data splits to torch datasets and iterators
        self.torch_datasets = {k: OnlineDataset(sentences, self.sent_func, self.sent_func_kwargs) for
                               k, sentences in self.sentences.items()}
        self.torch_iterators = {k: DataLoader(dataset, batch_size=batch_size, shuffle=k == TRAIN)
                                for k, dataset in self.torch_datasets.items()}

    def get_torch_iterator(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: torch batches iterator for this part of the datset
        """
        return self.torch_iterators[data_subset]

    def get_labels(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: numpy array with the labels of the requested part of the datset in the same order of the
        examples.
        """
        return np.array([sent.sentiment_class for sent in self.sentences[data_subset]])

    def get_input_shape(self):
        """
        :return: the shape of a single example from this dataset (only of x, ignoring y the label).
        """
        return self.torch_datasets[TRAIN][0][0].shape


# ------------------------------------ Models ----------------------------------------------------

class LSTM(nn.Module):
    """
    An LSTM for sentiment analysis with architecture as described in the exercise description.
    """

    def __init__(self, embedding_dim, hidden_dim, n_layers, dropout):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = n_layers
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=n_layers,
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.linear = nn.Linear(2 * hidden_dim, 1)

    def forward(self, text):
        nn_output, hidden_tuple = self.lstm(text)
        forwards_history = nn_output[:, -1, :self.hidden_dim]
        backwards_history = nn_output[:, 0, self.hidden_dim:]
        concatenated_hidden_states = torch.cat((forwards_history, backwards_history), 1)
        nn_output = self.linear(concatenated_hidden_states)
        return nn_output

    def predict(self, text):
        nn_out = self.forward(text)
        return nn.Sigmoid()(nn_out)


class LogLinear(nn.Module):
    """
    general class for the log-linear models for sentiment analysis.
    """

    def __init__(self, embedding_dim):
        super(LogLinear, self).__init__()
        self.linear_layer1 = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        return self.linear_layer1.forward(x)

    def predict(self, x):
        nn_out = self.linear_layer1(x)
        return nn.Sigmoid()(nn_out)


# ------------------------- training functions -------------


def binary_accuracy(preds, y):
    """
    This method returns tha accuracy of the predictions, relative to the labels.
    You can choose whether to use numpy arrays or tensors here.
    :param preds: a vector of predictions
    :param y: a vector of true labels
    :return: scalar value - (<number of accurate predictions> / <number of examples>)
    """
    rounded_predictions = preds
    rounded_predictions[rounded_predictions < 0.5] = 0
    rounded_predictions[rounded_predictions >= 0.5] = 1
    return (rounded_predictions == y).sum().item() / rounded_predictions.size(0)


def train_epoch(model, data_iterator, optimizer, criterion):
    """
    This method operates one epoch (pass over the whole train set) of training of the given model,
    and returns the accuracy and loss for this epoch
    :param model: the model we're currently training
    :param data_iterator: an iterator, iterating over the training data for the model.
    :param optimizer: the optimizer object for the training process.
    :param criterion: the criterion object for the training process.
    """
    model.train()
    nn_loss, accuracy = 0, 0
    for inputs, labels in data_iterator:
        # zero the gradients from the previous steps
        optimizer.zero_grad()
        nn_outputs = model(inputs.float())
        loss = criterion(nn_outputs.float(), labels.unsqueeze(1))
        # backpropagation
        loss.backward()
        # updating model weights based on the results
        optimizer.step()
        accuracy += binary_accuracy(model.predict(inputs.float()), labels.unsqueeze(1))
        nn_loss += loss.item()

    return accuracy / len(data_iterator), nn_loss / len(data_iterator)


def evaluate(model, data_iterator, criterion):
    """
    evaluate the model performance on the given data
    :param model: one of our models..
    :param data_iterator: torch data iterator for the relevant subset
    :param criterion: the loss criterion used for evaluation
    :return: tuple of (average loss over all examples, average accuracy over all examples)
    """
    model.eval()
    nn_loss, accuracy = 0, 0
    with torch.no_grad():
        for inputs, labels in data_iterator:
            nn_outputs = model(inputs.float())
            loss = criterion(nn_outputs.float(), labels.unsqueeze(1))
            nn_loss += loss.item()
            accuracy += binary_accuracy(model.predict(inputs.float()), labels.unsqueeze(1))

    return accuracy / len(data_iterator), nn_loss / len(data_iterator)


def get_predictions_for_data(model, data_iter):
    """

    This function should iterate over all batches of examples from data_iter and return all of the models
    predictions as a numpy ndarray or torch tensor (or list if you prefer). the prediction should be in the
    same order of the examples returned by data_iter.
    :param model: one of the models you implemented in the exercise
    :param data_iter: torch iterator as given by the DataManager
    :return:
    """
    predictions = []
    for inputs, labels in data_iter:
        predictions.append(model.predict(inputs.float()))

    return np.array(predictions)


def train_model(model, data_manager, n_epochs, lr, weight_decay=0.):
    """
    Runs the full training procedure for the given model. The optimization should be done using the Adam
    optimizer with all parameters but learning rate and weight decay set to default.
    :param model: module of one of the models implemented in the exercise
    :param data_manager: the DataManager object
    :param n_epochs: number of times to go over the whole training set
    :param lr: learning rate to be used for optimization
    :param weight_decay: parameter for l2 regularization
    """
    train_accuracy, train_loss = [], []
    eval_accuracy, eval_loss = [], []
    # getting the train iterator
    train_data_iterator = data_manager.get_torch_iterator(TRAIN)
    # getting the validation iterator
    eval_data_iterator = data_manager.get_torch_iterator(VAL)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    epoch = 1
    for _ in range(n_epochs):
        print("Starts training " + str(epoch) + " epoch")
        train_epoch_accuracy, train_epoch_loss = train_epoch(model, train_data_iterator, optimizer, criterion)
        print("finish training " + str(epoch) + " epoch")
        train_accuracy.append(train_epoch_accuracy)
        train_loss.append(train_epoch_loss)
        eval_epoch_accuracy, eval_epoch_loss = evaluate(model, eval_data_iterator, criterion)
        print("finish evaluating " + str(epoch) + " epoch")
        print()
        eval_accuracy.append(eval_epoch_accuracy)
        eval_loss.append(eval_epoch_loss)
        epoch += 1
    # evaluating the performance of the model on the test set, and the special sets
    test_accuracy, test_loss, neg_acc, rare_acc = evaluate_test_and_special_sets(model, data_manager, criterion)

    return train_accuracy, train_loss, eval_accuracy, eval_loss, test_accuracy, test_loss, neg_acc, rare_acc


def evaluate_test_and_special_sets(model, data_manager, criterion):
    """
    this function will calculate the test accuracy and the test loss,
    as well as the negated polarity and rare words accuracy
    :param model: represents the given model
    :param data_manager: represents the data set manager
    :param criterion: represents the criterion object
    :return: test accuracy, test loss, negated polarity accuracy, rare words accuracy
    """
    # evaluating the test set
    test_data_iterator = data_manager.get_torch_iterator(TEST)
    test_accuracy, test_loss = evaluate(model, test_data_iterator, criterion)

    # evaluating the negated polarity words and rare words
    # evaluating negated polarity words
    negated_data_iterator = data_manager.get_torch_iterator(NEGATED_POLARITY_SET)
    neg_acc, neg_loss = evaluate(model, negated_data_iterator, criterion)
    # evaluating rare words accuracy
    rare_data_iterator = data_manager.get_torch_iterator(RARE_SET)
    rare_acc, rare_loss = evaluate(model, rare_data_iterator, criterion)

    return test_accuracy, test_loss, neg_acc, rare_acc


def print_special_sets_results(test_accuracy, test_loss, neg_acc, rare_acc):
    """
    this function will print the test set accuracy and the test set loss, as well as the negated polarity
    and rare words accuracy
    :param test_accuracy: represents the test set accuracy
    :param test_loss: represents the test set loss
    :param neg_acc: represents the negated polarity words accuracy
    :param rare_acc: represents the rare words accuracy
    :return:
    """
    results_array = [("Test Set", "Accuracy", test_accuracy),
                     ("Test Set", "Loss", test_loss),
                     ("Negated Polarity Set", "Accuracy", neg_acc),
                     ("Rare Words Set", "Accuracy", rare_acc)]
    for set_name, result_name, result in results_array:
        print("The " + result_name + " for " + set_name + " is: " + str(result))


def train_log_linear_with_one_hot():
    """
    Here comes your code for training and evaluation of the log linear model with one hot representation.
    """
    data_manager = DataManager(batch_size=BATCH_SIZE)
    embedding_dim = data_manager.get_input_shape()[0]
    log_linear_model = LogLinear(embedding_dim)
    train_accuracy, train_loss, eval_accuracy, eval_loss, test_accuracy, test_loss, neg_acc, rare_acc = train_model(
        log_linear_model,
        data_manager,
        n_epochs=N_EPOCHS,
        lr=LR,
        weight_decay=WEIGHT_DECAY)

    # plotting the loss and accuracy graphs, for the train set and validation set
    plot_accuracy_and_loss(train_loss, eval_loss, name="Loss",
                           model_name="LogLinear with one hot")
    plot_accuracy_and_loss(train_accuracy, eval_accuracy, name="Accuracy",
                           model_name="LogLinear with one hot")
    print("---------- Results for Log Linear With One Hot Model ----------")
    print_special_sets_results(test_accuracy, test_loss, neg_acc, rare_acc)


def train_log_linear_with_w2v():
    """
    Here comes your code for training and evaluation of the log linear model with word embeddings
    representation.
    """
    data_manager = DataManager(data_type=W2V_AVERAGE, batch_size=BATCH_SIZE, embedding_dim=W2V_EMBEDDING_DIM)
    log_linear_w2v_model = LogLinear(W2V_EMBEDDING_DIM)
    train_accuracy, train_loss, eval_accuracy, eval_loss, test_accuracy, test_loss, neg_acc, rare_acc = train_model(
        log_linear_w2v_model,
        data_manager,
        n_epochs=N_EPOCHS,
        lr=LR,
        weight_decay=WEIGHT_DECAY)
    # plotting the loss and accuracy graphs, for the train set and validation set
    plot_accuracy_and_loss(train_loss, eval_loss, name="Loss",
                           model_name="LogLinear with w2v")
    plot_accuracy_and_loss(train_accuracy, eval_accuracy, name="Accuracy",
                           model_name="LogLinear with w2v")
    print("---------- Results for Log Linear With W2V ----------")
    print_special_sets_results(test_accuracy, test_loss, neg_acc, rare_acc)


def train_lstm_with_w2v():
    """
    Here comes your code for training and evaluation of the LSTM model.
    """
    data_manager = DataManager(data_type=W2V_SEQUENCE, batch_size=BATCH_SIZE, embedding_dim=W2V_EMBEDDING_DIM)
    lstm_model = LSTM(W2V_EMBEDDING_DIM, hidden_dim=100, n_layers=1, dropout=0.5)
    train_accuracy, train_loss, eval_accuracy, eval_loss, test_accuracy, test_loss, neg_acc, rare_acc = train_model(
        lstm_model,
        data_manager,
        n_epochs=4,
        lr=0.001,
        weight_decay=0.0001)
    # plotting the loss and accuracy graphs, for the train set and validation set
    plot_accuracy_and_loss(train_loss, eval_loss, name="Loss",
                           model_name="Bi Directional LSTM", n_epochs=4)
    plot_accuracy_and_loss(train_accuracy, eval_accuracy, name="Accuracy",
                           model_name="Bi Directional LSTM", n_epochs=4)
    print("---------- Results for Bi Directional LSTM ----------")
    print_special_sets_results(test_accuracy, test_loss, neg_acc, rare_acc)


def plot_accuracy_and_loss(y_axis1, y_axis2, name, model_name, n_epochs=N_EPOCHS):
    """
    this function will plot the loss or the accuracy of the different models, according to their performance
    on the train set and on the validation set.
    :param y_axis1: represents the loss or the accuracy of the train set.
    :param y_axis2: represents the loss or the accuracy of the validation set.
    :param name: represents the loss or the accuracy title.
    :param model_name: represents the model name.
    :param n_epochs: represents the x axis, this is the number of epochs.
    :return:
    """
    plt.plot(np.arange(n_epochs), y_axis1, color='blue')
    plt.plot(np.arange(n_epochs), y_axis2, color='orange')
    plt.xlabel("Epochs")
    plt.ylabel(name)
    plt.title(f"{model_name}:\n{name} as a function of epoch number")
    plt.legend(["Train", "Validation"])
    plt.show()


if __name__ == '__main__':
    train_log_linear_with_one_hot()
    train_log_linear_with_w2v()
    train_lstm_with_w2v()
