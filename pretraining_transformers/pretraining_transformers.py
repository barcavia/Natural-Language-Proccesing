###################################################
# Exercise 5 - Natural Language Processing 67658  #
###################################################

import numpy as np
import matplotlib.pyplot as plt

# subset of categories that we will use
category_dict = {'comp.graphics': 'computer graphics',
                 'rec.sport.baseball': 'baseball',
                 'sci.electronics': 'science, electronics',
                 'talk.politics.guns': 'politics, guns'
                 }


def get_data(categories=None, portion=1.):
    """
    Get data for given categories and portion
    :param portion: portion of the data to use
    :return:
    """
    # get data
    from sklearn.datasets import fetch_20newsgroups
    data_train = fetch_20newsgroups(categories=categories, subset='train', remove=('headers', 'footers', 'quotes'),
                                    random_state=21)
    data_test = fetch_20newsgroups(categories=categories, subset='test', remove=('headers', 'footers', 'quotes'),
                                   random_state=21)

    # train
    train_len = int(portion * len(data_train.data))
    x_train = np.array(data_train.data[:train_len])
    y_train = data_train.target[:train_len]
    # remove empty entries
    non_empty = x_train != ""
    x_train, y_train = x_train[non_empty].tolist(), y_train[non_empty].tolist()

    # test
    x_test = np.array(data_test.data)
    y_test = data_test.target
    non_empty = np.array(x_test) != ""
    x_test, y_test = x_test[non_empty].tolist(), y_test[non_empty].tolist()
    return x_train, y_train, x_test, y_test


# Q1
def linear_classification(portion=1.):
    """
    Perform linear classification
    :param portion: portion of the data to use
    :return: classification accuracy
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    tf = TfidfVectorizer(stop_words='english', max_features=1000)
    x_train, y_train, x_test, y_test = get_data(categories=category_dict.keys(), portion=portion)
    # Add your code here
    x_train_transform = tf.fit_transform(x_train)
    x_test_transform = tf.transform(x_test)
    # creating a logistic regression model and fitting it with the training set
    logistic_reg_classifier = LogisticRegression().fit(x_train_transform, y_train)
    # predicting and evaluating the accuracy
    y_pred = logistic_reg_classifier.predict(x_test_transform)
    return accuracy_score(y_test, y_pred)


# Q2
def transformer_classification(portion=1.):
    """
    Transformer fine-tuning.
    :param portion: portion of the data to use
    :return: classification accuracy
    """
    import torch

    class Dataset(torch.utils.data.Dataset):
        """
        Dataset object
        """

        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    from datasets import load_metric
    metric = load_metric("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    from transformers import Trainer, TrainingArguments
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('distilroberta-base', cache_dir=None)
    model = AutoModelForSequenceClassification.from_pretrained('distilroberta-base',
                                                               cache_dir=None,
                                                               num_labels=len(category_dict),
                                                               problem_type="single_label_classification")

    x_train, y_train, x_test, y_test = get_data(categories=category_dict.keys(), portion=portion)

    # Add your code here
    # getting the train set
    x_train = tokenizer(x_train, truncation=True, padding=True)
    x_test = tokenizer(x_test, truncation=True, padding=True)
    # getting the test set
    train_set = Dataset(x_train, y_train)
    test_set = Dataset(x_test, y_test)
    # creating the training arguments, using the mentioned parameters
    training_args = TrainingArguments(output_dir="./Q2",
                                      learning_rate=5e-5,
                                      per_device_train_batch_size=16,
                                      per_device_eval_batch_size=16,
                                      num_train_epochs=5)
    # creating the trainer object
    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=train_set,
                      eval_dataset=test_set,
                      tokenizer=tokenizer,
                      compute_metrics=compute_metrics)
    # training the model
    trainer.train()
    # see https://huggingface.co/docs/transformers/v4.25.1/en/quicktour#trainer-a-pytorch-optimized-training-loop
    # Use the DataSet object defined above. No need for a DataCollator
    # evaluating the accuracy on the test set
    return trainer.evaluate(test_set)["eval_accuracy"]


# Q3
def zeroshot_classification(portion=1.):
    """
    Perform zero-shot classification
    :param portion: portion of the data to use
    :return: classification accuracy
    """
    from transformers import pipeline
    from sklearn.metrics import accuracy_score
    import torch
    categories = [category_dict[category_key] for category_key in list(category_dict.keys())]
    x_train, y_train, x_test, y_test = get_data(categories=list(category_dict.keys()), portion=portion)
    clf = pipeline("zero-shot-classification", model='cross-encoder/nli-MiniLM2-L6-H768')
    candidate_labels = list(category_dict.values())

    # Add your code here
    # see https://huggingface.co/docs/transformers/v4.25.1/en/main_classes/pipelines#transformers.ZeroShotClassificationPipeline
    y_pred = []
    # iterating over each sample in the test set and uses the clf pipeline to predict the category for that sample.
    for sample in x_test:
        # finding the index of the predicted category in the categories list and appends it to the y_pred list.
        clf_prediction = clf(sample, candidate_labels)["labels"][0]
        category_prediction = categories.index(clf_prediction)
        y_pred.append(category_prediction)
    # evaluating the accuracy on the test set
    return accuracy_score(y_test, y_pred)


def plot_accuracy_graph(model_name, accuracy, portions):
    """
    this function will plot the given model accuracy as a function of the data portion
    :param model_name: represents the given model
    :param accuracy: represents the accuracy array
    :param portions: represents the data portions array
    :return:
    """
    plt.plot(np.array(portions), np.array(accuracy), color='blue')
    plt.xlabel("Data Portions")
    plt.ylabel("Accuracy")
    plt.title(f"{model_name} Accuracy as a function of the data portion")
    plt.show()


if __name__ == "__main__":
    portions = [0.1, 0.5, 1.]
    # Q1
    logistic_accuracy = []
    print("Logistic regression results:")
    for p in portions:
        print(f"Portion: {p}")
        p_accuracy = linear_classification(p)
        print(p_accuracy)
        logistic_accuracy.append(p_accuracy)
    # plotting the accuracies we gained for the training data portions
    plot_accuracy_graph("Logistic Regression", logistic_accuracy, portions)

    # Q2
    transformer_accuracy = []
    print("\nFinetuning results:")
    for p in portions:
        print(f"Portion: {p}")
        p_accuracy = transformer_classification(portion=p)
        print(p_accuracy)
        transformer_accuracy.append(p_accuracy)
    # plotting the accuracies we gained for the training data portions
    plot_accuracy_graph("Transformer Classification", transformer_accuracy, portions)

    # Q3
    print("\nZero-shot result:")
    print(zeroshot_classification())
