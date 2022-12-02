from transformers import pipeline
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset
from datasets import load_dataset


def classify_and_subset(dataset, column, classifier, label, threshold, filename):
    '''
    classifies a text into being about the label or not about the label.
    text is contained in specified column of dataset.
    threshold is level of certainty required (0-1)
    returns a subsetted dataset
    '''
    subset_indices = []
    # Zero-shot classification requires all the possible labels listed out.
    labels = [f'about {label}', f'not about {label}']
    for i, article in tqdm(enumerate(classifier(KeyDataset(dataset, column), labels))):
    if article['labels'][0] == labels[0] and article['scores'][0]>threshold:
        subset_indices.append(i)
    subset = dataset.select(subset_indices)
    subset.save_to_disk(filename)


if __name__ == '__main__':
    dataset = 'cnn_dailymail'
    version = '3.0.0'
    dataset = load_dataset(dataset, version)
    classifier = pipeline("zero-shot-classification", device = 0, model = 'valhalla/distilbart-mnli-12-1')
    classify_and_subset(dataset['train'], 'article', classifier, 'sports', 0.6, 'train_subset')
