import nltk
import sys,os,cv2,sys

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    inf = {}
    os.chdir(r"{}".format(directory))
    files = os.listdir()
    for file in files:
        f = open(file, encoding="utf8")
        data = f.read()
        inf[file] = data
    #print(len(inf["python.txt"]))
    #raise NotImplementedError


def tokenize(document):

    raise NotImplementedError


def compute_idfs(documents):

    raise NotImplementedError


def top_files(query, files, idfs, n):

    raise NotImplementedError


def top_sentences(query, sentences, idfs, n):

    raise NotImplementedError


if __name__ == "__main__":
    main()
