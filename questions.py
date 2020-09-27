import nltk
import sys,os,cv2,sys,string

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
    return inf
    #print(len(inf["python.txt"][0]))


def tokenize(document):
    #nltk.download()
    document = document.lower()
    tokens = nltk.word_tokenize(document)
    #delete stopwords
    stopwords = nltk.corpus.stopwords.words("english")
    #delete string punctuation
    punctuation = string.punctuation
    to_delete = []
    for t in tokens:
        if t in stopwords:
            to_delete.append(t)
        elif t in punctuation:
            to_delete.append(t)
    for td in to_delete:
        tokens.remove(td)
    tokens.sort()
    #print(tokens)
    return tokens


def compute_idfs(documents):
    inverse_doc = {}
    unique_doc =[]
    num_dic = len(documents)
    for k,v in documents.items():
        documents[k] = set(v)
        unique_doc.append(documents[k])
    for k,v in documents.items():
        for word in v:
            count = 0
            for ud in unique_doc:
                if word in ud:
                    count += 1
            if word not in inverse_doc.keys():
                inverse_doc[word] = num_dic/count
    return inverse_doc
    #print(len(inverse_doc))


def top_files(query, files, idfs, n):
    tf = []
    q_c_f = {}
    for k,v in files.items():
        for q in query:
            count = 0
            for word in v:
                if word == q:
                    count += 1
            if count != 0:
                if k not in q_c_f:
                    q_c_f[k] = count*idfs[q]
                else:
                    q_c_f[k] += count*idfs[q]
    q_c_f = {k: v for k, v in sorted(q_c_f.items(), key=lambda item: item[1],reverse=True)}
    #print(q_c_f)
    fs = []
    for k in q_c_f:
        fs.append(k)
    tf = fs[:n]
    #print (tf)
    return tf


def top_sentences(query, sentences, idfs, n):
    tf = []
    q_c_f = {}
    for k,v in sentences.items():
        for q in query:
            if q in v:
                if k not in q_c_f:
                    q_c_f[k] = idfs[q]
                else:
                    q_c_f[k] += idfs[q]
    q_c_f = {k: v for k, v in sorted(q_c_f.items(), key=lambda item: item[1],reverse=True)}
    #print(q_c_f)
    #check if we have more than one match
    highest = max(q_c_f.values())
    match_sentence = [k for k, v in q_c_f.items() if v == highest]
    term_density = {}
    if len(match_sentence) > 1:
        for sentence in match_sentence:
            intersec = set(sentence).intersection(query)
            term_density[sentence] = len(intersec)
        term_density = {k: v for k, v in sorted(term_density.items(), key=lambda item: item[1],reverse=True)}
        fs = []
        for k in q_c_f:
            fs.append(k)
        td = fs[:n]
        return td
    #print(match_sentence)
    fs = []
    for k in q_c_f:
        fs.append(k)
    sen = fs[:n]
    #print(tf)
    return sen


if __name__ == "__main__":
    main()
