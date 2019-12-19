import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import copy
import functools

nltk.download('wordnet')


def lemmatize(qs):
    '''
    :param qs: List of NL questions
    :return:
    '''

    lem = WordNetLemmatizer()
    queries = []
    for q in qs:
        processed = q.lower()
        processed = re.sub('[?]', '', processed) # Can I just remove last char?
        tokens = processed.split()
        tokens = [token for token in tokens if token not in stopwords.words('english')]
        lem_tokens = [lem.lemmatize(token) for token in tokens]
        processed = ' '.join(lem_tokens)
        queries.append(processed)
    return queries


def lemmatize_w_qwords(qs):
    '''
    :param qs: List of NL questions
    :return:
    '''
    swords = stopwords.words('english')
    swords = set(swords)
    qwords = {'what', 'which', 'who', 'whom', 'when', 'where', 'why', 'how'}
    swords = swords - qwords

    lem = WordNetLemmatizer()
    queries = []
    for q in qs:
        processed = q.lower()
        processed = re.sub('[?]', '', processed)  # Can I just remove last char?
        tokens = processed.split()
        tokens = [token for token in tokens if token not in swords]
        lem_tokens = [lem.lemmatize(token) for token in tokens]
        processed = ' '.join(lem_tokens)
        queries.append(processed)
    return queries


def lemmatize_single(q):

    lem = WordNetLemmatizer()
    processed = q.lower()
    processed = re.sub('[?]', '', processed)  # Can I just remove last char?
    tokens = processed.split()
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    lem_tokens = [lem.lemmatize(token) for token in tokens]
    processed = ' '.join(lem_tokens)
    return processed


def lemmatize_single_w_qwords(q):
    swords = stopwords.words('english')
    swords = set(swords)
    qwords = {'what', 'which', 'who', 'whom', 'when', 'where', 'why', 'how'}
    swords = swords - qwords

    lem = WordNetLemmatizer()
    processed = q.lower()
    processed = re.sub('[?]', '', processed)  # Can I just remove last char?
    tokens = processed.split()
    tokens = [token for token in tokens if token not in swords]
    lem_tokens = [lem.lemmatize(token) for token in tokens]
    processed = ' '.join(lem_tokens)
    return processed


def lemmatize_google(qs):
    '''
    :param qs: List of NL questions
    :return:
    '''

    lem = WordNetLemmatizer()
    queries = []
    for q in qs:
        tokens = q.split()
        tokens = [token for token in tokens if token not in stopwords.words('english')]
        lem_tokens = [lem.lemmatize(token) for token in tokens]
        processed = ' '.join(lem_tokens)
        queries.append(processed)
    return queries


def output_google_queries():
    with open('v1.0-simplified_simplified-nq-train.jsonl', 'r') as json_file:
        json_list = list(json_file)

    qs = []
    for json_str in json_list:
        result = json.loads(json_str)
        qs.append(result['question_text'])
    queries = lemmatize_google(qs)
    both = zip(queries, qs)

    with open('google_queries.txt', 'w') as text_file:
        for query, q in both:
            text_file.write(query + '\n' + q + '\n\n')


def output_squad_queries():
    json_file = open('train-v2.0.json')
    json_str = json_file.read()
    json_data = json.loads(json_str)
    qs = [q['question'] for i in json_data['data'] for d in i['paragraphs'] for q in d['qas']]

    '''
    # Check if all end in ?. At least one does not.
    qs = [q['question'][-1] == '?' for i in json_data['data'] for d in i['paragraphs'] for q in d['qas']]
    print(all(qs))
    '''

    queries = lemmatize(qs)
    #queries = lemmatize_w_qwords(qs)

    both = zip(queries, qs)




    with open('squad_queries.txt', 'w') as text_file:

        for query, q in both:
            text_file.write(query + '\n' + q + '\n\n')

def output_squad_queries_qwords():
    json_file = open('train-v2.0.json')
    json_str = json_file.read()
    json_data = json.loads(json_str)
    qs = [q['question'] for i in json_data['data'] for d in i['paragraphs'] for q in d['qas']]

    '''
    # Check if all end in ?. At least one does not.
    qs = [q['question'][-1] == '?' for i in json_data['data'] for d in i['paragraphs'] for q in d['qas']]
    print(all(qs))
    '''

    queries = lemmatize(qs)
    queries2 = lemmatize_w_qwords(qs)

    both = zip(queries2, queries, qs)




    with open('squad_queries_3.txt', 'w') as text_file:

        for query2, query, q in both:
            text_file.write(query2 + '\n' + query + '\n' + q + '\n\n')

def output_squad_queries_somin():
    json_file = open('train-v2.0.json')
    json_str = json_file.read()
    json_data = json.loads(json_str)
    qs = [q['question'] for i in json_data['data'] for d in i['paragraphs'] for q in d['qas']]
    processed = [re.sub('[?]', '', q) for q in qs]
    '''
    # Check if all end in ?. At least one does not.
    qs = [q['question'][-1] == '?' for i in json_data['data'] for d in i['paragraphs'] for q in d['qas']]
    print(all(qs))
    '''



    queries = lemmatize(processed)
    both = zip(queries, processed)

    with open('train_somin_2.txt', 'w') as text_file:

        for query, q in both:
            text_file.write(query + '.\t' + q + '.\n')


    json_file = open('dev-v2.0.json')
    json_str = json_file.read()
    json_data = json.loads(json_str)
    qs = [q['question'] for i in json_data['data'] for d in i['paragraphs'] for q in d['qas']]

    '''
    # Check if all end in ?. At least one does not.
    qs = [q['question'][-1] == '?' for i in json_data['data'] for d in i['paragraphs'] for q in d['qas']]
    print(all(qs))
    '''

    queries = lemmatize(processed)
    both = zip(queries, processed)

    with open('dev_somin_2.txt', 'w') as text_file:

        for query, q in both:
            text_file.write(query + '.\t' + q + '.\n')

def output_squad_queries_original():
    json_file = open('train-v2.0.json')
    json_str = json_file.read()
    json_data = json.loads(json_str)
    qs = [q['question'] for i in json_data['data'] for d in i['paragraphs'] for q in d['qas']]

    '''
    # Check if all end in ?. At least one does not.
    qs = [q['question'][-1] == '?' for i in json_data['data'] for d in i['paragraphs'] for q in d['qas']]
    print(all(qs))
    '''



    queries = lemmatize(qs)
    both = zip(queries, qs)

    with open('queries_train.txt', 'w') as text_file:
        with open('questions_train.txt', 'w') as text_file2:
            for query, q in both:
                text_file.write(query + '\n')
                text_file2.write(q + '\n')

    json_file = open('dev-v2.0.json')
    json_str = json_file.read()
    json_data = json.loads(json_str)
    qs = [q['question'] for i in json_data['data'] for d in i['paragraphs'] for q in d['qas']]

    '''
    # Check if all end in ?. At least one does not.
    qs = [q['question'][-1] == '?' for i in json_data['data'] for d in i['paragraphs'] for q in d['qas']]
    print(all(qs))
    '''

    queries = lemmatize(qs)
    both = zip(queries, qs)

    with open('queries_dev.txt', 'w') as text_file:
        with open('questions_dev.txt', 'w') as text_file2:
            for query, q in both:
                text_file.write(query + '\n')
                text_file2.write(q + '\n')


def output_squad_queries_nparrays():
    json_file = open('train-v2.0.json')
    json_str = json_file.read()
    json_data = json.loads(json_str)
    qs = [q['question'] for i in json_data['data'] for d in i['paragraphs'] for q in d['qas']]

    '''
    # Check if all end in ?. At least one does not.
    qs = [q['question'][-1] == '?' for i in json_data['data'] for d in i['paragraphs'] for q in d['qas']]
    print(all(qs))
    '''

    queries = lemmatize(qs)
    both = zip(queries, qs)

    big_array = []

    for query, q in both:
        big_array.append([query, q])

    big_array = np.array(big_array)
    print(big_array.shape)

    #big_array = np.savetxt('keras_input_train.txt', big_array, fmt="%s", delimiter='/t')
    big_array = np.save('keras_input_train.npy', big_array)

    # json_file = open('dev-v2.0.json')
    # json_str = json_file.read()
    # json_data = json.loads(json_str)
    # qs = [q['question'] for i in json_data['data'] for d in i['paragraphs'] for q in d['qas']]
    #
    # '''
    # # Check if all end in ?. At least one does not.
    # qs = [q['question'][-1] == '?' for i in json_data['data'] for d in i['paragraphs'] for q in d['qas']]
    # print(all(qs))
    # '''
    #
    # queries = lemmatize(qs)
    # both = zip(queries, qs)
    #
    # with open('queries_dev.txt', 'w') as text_file:
    #     with open('questions_dev.txt', 'w') as text_file2:
    #         for query, q in both:
    #             text_file.write(query + '\n')
    #             text_file2.write(q + '\n')


def squad_stats():
    json_file = open('train-v2.0.json')
    json_str = json_file.read()
    json_data = json.loads(json_str)
    qs = [q['question'] for i in json_data['data'] for d in i['paragraphs'] for q in d['qas']]

    json_file_dev = open('dev-v2.0.json')
    json_str_dev = json_file_dev.read()
    json_data_dev = json.loads(json_str_dev)
    qs_dev = [q['question'] for i in json_data_dev['data'] for d in i['paragraphs'] for q in d['qas']]


    print('number of docs: ', len(json_data['data']) + len(json_data_dev['data']))

    # for i in json_data['data']:
    #     for d in i['paragraphs']:
    #         len(d['qas'])

    num_qs_per_para = np.array([len(d['qas']) for i in json_data['data'] for d in i['paragraphs']])
    print('qs per para, train: ', np.mean(num_qs_per_para))

    # Number of questions
    print(len(qs)) # 130319.

    # Number of dev questions
    print(len(qs_dev))

    qs.extend(qs_dev)

    print('train + dev: ', len(qs))

    # Number of unanswerable questions.
    unanswer_qs = [q['question'] for i in json_data['data'] for d in i['paragraphs'] for q in d['qas'] if q['is_impossible']]
    print(len(unanswer_qs)) # 43498

    unanswer_qs_dev = [q['question'] for i in json_data_dev['data'] for d in i['paragraphs'] for q in d['qas'] if
                   q['is_impossible']]
    print(len(unanswer_qs_dev))

    print('train + dev unanswerable: ', len(unanswer_qs)+len(unanswer_qs_dev))

    #context_lens = np.array([len(d['context'].split()) for i in json_data['data'] for d in i['paragraphs']])
    context_lens = [len(d['context'].split()) for i in json_data['data'] for d in i['paragraphs']]

    context_lens.extend([len(d['context'].split()) for i in json_data_dev['data'] for d in i['paragraphs']])
    context_lens = np.array(context_lens)


    avg_len_context = np.mean(context_lens)
    max_len_context = np.max(context_lens)
    min_len_context = np.min(context_lens)
    print('avg context len: ', avg_len_context) # 116.58550039401104
    print('max context len: ', max_len_context) # 653
    print('min context len: ', min_len_context) # 20

    len_qs = np.array([len(q.split()) for q in qs])
    avg_len_qs = np.mean(len_qs)
    max_len_qs = np.max(len_qs)
    min_len_qs = np.min(len_qs)

    queries = lemmatize(qs)

    len_queries = np.array([len(q.split()) for q in queries])
    avg_len_queries = np.mean(len_queries)
    max_len_queries = np.max(len_queries)
    min_len_queries = np.min(len_queries)

    print('Questions:')
    print('avg len: ', avg_len_qs) # 9.893822082735442
    print('max len: ', max_len_qs) # 40
    print('min len: ', min_len_qs) # 1

    print('Queries:')
    print('avg len: ', avg_len_queries) # 5.3351545054827
    print('max len: ', max_len_queries) # 31
    print('min len: ', min_len_queries) # 0


def google_stats():
    with open('v1.0-simplified_simplified-nq-train.jsonl', 'r') as json_file:
        json_list = list(json_file)

    qs = []
    for json_str in json_list:
        result = json.loads(json_str)
        qs.append(result['question_text'])
        #print(result['question_text'])
        #print(result['document_text'])


    print(len(qs))

    len_qs = np.array([len(q.split()) for q in qs])
    avg_len_qs = np.mean(len_qs)
    max_len_qs = np.max(len_qs)
    min_len_qs = np.min(len_qs)

    queries = lemmatize(qs)

    len_queries = np.array([len(q.split()) for q in queries])
    avg_len_queries = np.mean(len_queries)
    max_len_queries = np.max(len_queries)
    min_len_queries = np.min(len_queries)

    print('Questions:')
    print('avg len: ', avg_len_qs)  # 9.893822082735442
    print('max len: ', max_len_qs)  # 40
    print('min len: ', min_len_qs)  # 1

    print('Queries:')
    print('avg len: ', avg_len_queries)  # 5.3351545054827
    print('max len: ', max_len_queries)  # 31
    print('min len: ', min_len_queries)  # 0


def hotpot_stats():
    json_file = open('hotpot_train_v1.1.json')
    json_str = json_file.read()
    json_data = json.loads(json_str)
    qs = [item['question'] for item in json_data]

    json_file_dev = open('hotpot_dev_distractor_v1.json')
    json_str_dev = json_file_dev.read()
    json_data_dev = json.loads(json_str_dev)
    qs_dev = [item['question'] for item in json_data_dev]
    qs.extend(qs_dev)


    context_lens = []

    for item in json_data:
        context_lens.append(np.sum(np.array([functools.reduce(lambda x, y: x + len(y.split()), para[1], 0) for para in item['context']])))
    for item in json_data_dev:
        context_lens.append(np.sum(
            np.array([functools.reduce(lambda x, y: x + len(y.split()), para[1], 0) for para in item['context']])))

    context_lens = np.array(context_lens)


    # Number of questions
    print(len(qs))  # 90447
    print(len(qs_dev))
    print('Train + dev: ', len(qs)+len(qs_dev))

    avg_len_context = np.mean(context_lens)
    max_len_context = np.max(context_lens)
    min_len_context = np.min(context_lens)
    print('avg context len: ', avg_len_context)  #
    print('max context len: ', max_len_context)  #
    print('min context len: ', min_len_context)  #

    len_qs = np.array([len(q.split()) for q in qs])
    avg_len_qs = np.mean(len_qs)
    max_len_qs = np.max(len_qs)
    min_len_qs = np.min(len_qs)

    queries = lemmatize(qs)

    len_queries = np.array([len(q.split()) for q in queries])
    avg_len_queries = np.mean(len_queries)
    max_len_queries = np.max(len_queries)
    min_len_queries = np.min(len_queries)

    print('Questions:')
    print('avg len: ', avg_len_qs)  #
    print('max len: ', max_len_qs)  #
    print('min len: ', min_len_qs)  #

    print('Queries:')
    print('avg len: ', avg_len_queries)  #
    print('max len: ', max_len_queries)  #
    print('min len: ', min_len_queries)  #


def squad_query_json():
    json_file = open('train-v2.0.json')
    #json_file = open('dev-v2.0.json')
    json_str = json_file.read()
    json_data = json.loads(json_str)

    json_query = copy.deepcopy(json_data)
    for i in json_query['data']:
        for d in i['paragraphs']:
            for q in d['qas']:
                #q['question'] = lemmatize_single(q['question'])
                q['question'] = lemmatize_single_w_qwords(q['question'])
    json_query_str = json.dumps(json_query)
    with open('squad_query_qwords_train.json', 'w') as f:
   # with open('squad_query_qwords_dev.json', 'w') as f:
        f.write(json_query_str)


def output_hotpot_queries():
    json_file = open('hotpot_train_v1.1.json')
    json_str = json_file.read()
    json_data = json.loads(json_str)
    qs = [item['question'] for item in json_data]
    queries = lemmatize(qs)
    both = zip(queries, qs)

    with open('hotpot_queries.txt', 'w') as text_file:
        for query, q in both:
            text_file.write(query + '\n' + q + '\n\n')


if __name__ == "__main__":
    #output_hotpot_queries()
    #output_squad_queries()
    #squad_stats()
    #squad_query_json()
    #output_google_queries()
    #print(stopwords.words('english'))
    #google_stats()
    #hotpot_stats()

    # with open('v1.0-simplified_simplified-nq-train.jsonl', 'r') as json_file:
    #     json_list = list(json_file)
    #
    # qs = []
    # for json_str in json_list:
    #     result = json.loads(json_str)
    #     print(result.keys())
    #     break

    #output_squad_queries_nparrays()

    # f = np.loadtxt('keras_input_train.txt', dtype='str', delimiter='/t')
    #
    # print(f.shape)

    #squad_query_json()

    # f = np.load('keras_input_train.npy')
    # print(f.shape)

    # l = [['a b c', 'd e f'], ['g h i', 'j k i']]
    # f = np.array(l)
    # print(f.shape)

    output_squad_queries_somin() # He asked for a certain format....