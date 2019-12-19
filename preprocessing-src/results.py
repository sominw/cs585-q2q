import json
from collections import Counter


# get {id,gold answer list} dict
# get question text
with open('dev-v2.0.json') as json_file:
    json_str = json_file.read()
    json_data_dev = json.loads(json_str)
    qs = [q['question'] for i in json_data_dev['data'] for d in i['paragraphs'] for q in d['qas']]
    qid_ans_gold = {q['id']: [ans['text'] for ans in q['answers']] for i in json_data_dev['data'] for d in i['paragraphs'] for q in d['qas']}
    question_text = {q['id']: q['question'] for i in json_data_dev['data'] for d in i['paragraphs'] for q in d['qas']}

# get query text
with open('squad_query_dev.json') as json_file:
    json_str = json_file.read()
    json_data = json.loads(json_str)
    qs = [q['question'] for i in json_data['data'] for d in i['paragraphs'] for q in d['qas']]
    query_text = {q['id']: q['question'] for i in json_data['data'] for d in i['paragraphs'] for q in d['qas']}

# get wh-word text
with open('squad_query_qwords_dev.json') as json_file:
    json_str = json_file.read()
    json_data = json.loads(json_str)
    qs = [q['question'] for i in json_data['data'] for d in i['paragraphs'] for q in d['qas']]
    wh_words_text = {q['id']: q['question'] for i in json_data['data'] for d in i['paragraphs'] for q in d['qas']}


# get query model predictions
with open('predictions_query.json') as json_file:
    json_str = json_file.read()
    json_data = json.loads(json_str)
    qid_ans_query = json_data

# get question model predictions
with open('predictions_question.json') as json_file:
    json_str = json_file.read()
    json_data = json.loads(json_str)
    qid_ans_question = json_data

# get query-on-question model predictions
with open('predictions_query_on_quest.json') as json_file:
    json_str = json_file.read()
    json_data = json.loads(json_str)
    qid_ans_query_on_quest = json_data

# get wh-words model predictions
with open('predictions_wh_words.json') as json_file:
    json_str = json_file.read()
    json_data = json.loads(json_str)
    qid_ans_wh_words = json_data

# get wh-words model predictions
with open('predictions_wh_words_on_quest.json') as json_file:
    json_str = json_file.read()
    json_data = json.loads(json_str)
    qid_ans_wh_words_on_quest = json_data

# print out ids where query model failed while question model didn't (may be a bug because I made it complicated, but seems to work)
with open('ea_query_wrong_question_right.txt', 'w') as out:
    i = 0
    for k, v in qid_ans_question.items():
        if ((qid_ans_query[k] == '' and qid_ans_gold[k]) or (qid_ans_gold[k] and qid_ans_query[k] not in qid_ans_gold[k]) or (qid_ans_query[k] and not qid_ans_gold[k])) and ((v == '' and not qid_ans_gold[k]) or (v in qid_ans_gold[k])):
            correct = qid_ans_gold[k]
            incorrect = v
            out.write('id: ' + k + '\n' + 'Question: '+ question_text[k] + '\n' + 'Query: ' + query_text[k]+ '\n' + 'right answer: ' + str(qid_ans_gold[k]) + '\n' + 'wrong answer: ' + qid_ans_question[k] + '\n\n')
            i += 1
print(i)

# # print out ids where query model failed while question model didn't (may be a bug because I made it complicated, but seems to work)
# with open('ea_query_right_question_wrong.txt', 'w') as out:
#     i = 0
#     for k, v in qid_ans_query.items():
#         if ((qid_ans_question[k] == '' and qid_ans_gold[k]) or (qid_ans_gold[k] and qid_ans_question[k] not in qid_ans_gold[k]) or (qid_ans_question[k] and not qid_ans_gold[k])) and ((v == '' and not qid_ans_gold[k]) or (v in qid_ans_gold[k])):
#             correct = qid_ans_gold[k]
#             incorrect = v
#             out.write('id: ' + k + '\n' + 'Question: '+ question_text[k] + '\n' + 'Query: ' + query_text[k]+ '\n' + 'right answer: ' + str(qid_ans_gold[k]) + '\n' + 'wrong answer: ' + qid_ans_question[k] + '\n\n')
#             i += 1
# print(i)

# print out ids where query model failed while question model didn't (may be a bug because I made it complicated, but seems to work)
# with open('ea_wh_query_on_wh_query_fail_wh_query_on_quest_pass.txt', 'w') as out:
#     i = 0
#     for k, v in qid_ans_wh_words_on_quest.items():
#         if ((qid_ans_wh_words[k] == '' and qid_ans_gold[k]) or (qid_ans_gold[k] and qid_ans_wh_words[k] not in qid_ans_gold[k]) or (qid_ans_wh_words[k] and not qid_ans_gold[k])) and ((v == '' and not qid_ans_gold[k]) or (v in qid_ans_gold[k])):
#             correct = qid_ans_gold[k]
#             incorrect = v
#             out.write('id: ' + k + '\n' + 'Question: '+ question_text[k] + '\n' + 'Query: ' + wh_words_text[k]+ '\n' + 'right answer: ' + str(qid_ans_gold[k]) + '\n' + 'wrong answer: ' + qid_ans_wh_words[k] + '\n\n')
#             i += 1
# print(i)
#
# # print out ids where query model failed while question model didn't (may be a bug because I made it complicated, but seems to work)
# with open('ea_query_wrong_query_on_quest_right.txt', 'w') as out:
#     i = 0
#     set_titles = set()
#     ctr = Counter()
#     for k, v in qid_ans_query.items():
#         if ((v == '' and qid_ans_gold[k]) or (qid_ans_gold[k] and v not in qid_ans_gold[k]) or (v and not qid_ans_gold[k])) and ((qid_ans_query_on_quest[k] == '' and not qid_ans_gold[k]) or (qid_ans_query_on_quest[k] in qid_ans_gold[k])):
#             correct = qid_ans_gold[k]
#             incorrect = v
#             out.write('id: ' + k + '\n' + 'Question: '+ question_text[k] + '\n' + 'Query: ' + query_text[k]+ '\n' + 'right answer: ' + str(qid_ans_gold[k]) + '\n' + 'wrong answer: ' + qid_ans_query[k] + '\n\n')
#             i += 1
#
#             title = [i['title'] for i in json_data_dev['data'] for d in i['paragraphs'] for q in d['qas'] if q['id'] == k]
#             #print(title)
#             set_titles.add(title[0])
#             # ctr[title[0]] += 1
# print(i)
# print(len(set_titles))
# #print(ctr)
#

# # print out ids where query model failed while question model didn't (may be a bug because I made it complicated, but seems to work)
# with open('ea_query_wrong_query_on_quest_right_ans.txt', 'w') as out:
#     i = 0
#     set_titles = set()
#     ctr = Counter()
#     for k, v in qid_ans_query.items():
#         if (not(v == '' and qid_ans_gold[k]) and (qid_ans_gold[k] and v not in qid_ans_gold[k]) and not(v and not qid_ans_gold[k])) and ((qid_ans_query_on_quest[k] == '' and not qid_ans_gold[k]) or (qid_ans_query_on_quest[k] in qid_ans_gold[k])):
#             correct = qid_ans_gold[k]
#             incorrect = v
#             out.write('id: ' + k + '\n' + 'Question: '+ question_text[k] + '\n' + 'Query: ' + query_text[k]+ '\n' + 'right answer: ' + str(qid_ans_gold[k]) + '\n' + 'wrong answer: ' + qid_ans_query[k] + '\n\n')
#             i += 1
#
#             title = [i['title'] for i in json_data_dev['data'] for d in i['paragraphs'] for q in d['qas'] if q['id'] == k]
#             #print(title)
#             set_titles.add(title[0])
#             ctr[title[0]] += 1
# print(i)
# print(len(set_titles))
# print(ctr)

# # print out ids where query model failed while question model didn't (may be a bug because I made it complicated, but seems to work)
# with open('ea_query_query_on_quest_both_fail_quest_passes.txt', 'w') as out:
#     for k, v in qid_ans_query.items():
#         if ((v == '' and qid_ans_gold[k]) or (qid_ans_gold[k] and v not in qid_ans_gold[k]) or (v and not qid_ans_gold[k])) and ((qid_ans_query_on_quest[k] == '' and qid_ans_gold[k]) or (qid_ans_gold[k] and qid_ans_query_on_quest[k] not in qid_ans_gold[k]) or (qid_ans_query_on_quest[k] and not qid_ans_gold[k])):
#             correct = qid_ans_gold[k]
#             incorrect = v
#             out.write('id: ' + k + '\n' + 'Question: '+ question_text[k] + '\n' + 'Query: ' + query_text[k]+ '\n' + 'right answer: ' + str(qid_ans_gold[k]) + '\n' + 'wrong answer query: ' + qid_ans_query[k] + '\n' + 'wrong answer query_on_quest: ' + qid_ans_query_on_quest[k] + '\n\n')


# with open('ea_query_pass_query_on_quest_fail.txt', 'w') as out:
#
#     for k, v in qid_ans_query.items():
#
#         if ((qid_ans_query_on_quest[k] == '' and qid_ans_gold[k]) or (qid_ans_gold[k] and qid_ans_query_on_quest[k] not in qid_ans_gold[k]) or (qid_ans_query_on_quest[k] and not qid_ans_gold[k])) and ((v == '' and not qid_ans_gold[k]) or (v in qid_ans_gold[k])):
#             correct = qid_ans_gold[k]
#             incorrect = qid_ans_query_on_quest[k]
#             out.write('id: ' + k + '\n' + 'Question: '+ question_text[k] + '\n' + 'Query: ' + query_text[k]+ '\n' +  'right answer: ' + str(qid_ans_gold[k]) + '\n' + 'wrong answer: ' + incorrect + '\n\n')


# print out ids where query model succeeded and wh-model failed (may be a bug because I made it complicated, but seems to work)
# with open('ea_query_pass_wh_fail.txt', 'w') as out:
#     total = 0
#     wh_fail = 0
#     for k, v in qid_ans_query.items():
#         total +=1
#         if ((qid_ans_wh_words[k] == '' and qid_ans_gold[k]) or (qid_ans_gold[k] and qid_ans_wh_words[k] not in qid_ans_gold[k]) or (qid_ans_wh_words[k] and not qid_ans_gold[k])) and ((v == '' and not qid_ans_gold[k]) or (v in qid_ans_gold[k])):
#             correct = qid_ans_gold[k]
#             incorrect = qid_ans_wh_words[k]
#             out.write('id: ' + k + '\n' + 'Question: '+ question_text[k] + '\n' + 'Query: ' + query_text[k]+ '\n' + 'Wh-query: ' + wh_words_text[k] + '\n' + 'right answer: ' + str(qid_ans_gold[k]) + '\n' + 'wrong answer: ' + incorrect + '\n\n')

# print out ids where query model succeeded and wh-model failed (may be a bug because I made it complicated, but seems to work)
with open('ea_wh_on_quest_pass_wh_fail.txt', 'w') as out:
    total = 0
    wh_fail = 0
    for k, v in qid_ans_wh_words_on_quest.items():
        total +=1
        if ((qid_ans_wh_words[k] == '' and qid_ans_gold[k]) or (qid_ans_gold[k] and qid_ans_wh_words[k] not in qid_ans_gold[k]) or (qid_ans_wh_words[k] and not qid_ans_gold[k])) and ((v == '' and not qid_ans_gold[k]) or (v in qid_ans_gold[k])):
            correct = qid_ans_gold[k]
            incorrect = qid_ans_wh_words[k]
            out.write('id: ' + k + '\n' + 'Question: '+ question_text[k] + '\n' + 'Query: ' + query_text[k]+ '\n' + 'Wh-query: ' + wh_words_text[k] + '\n' + 'right answer: ' + str(qid_ans_gold[k]) + '\n' + 'wrong answer: ' + incorrect + '\n\n')



# # print out ids where query model succeeded and wh-model failed (may be a bug because I made it complicated, but seems to work)
# #with open('ea_query_pass_wh_fail.txt', 'w') as out:
#     total = 0
#     wh_on_quest_fail = 0
#     for k, v in qid_ans_query.items():
#         total +=1
#         if ((qid_ans_wh_words_on_quest[k] == '' and qid_ans_gold[k]) or (qid_ans_gold[k] and qid_ans_wh_words_on_quest[k] not in qid_ans_gold[k]) or (qid_ans_wh_words_on_quest[k] and not qid_ans_gold[k])):
#             correct = qid_ans_gold[k]
#             incorrect = qid_ans_wh_words[k]
#             #out.write('id: ' + k + '\n' + 'Question: '+ question_text[k] + '\n' + 'Query: ' + query_text[k]+ '\n' + 'Wh-query: ' + wh_words_text[k] + '\n' + 'right answer: ' + str(qid_ans_gold[k]) + '\n' + 'wrong answer: ' + incorrect + '\n\n')
#             wh_on_quest_fail +=1
# print(wh_on_quest_fail)
# print(total)


#print out ids where query model succeeded and wh-model failed (may be a bug because I made it complicated, but seems to work)

# true_pos = 0
# true_neg = 0
# false_pos = 0
# false_neg = 0
# pos = 0
# neg = 0
# true_neg_incorrect = 0
# group = qid_ans_question
# for k, v in qid_ans_query.items():
#     if ((group[k] == '' and qid_ans_gold[k]) ):
#         correct = qid_ans_gold[k]
#         incorrect = qid_ans_wh_words[k]
#         false_pos +=1
#     elif ((group[k] == '' and not qid_ans_gold[k]) ):
#         true_pos +=1
#
#     elif ((group[k] != '' and not qid_ans_gold[k])):
#         false_neg += 1
#     elif ((group[k] != '' and qid_ans_gold[k])):
#         true_neg += 1
#     if (group[k] != '' and qid_ans_gold[k] and group[k] not in qid_ans_gold[k]):
#         true_neg_incorrect +=1
#         #out.write('id: ' + k + '\n' + 'Question: '+ question_text[k] + '\n' + 'Query: ' + query_text[k]+ '\n' + 'Wh-query: ' + wh_words_text[k] + '\n' + 'right answer: ' + str(qid_ans_gold[k]) + '\n' + 'wrong answer: ' + incorrect + '\n\n')
# print(true_pos,true_neg,false_pos,false_neg, true_neg_incorrect)

#
# wh_empty = 0
# query_empty = 0
# gold_empty = 0
# question_empty = 0
# i = 0
# for k, v in qid_ans_query.items():
#     if v == '':
#         query_empty+=1
#     if not qid_ans_gold[k]:
#         gold_empty+=1
#     if qid_ans_wh_words[k] == '':
#         wh_empty+=1
#     if qid_ans_question[k] == '':
#         question_empty +=1
#     i+=1
# print(wh_empty/i, query_empty/i, gold_empty/i, question_empty/i)