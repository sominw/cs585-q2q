import json

id = input('id: ')
with open('dev-v2.0.json') as json_file:
    json_str = json_file.read()
    json_data = json.loads(json_str)
    qs = [q['question'] for i in json_data['data'] for d in i['paragraphs'] for q in d['qas']]
    qid_ans_gold = {q['id']: [ans['text'] for ans in q['answers']] for i in json_data['data'] for d in i['paragraphs'] for q in d['qas']}
    question_text = {q['id']: q['question'] for i in json_data['data'] for d in i['paragraphs'] for q in d['qas']}
    context = [d['context']  for i in json_data['data'] for d in i['paragraphs'] for q in d['qas'] if q['id']==id]
    # title = [i['title'] for i in json_data['data'] for d in i['paragraphs'] for q in d['qas'] if q['id']==id]
    print(context)

#get question model predictions
with open('predictions_question.json') as json_file:
    json_str = json_file.read()
    json_data = json.loads(json_str)
    qid_ans_question = json_data
    for k,v in qid_ans_question.items():
        if k==id:
            print('Question prediction: ', v)
