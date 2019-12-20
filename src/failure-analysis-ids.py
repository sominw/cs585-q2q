from collections import defaultdict

def data_generator(f):
    lines = []
    for line in f:
        if line.strip() == '':
            continue

        lines.append(line.strip())

        if len(lines) == 5:
            yield lines.copy()
            lines = []


def get_ids(f):
    ids = defaultdict(dict)

    for lines in data_generator(f):
        q_id = lines[0].split(':', 1)[1].strip()
        ids[q_id]['id'] = lines[0].split(':', 1)[1].strip()
        ids[q_id]['question'] = lines[1].split(':', 1)[1].strip()
        ids[q_id]['query'] = lines[2].split(':', 1)[1].strip()
        ids[q_id]['correct'] = lines[3].split(':', 1)[1].strip()
        ids[q_id]['wrong'] = lines[4].split(':', 1)[1].strip()

    return ids

files = ['ea_query_on_query', 'ea_query_wh_on_query_wh', 'ea_query_wh_on_query', 'ea_query_wh_on_question', 'ea_wh_on_query', 'ea_wh_on_question']

ids = {}
intersect_ids = set()

for filename in files:
    fd = open(f'{filename}.txt', 'r')
    fids = get_ids(fd)

    ids[filename] = fids

    if len(intersect_ids) == 0:
        intersect_ids |= set(fids.keys())
    else:
        intersect_ids &= set(fids.keys())

    fd.close()


print(f'Number of questions that all evaluations fail on: {len(intersect_ids)}')


with open('ea_all_six.txt', 'w') as f:
    any_id = next(iter(ids.values()))

    for qid in intersect_ids:
        f.write(f'id: {any_id[qid]["id"]}\n')
        f.write(f'question: {any_id[qid]["question"]}\n')

        for filename in files:
            f.write(f'query {filename}: {ids[filename][qid]["query"]}\n')

        f.write(f'right answer: {any_id[qid]["correct"]}\n')

        for filename in files:
            f.write(f'wrong answer {filename}: {ids[filename][qid]["wrong"]}\n')

        f.write('\n')