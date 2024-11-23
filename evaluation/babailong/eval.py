from modelzipper.tutils import *
import matplotlib.pylab as plt
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import numpy as np


TASK_LABELS = {'qa1': ['bathroom', 'bedroom', 'garden', 'hallway', 'kitchen', 'office'], 
 'qa2': ['bathroom', 'bedroom', 'garden', 'hallway', 'kitchen', 'office'], 
 'qa3': ['bathroom', 'bedroom', 'garden', 'hallway', 'kitchen', 'office'], 
 'qa4': ['bathroom', 'bedroom', 'garden', 'hallway', 'kitchen', 'office'], 
 'qa5': ['Bill', 'Fred', 'Jeff', 'Mary', 'apple', 'football', 'milk'], 
 'qa6': ['no', 'yes'], 
 'qa7': ['none', 'one', 'three', 'two'], 
 'qa8': ['apple', 'football', 'milk', 'nothing'], 
 'qa9': ['no', 'yes'], 
 'qa10': ['maybe', 'no', 'yes'],
 'qa11': ['bathroom', 'bedroom', 'garden', 'hallway', 'kitchen', 'office'], 
 'qa12': ['bathroom', 'bedroom', 'garden', 'hallway', 'kitchen', 'office'], 
 'qa13': ['bathroom', 'bedroom', 'garden', 'hallway', 'kitchen', 'office'], 
 'qa14': ['bedroom', 'cinema', 'kitchen', 'office', 'park', 'school'], 
 'qa15': ['cat', 'mouse', 'sheep', 'wolf'], 
 'qa16': ['gray', 'green', 'white', 'yellow'], 
 'qa17': ['no', 'yes'], 
 'qa18': ['no', 'yes'], 
 'qa19': ['e,e', 'e,n', 'e,s', 'n,e', 'n,n', 'n,w', 's,e', 's,s', 's,w', 'w,n', 'w,s', 'w,w'], 
 'qa20': ['bedroom', 'bored', 'garden', 'hungry', 'kitchen', 'thirsty', 'tired']
}


def preprocess_output(output):
    output = output.lower()
    # take only the first sentence from output
    output = output.split('.')[0]
    # filter responses when model tries to generate examples
    output = output.split('<context>')[0]
    output = output.split('<example>')[0]
    output = output.split('Question')[0]
    return output


def compare_answers(target, output, task_labels):
    output = preprocess_output(output)
    target = target.lower()
    
    # extract labels that were mentioned in the model output
    labels_in_output = {label for label in task_labels if label in output}

    # check if the target is the only prediction
    if ',' in target and len(target) > 3: 
        # if target contains multiple subtargets in qa8
        subtargets = target.split(',')
        num_subtargets = len(subtargets)
        if all([t in labels_in_output for t in subtargets]) and len(labels_in_output) == num_subtargets:
            return True
    else:
        if target in labels_in_output and len(labels_in_output) == 1:
            return True

    return False


def eval_fn(pred_path):
    tasks = ['qa1', 'qa2', 'qa3', 'qa4', 'qa5', 'qa6', 'qa7', 'qa8', 'qa9', 'qa10']
    lengths = ['4k', '8k', '16k', '32k', '64k', '128k']
    accuracy = np.ones((len(tasks), len(lengths))) * -1

    content = auto_read_data(pred_path)

    df = pd.DataFrame(columns=['task', 'ctx_length', 'preds', 'goldens', 'task_labels'])

    # 填充这个 DataFrame 的所有可能组合，使用列表来收集数据
    for task in tasks:
        for length in lengths:
            df = df.append({'task': task, 'ctx_length': length, 'preds': [], 'goldens': [], 'task_labels': []}, ignore_index=True)
    
    # 填充 DataFrame
    for item in content:
        task, ctx_length = item['task'], item['ctx_length']
        pred, golden = item['pred'][0], item['golden']
        idx = df.index[(df['task'] == task) & (df['ctx_length'] == ctx_length)]
        df.at[idx, 'preds'].append(pred)
        df.at[idx, 'goldens'].append(golden)
        df.at[idx, 'task_labels'].append(TASK_LABELS[task])

    df['accuracy'] = df.apply(lambda row: compare_answers(row['preds'], row['goldens'], row['task_labels']), axis=1)

    print(df)


        
