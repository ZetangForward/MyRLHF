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
    content = auto_read_data(pred_path)

    all_results = dict([(task, dict([(length, []) for length in lengths])) for task in tasks])

    # 填充 DataFrame
    for item in content:
        task, ctx_length = item['task'], item['ctx_length']
        pred, golden = item['pred'][0], item['golden']
        all_results[task][ctx_length].append((golden, pred, TASK_LABELS[task]))

    for task, item in all_results.items():
        for length, content in item.items():
            acc = np.array([compare_answers(*item) for item in content]).mean()
            all_results[task][length] = acc
    
    # 转换字典为DataFrame
    df = pd.DataFrame(all_results).T
    df.index.name = 'Task'

    # 重排序：任务按照qa1到qa10排序，长度按4k, 8k, 16k, 32k, 64k, 128k排序
    df = df.sort_index(axis=0, ascending=True)  # 排序任务名
    df = df[sorted(df.columns, key=lambda x: int(x.replace('k', '')))]  # 排序长度

    # 绘制热力图
    matplotlib.rc('font', size=14)  # 设置字体大小
    cmap = LinearSegmentedColormap.from_list('ryg', ["red", "yellow", "green"], N=256)  # 自定义颜色
    figsize = (5, 3.5)
    fig, ax = plt.subplots(1, 1, figsize=figsize)  # 创建图形

    # 使用seaborn绘制热力图
    sns.heatmap(df, cmap=cmap, vmin=0, vmax=1, annot=True, fmt=".2f", linewidths=.5, ax=ax)

    # 设置标题和标签
    ax.set_xlabel('Context size')
    ax.set_ylabel('Tasks')

    # 调整边距以删除多余的白色边框
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.savefig('/mnt/petrelfs/tangzecheng/MyRLHF/evaluation/babailong/heatmap_result.png', bbox_inches='tight', dpi=300)


if __name__ == "__main__":
    eval_fn('/mnt/petrelfs/tangzecheng/local_data/inference_results/llama-3_1-8B-Instruct/babilong/reasoning/preds_babilong.jsonl')



        
