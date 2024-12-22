DOUBAO_TEMPLATES="""You are an AI assistant that evaluates whether the response to the question is correct according to the golden label.I'll give you a question, a response and a golden label, where the golden label is the true answer to the question. You should judge whether the response to the question is the same as the golden label. Follow these instructions:\n\n1.You can only reply `True` or `False`,where `True` means the response to the question is the same as th golden label `False` means the opposite.\n2.Reply True or False"""




import os,time
from typing import Optional, List
from openai import OpenAI

def chat(ak       : str = "ea28bf46-979c-49b9-b08a-92303bb99052", 
         url      : str = "https://ark.cn-beijing.volces.com/api/v3", 
         model    : str = "ep-20240725161435-hjkcn",
         message  : Optional[str] = None, 
         template : List  = None, 
         **kwargs
         ) -> Optional[None | str]:
    # ....

    # 1. 封装 message 到template 里面，如果用户没有提供message，用默认的template封装
    # try：、、 sleep 5 s, try 3 次失败返回None
    # 2. 输入到API(model 默认是gpt40)
    
    
    client = OpenAI(
    api_key = ak,
    base_url = url,
    )


    messages=template if message is None \
        else [{"role":"user","content":message}]

    while True:
        error_times=0
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            if error_times==3:break
            print(f"发生错误: {str(e)}")
            print("5秒后重试...")
            time.sleep(5)
            error_times+=1

    print("询问失败! 返回 None")

def compare_answers_o1(target, output, question, task_labels):
    query=f"<question>{question}\n<response>{output}\n<golden>{target}"
    template=[{'role':'system','content':DOUBAO_TEMPLATES},
              {'role':'user','content':"<question>Where was Marry before afternoon?\n<response>Marry was in the bedroom.**Step 1.Marry went to the playground in the\n<golden>bedroom"},
              {'role':'assistant','content':'True'},
              {'role':'user','content':query}]


    output=chat(template=template,max_tokens=5)

    if "true" in output.lower():
        return True
    return False

