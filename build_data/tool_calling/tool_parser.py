from modelzipper.tutils import *
from prompt_pool import *
from tqdm import tqdm
from typing import List, Optional, Dict, Any, Tuple
import sys
import random
import json
import re


def rename_api(s):  # api
    return change_name(standardize(s))


def rename_category(s):  # category
    return standardize_category(standardize(s))


def change_name(name):
    change_list = ["from", "class", "return", "false", "true", "id", "and"]
    if name in change_list:
        name = "is_" + name
    return name


def standardize_category(category):
    save_category = category.replace(" ", "_").replace(",", "_").replace("/", "_")
    while " " in save_category or "," in save_category:
        save_category = save_category.replace(" ", "_").replace(",", "_")
    save_category = save_category.replace("__", "_")
    return save_category


def standardize(string):
    res = re.compile("[^\\u4e00-\\u9fa5^a-z^A-Z^0-9^_]")
    string = res.sub("_", string)
    string = re.sub(r"(_)\1+","_", string).lower()
    while True:
        if len(string) == 0:
            return string
        if string[0] == "_":
            string = string[1:]
        else:
            break
    while True:
        if len(string) == 0:
            return string
        if string[-1] == "_":
            string = string[:-1]
        else:
            break
    if string[0].isdigit():
        string = "get_" + string
    return string


class SingleAPIParser:
    def __init__(self, api_data):
        self.category_name = api_data["category_name"]
        self.tool_name = api_data["tool_name"]
        self.tool_description = api_data["tool_description"]
        self.api_id = api_data["api_cnt"] if "api_cnt" in api_data else api_data["api_id"]
        self.api_name = rename_api(api_data["api_name"])
        self.api_description = api_data["api_description"]
        self.properties = api_data["parameters"]["properties"]
        self.required_properties = api_data["parameters"]["required"]
        self.parameters = api_data["parameters"] if "parameters" in api_data else None
        if "golden_api_list" in api_data:
            self.call_parameter = api_data["golden_api_list"]
        elif "call_parameter" in api_data:
            self.call_parameter = api_data["call_parameter"]
        else:
            self.call_parameter = None
        self.api_calling = api_data["api_calling"] if "api_calling" in api_data else None
        self.conv = api_data["conv"] if "conv" in api_data else None

    def get_call_parameter(self):
        flatten_call_parameter = ">>> call_parameter:\n"
        for k in self.api_calling["call_parameter"]:
            flatten_call_parameter += f"parameter: {k} | value: {self.api_calling['call_parameter'][k]}\n"
        flatten_call_parameter += f">>> call_response: {self.api_calling['call_response']}"
        return flatten_call_parameter

    def get_api_info(self):
        return f""">>> api_id: {self.api_id}\n>>> category_name: {self.category_name}\n>>> tool_name: {self.tool_name}\n>>> tool_description: {self.tool_description}\n{self.flatten_api_info()}"""

    def flatten_api_info(self):
        flatten_api_info = f"<API_{self.api_id}>\n>>> api_name: {self.api_name}\n>>> api_description: {self.api_description}\n>>> param_properties:\n"
        for k in self.properties:
            flatten_api_info += self.api_properties_prompt(k, self.properties[k]["description"], self.properties[k]["type"], self.properties[k]["default_value"] if "default_value" in self.properties[k] else "")
        flatten_api_info += f">>> required_properties: {', '.join(self.required_properties)}\n</API_{self.api_id}>"
        return flatten_api_info

    def api_properties_prompt(self, n, d, t, default_value=""):
        if len(n) == 0: n = "None"
        if len(d) == 0: d = "None"
        if len(t) == 0: t = "None"
        if len(default_value) == 0: default_value = "None"
        return f"property_name: {n} | description: {d} | type: {t} | default_value: {default_value}\n"


class ToolSample:
    def __init__(self, sample = None, model_name: str = 'llama-3', type: str = 'parallel', benchmark_type: str = 'tool_calling', total_pool: List = None):  
        self.model_name = model_name
        self.type = type
        self.all_apis = total_pool
        self.benchmark_type = benchmark_type

        if sample is not None:
            self.query = sample["query"].strip()
            if type == 'multiple':
                self.api_list = [sample["api_1"][0], sample["api_2"][0]]
                self.call_parameter: List = [
                    {
                        "api_name": rename_api(sample["api_1"][0]['api_name']),
                        "call_parameter": sample["api_1"][0]['call_parameter']
                    },
                    {
                        "api_name": rename_api(sample["api_2"][0]['api_name']),
                        "call_parameter": sample["api_2"][0]['call_parameter']
                    },
                ]
            elif type == 'single':
                self.api_list = [sample]
                self.call_parameter: List = [
                    {
                        "api_name": rename_api(sample["api_name"]),
                        "call_parameter": sample["call_parameter"]
                    }
                ]
            elif type == 'parallel':
                self.api_list = [sample]
                self.call_parameter: List = [
                    {
                        "api_name": rename_api(sample["api_name"]),
                        "call_parameter": item
                    } for item in sample["call_parameter"]
                ]
            
            if type == "single":
                self.plan_list: List = [sample["plan"]]
                self.answer_list: List = [sample["answer"]]
                self.selection_list: List = [sample["selection"]]
            elif type in ["multiple", "parallel"]:
                self.plan_list: List = sample["plan_list"]
                self.answer_list: List = sample["answer_list"]
                self.selection_list: List = sample["selection_list"]
           
            self.golden_api_names = set([rename_api(api["api_name"]) for api in self.selection_list])

        ### prompt template
        if benchmark_type == 'tool_calling':
            self.system_prompt_template = TOOL_CALLING_SYSTEM_PROMPT_TEMPLATE
            self.demonstration_prompt = TOOL_DEMONSTRATION_PROMPT
        elif benchmark_type == 'tool_location':
            self.system_prompt_template = TOOL_LOCATION_SYSTEM_PROMPT_TEMPLATE
            self.demonstration_prompt = TOOL_LOCATION_DEMONSTRATION_PROMPT

        self.query_prompt = "<QUERY> {query} </QUERY>\n"
        self.plan_then_gen_prompt = "<PLAN> {plan} </PLAN>\n<ANSWER> {answer} </ANSWER>\n"
        self.call_parameter_prompts = {
            "api_name": "<API_{id}> {api_name} </API_{id}>",
            "param_value": "<PARAM> {param} </PARAM> <VALUE> {value} </VALUE>"
        } 

        self.post_process_api_list()
    
    def search_api_id(self, api_name):
        return self.all_api_dict[api_name]

    def create_demonstration(self, benchmark_type='tool_calling', return_str=False):
        system_prompt = self.create_system_prompt()
        source_docs = self.create_current_api_str()
        if return_str:
            return system_prompt + '\n' + self.demonstration_prompt.format(query=self.query, demonstration=self.flatten_conv(benchmark_type))
        query = self.query_prompt.format(query=self.query)
        query = query.rstrip("\n")
        return {
            "system_prompt": system_prompt,
            "query": query,
            "answer": self.flatten_conv(benchmark_type),
            "source_docs": source_docs,
        }

    def post_process_api_list(self):
        """
        note: each sample in self.all_apis denotes one api
        """
        # if self.type == "multiple":
        #     tmp_1 = set([rename_api(api['api_1'][0]["api_name"]) for api in self.all_apis if "api_1" in api])
        #     tmp_2 = set([rename_api(api['api_2'][0]["api_name"]) for api in self.all_apis if "api_2" in api])
        #     self.all_api_names = tmp_1.union(tmp_2)
        # else:
        self.all_api_names = set([rename_api(api["api_name"]) for api in self.all_apis])

        # create api list prompt
        api_list_prompt = ""
        self.all_api_dict = {}
        self.all_str_api = []
        for api in self.all_apis:
            # if self.type == "multiple":
            #     if 'api_1' not in api or 'api_2' not in api:
            #         return False
            #     single_api_1 = SingleAPIParser(api['api_1'][0])
            #     single_api_2 = SingleAPIParser(api['api_2'][0])
            #     api_1_id, api_2_id = single_api_1.api_id, single_api_2.api_id
            #     # api_1
            #     api_list_prompt += single_api_1.flatten_api_info() + "\n"
            #     self.all_api_dict[single_api_1.api_name] = api_1_id
            #     # api_2
            #     api_list_prompt += single_api_2.flatten_api_info() + "\n"
            #     self.all_api_dict[single_api_2.api_name] = api_2_id
            #     self.all_str_api.append(single_api_1.flatten_api_info())
            #     self.all_str_api.append(single_api_2.flatten_api_info())
            # else:  
            single_api = SingleAPIParser(api)
            api_id = single_api.api_id
            api_list_prompt += single_api.flatten_api_info() + "\n"
            self.all_api_dict[single_api.api_name] = api_id
            self.all_str_api.append(single_api.flatten_api_info())

        api_list_prompt = api_list_prompt.rstrip("\n")
        self.system_prompt = self.system_prompt_template.format(tools=api_list_prompt)
        

    def create_current_api_str(self):
        current_api_str = ""
        for api in self.api_list:
            parsed_api = SingleAPIParser(api)
            parsed_api.api_id = self.search_api_id(parsed_api.api_name)
            current_api_str += f"{parsed_api.flatten_api_info()}\n"
        current_api_str = current_api_str.rstrip("\n")
        return current_api_str


    def create_call_parameters(self):
        # flatten the list of apis into a string
        flattern_call_parameters = ""
        if self.type in ["multiple", "parallel"]:
            for i, item in enumerate(self.call_parameter):
                api_name = item["api_name"]
                param_values = item["call_parameter"]
                api_id = self.search_api_id(api_name)
                flattern_call_parameters += self.call_parameter_prompts["api_name"].format(api_name=api_name, id=api_id) + " "
                for k, v in param_values.items():
                    flattern_call_parameters += self.call_parameter_prompts["param_value"].format(param=k, value=param_values[k]) + " "
                flattern_call_parameters = flattern_call_parameters.rstrip()
                flattern_call_parameters += "\n"
        elif self.type == "single":
            api_name = self.call_parameter[0]["api_name"]
            param_values = self.call_parameter[0]["call_parameter"]
            api_id = self.search_api_id(api_name)
            flattern_call_parameters += self.call_parameter_prompts["api_name"].format(api_name=api_name, id=api_id) + " "
            for k, v in param_values.items():
                flattern_call_parameters += self.call_parameter_prompts["param_value"].format(param=k, value=param_values[k]) + " "
            flattern_call_parameters = flattern_call_parameters.rstrip()
            flattern_call_parameters += "\n"
        # elif self.type == "parallel":
        #     api_name = self.call_parameter[0]["api_name"]
        #     api_id = self.search_api_id(api_name)
        #     param_values = self.call_parameter[0]["call_parameter"]
        #     for item in param_values:
        #         flattern_call_parameters += self.call_parameter_prompts["api_name"].format(api_name=api_name, id=api_id) + " "
        #         for k, v in item.items():
        #             flattern_call_parameters += self.call_parameter_prompts["param_value"].format(param=k, value=item[k]) + " "
        #         flattern_call_parameters = flattern_call_parameters.rstrip()
        #         flattern_call_parameters += "\n"
        else:
            raise ValueError("type not supported")
        flattern_call_parameters = flattern_call_parameters.rstrip("\n")
        return flattern_call_parameters


    def create_system_prompt(self):
        if not self.golden_api_names.issubset(self.all_api_names):
            return False
        return self.system_prompt

    
    def create_reason_answer_mseeage(self):
        system_prompt = self.create_system_prompt()
        if not system_prompt:
            import pdb; pdb.set_trace()
            return False
        # call_parameters = self.create_call_parameters()
        model_output = self.flatten_conv()
        source_docs = f'<TOOL_DOC>\n{self.create_current_api_str()}\n</TOOL_DOC>'

        return {"system_prompt": system_prompt, "query": self.query_prompt.format(query=self.query), "call_parameters": self.call_parameter, "model_output": model_output, "source_docs": source_docs}


    def create_action_plan(self):
        action_plan = []
        for i, api in enumerate(self.api_list):
            single_api = SingleAPIParser(api)
            action_plan.append(single_api.get_call_parameter())
        return action_plan


    def parse_golden_message(self):
        golden_api_lst = []
        for call_api in self.golden:
            parsed_api = SingleAPIParser(call_api)
            golden_api_str = parsed_api.get_flatten_single_api()
            plan_answer = parsed_api.flatten_conv()
            golden_api_lst.append((golden_api_str, plan_answer))
        return golden_api_lst    


    def flatten_conv(self, benchmark_type='tool_calling'):
        if len(self.plan_list) == 1:
            plan_str = self.plan_list[0]
        else:
            plan_str = ', '.join(self.plan_list)
        
        if len(self.answer_list) == 1:
            answer_str = self.answer_list[0]
        else:
            answer_str = ', '.join(self.answer_list)
        call_parameters = self.create_call_parameters()
        # answer = json.dumps(call_parameters)
        return self.plan_then_gen_prompt.format(plan=plan_str, answer=call_parameters)