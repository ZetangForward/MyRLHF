
ONCE_GENERATION_PROMPT = {
    "prefix": (
        "You are a thoughtful assistant designed to help the user solve practical problems using the available tools.\n"
        "You are required to strictly follow the descriptions to call and use the tools listed below as needed.\n"
        "{tools}"
    ), 
    "suffix": (
        "Before providing your final answer, plan your approach carefully and outline the steps you'll take to reach the best solution. Enclose this plan within <PLAN></PLAN>.\n"
        "After planning, generate your answer based on that plan and enclose it within <ANSWER></ANSWER>.\n"
        "For each API call, ensure the following:\n"
        "1. Use the correct API ID in the format <API_{{id}}> </API_{{id}}>.\n"
        "2. Each parameter must be enclosed using <PARAM> {{param}} </PARAM> and <VALUE> {{value}} </VALUE> separately.\n"
        "3. If an API is called multiple times with different parameters, use <API_{{id}}> </API_{{id}}> for each call, followed by the corresponding parameters.\n"
        "4. Remember to always provide your reasoning process within <PLAN></PLAN> and the final answer within <ANSWER></ANSWER>.\n"
    )
}

TWO_STEP_GENERATION_PROMPT = {
    "prefix": (
        "You are a thoughtful assistant designed to help the user solve practical problems using the available tools.\n"
        "Strictly follow the descriptions to locate and outline the relevant tool documentation listed below as needed.\n"
        "{tools}"
    ), 
    "suffix": (
        "Before planning your approach or generating a final answer, first identify and outline the relevant documentation that describes the tools available.\n"
        "Enclose this tool documentation within <TOOL_DOC></TOOL_DOC>.\n"
        "Ensure that you provide a clear and concise description of each tool's capabilities and usage based on the documentation provided.\n"
    )
}

SECOND_STEP_PROMPT = {
    "prefix": (
        "You have already retrieved the relevant information source"
    ), 
    "suffix": (
        "Based on your retrieval source, please proceed with the tool calling while adhering to the following requirements:\n"
        "Following your planning, construct your answer and enclose it within <ANSWER></ANSWER>.\n"
        "In implementing each API call, adhere to the following guidelines:\n"
        "1. Correctly use the API ID, formatted as <API_{{id}}></API_{{id}}>.\n"
        "2. Enclose each parameter and its value separately using <PARAM>{{param}}</PARAM> and <VALUE>{{value}}</VALUE>.\n"
        "3. If the same API is called multiple times with different parameters, repeat the use of <API_{{id}}></API_{{id}}> for each instance, followed by the specific parameters for that call.\n"
    )
}


TOOL_DEMONSTRATION_PROMPT = (
    "Here is one example:\n"
    "<QUERY> {query} </QUERY>\n"
    "{demonstration}\n"
    "In this example:\n"
    "1. The correct API ID is used with <API_{{id}}> </API_{{id}}>.\n"
    "2. Each parameter is enclosed using <PARAM> {{param}} </PARAM> and <VALUE> {{value}} </VALUE>.\n"
    "3. Multiple API calls with different parameters are properly formatted with separate <API_{{id}}> </API_{{id}}> blocks for each.\n"
    "4. The reasoning process is within <PLAN></PLAN> and the final answer is within <ANSWER></ANSWER>.\n"
    "Following this example, diectly provide your reasoning process within <PLAN></PLAN> and the final answer within <ANSWER></ANSWER>. Do not explain in the final answer, just provide the answer in <ANSWER></ANSWER>\n"
)