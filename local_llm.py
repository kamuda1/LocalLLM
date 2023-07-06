from langchain import PromptTemplate, LLMChain
from transformers import AutoModelForCausalLM, pipeline
from langchain import HuggingFacePipeline


def llm(input_prompt: str) -> str:
    template = """Question: {question}

    Answer: Let's think step by step."""

    prompt = PromptTemplate(template=template, input_variables=["question"])
    print(prompt)

    # max_length has typically been deprecated for max_new_tokens
    pipe = pipeline(
        "text-generation", model='gpt-neo-1.3B', max_new_tokens=64, model_kwargs={"temperature": 0}
    )
    hf = HuggingFacePipeline(pipeline=pipe)

    llm_chain = LLMChain(prompt=prompt, llm=hf)

    output = llm_chain.run(input_prompt)

    return output
