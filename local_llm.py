from langchain import LLMChain
from transformers import pipeline
from langchain import HuggingFacePipeline
import asyncio
import functools
import typing
import os
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate

def read_text_file(file_path):
    with open(file_path, 'r') as f:
        return f.read()


def read_all_few_shot_examples(file_path: str):
    file_names = os.listdir(file_path)
    all_examples = [read_text_file(os.path.join(file_path, file)) for file in file_names if file.endswith('.txt')]
    return all_examples


def to_thread(func: typing.Callable) -> typing.Coroutine:
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        return await asyncio.to_thread(func, *args, **kwargs)

    return wrapper


class LocalLLM:
    def __init__(self, model_name='gpt-neo-1.3B', temperature=0):
        self.llm_chain = self.init_llm_chain(model_name, temperature)

    @to_thread
    def local_llm(self, input_prompt: str) -> str:
        output = self.llm_chain.run(input_prompt)
        return output

    @staticmethod
    def init_llm_chain(model='gpt-neo-1.3B', temperature=0):
        all_few_shot_examples = read_all_few_shot_examples(file_path=r"D:\LLMDiscord\text_data\recipes")[:6]

        main_question = 'Give me a dinner menu with a grocery list for the following few days.'

        all_questions = [main_question] * len(all_few_shot_examples)
        all_answers = all_few_shot_examples

        examples = [{"question": question, "answer": answer} for question, answer in zip(all_questions, all_answers)]
        example_prompt = PromptTemplate(input_variables=["question", "answer"],
                                        template="Question: {question}\n{answer}")

        prompt = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            suffix="Question: {input}",
            input_variables=["input"])

        pipe = pipeline(
            "text-generation", model=model, max_new_tokens=128, model_kwargs={"temperature": temperature}
        )

        hf = HuggingFacePipeline(pipeline=pipe)

        llm_chain = LLMChain(prompt=prompt, llm=hf)
        return llm_chain


