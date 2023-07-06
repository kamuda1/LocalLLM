import discord
import asyncio
import functools
import typing
import os
from dotenv import load_dotenv
from langchain import PromptTemplate, LLMChain
from transformers import pipeline
from langchain import HuggingFacePipeline

load_dotenv()

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)


@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')


def to_thread(func: typing.Callable) -> typing.Coroutine:
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        return await asyncio.to_thread(func, *args, **kwargs)
    return wrapper


@to_thread
def local_llm(input_prompt: str) -> str:
    template = """Question: {question}

    Answer: Let's think step by step."""

    prompt = PromptTemplate(template=template, input_variables=["question"])
    print(prompt)

    # max_length has typically been deprecated for max_new_tokens
    pipe = pipeline("text-generation", model='gpt-neo-1.3B', max_new_tokens=64, model_kwargs={"temperature": 0})
    hf = HuggingFacePipeline(pipeline=pipe)

    llm_chain = LLMChain(prompt=prompt, llm=hf)

    output = llm_chain.run(input_prompt)

    return output


@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.startswith('/LocalLLM'):

        prompt = message.content[9:]
        if len(prompt) == 0:
            message.channel.send('What can I help you with?')
        llm_output = await local_llm(prompt)

        await message.channel.send(llm_output)


if __name__ == '__main__':
    discord_token = os.getenv("discord_token")
    client.run(discord_token)
