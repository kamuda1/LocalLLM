import discord
import os
from dotenv import load_dotenv
from local_llm import LocalLLM

load_dotenv()

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)

local_llm_model = LocalLLM()

@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')


@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.startswith('RecipeMe'):
        prompt = message.content[9:]
        if len(prompt) == 0:
            message.channel.send('What can I help you with?')
        llm_output = await local_llm_model.local_llm(prompt)

        await message.channel.send(llm_output)


if __name__ == '__main__':
    discord_token = os.getenv("discord_token")
    client.run(discord_token)
