#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

#
# Loads the SYSTEM_PROMPT content from the file 'prompt.txt'
#

import asyncio
import os
import sys

from loguru import logger

from noaa_sdk import NOAA
from openai.types.chat import ChatCompletionToolParam

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import EndFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.nim import NimLLMService
from pipecat.services.riva import FastPitchTTSService, ParakeetSTTService
from pipecat.services.together import TogetherLLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport


logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

# Load SYSTEM_PROMPT content from file 'prompt.txt'
with open("prompt.txt", "r") as f:
    SYSTEM_PROMPT = f.read()


async def main():
    print("___________________________________*")
    print("___________________________________*")
    print("___________________________________* Navigate to")
    print(
        "___________________________________* https://pc-34b1bdc94a7741719b57b2efb82d658e.daily.co/pipecat"
    )
    print("___________________________________* to talk to NVIDIA NIM bot.")
    print("___________________________________*")
    print("___________________________________*")

    # Url to talk to the NVIDIA NIM bot
    room_url = "https://pc-34b1bdc94a7741719b57b2efb82d658e.daily.co/pipecat"

    transport = DailyTransport(
        room_url,
        None,
        "NVIDIA NIM",
        DailyParams(
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            vad_audio_passthrough=True,
        ),
    )

    stt = ParakeetSTTService(api_key=os.getenv("NVIDIA_API_KEY"))

    allowed_services = [NimLLMService, TogetherLLMService]  # prevent imports from being removed
    # llm = NimLLMService(api_key=os.getenv("NVIDIA_API_KEY"), model="meta/llama-3.3-70b-instruct")
    llm = TogetherLLMService(
        api_key=os.getenv("TOGETHER_API_KEY"), model="meta-llama/Llama-3.3-70B-Instruct-Turbo"
    )

    tts = FastPitchTTSService(api_key=os.getenv("NVIDIA_API_KEY"))

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    ## tool calling
    async def start_fetch_weather(function_name, llm, context):
        print(f"Starting fetch_weather_from_api with function_name: {function_name}")

    async def get_noaa_simple_weather(latitude: float, longitude: float, **kwargs):
        print(f"noaa get simple weather for '{latitude}, {longitude}'")
        n = NOAA()
        description = False
        fahrenheit_temp = 0
        try:
            observations = n.get_observations_by_lat_lon(latitude, longitude, num_of_stations=1)
            for observation in observations:
                description = observation["textDescription"]
                celcius_temp = observation["temperature"]["value"]
                if description:
                    break

            fahrenheit_temp = (celcius_temp * 9 / 5) + 32

            # fallback to temperature if no description in any of the observations
            if not description:
                description = fahrenheit_temp
        except Exception as e:
            print(f"Error getting noaa weather: {e}")

        return description, fahrenheit_temp

    async def fetch_weather_from_api(
        function_name, tool_call_id, args, llm, context, result_callback
    ):
        location = args["location"]
        latitude = float(args["latitude"])
        longitude = float(args["longitude"])
        print(f"fetch_weather_from_api * location: {location}, zip_code: {latitude}, {longitude}")

        if latitude and longitude:
            description, fahrenheit_temp = await get_noaa_simple_weather(latitude, longitude)
        else:
            return await result_callback("I don't recognize that location.")

        if not description:
            await result_callback(
                f"I'm sorry, I can't get the weather for {location} right now. Can you ask again please?"
            )
        else:
            await result_callback(
                f"The weather in {location} is currently {round(fahrenheit_temp)} degrees and {description}."
            )

    tools = [
        ChatCompletionToolParam(
            type="function",
            function={
                "name": "get_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The location for the weather request.",
                        },
                        "latitude": {
                            "type": "string",
                            "description": "Infer the latitude from the location. Supply latitude as a string. For example, '42.3601'.",
                        },
                        "longitude": {
                            "type": "string",
                            "description": "Infer the longitude from the location. Supply longitude as a string. For example, '-71.0589'.",
                        },
                    },
                    "required": ["location", "latitude", "longitude"],
                },
            },
        ),
    ]

    llm.register_function(None, fetch_weather_from_api, start_callback=start_fetch_weather)

    context = OpenAILLMContext(messages, tools)
    context_aggregator = llm.create_context_aggregator(context)

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,  # STT
            context_aggregator.user(),  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            context_aggregator.assistant(),  # Assistant spoken responses
        ]
    )

    task = PipelineTask(
        pipeline,
        PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
        ),
    )

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        print(f"Participant left: {participant}")
        await task.queue_frame(EndFrame())

    runner = PipelineRunner()

    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())