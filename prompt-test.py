#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os
import sys

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMMessagesFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.nim import NimLLMService
from pipecat.services.riva import ParakeetSTTService
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.services.cartesia import CartesiaTTSService

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

PROMPT_FILENAME = "prompt.txt"


async def main():
    print("___________________________________*")
    print("___________________________________*")
    print("___________________________________* Navigate to")
    print(
        "___________________________________* https://pc-34b1bdc94a7741719b57b2efb82d658e.daily.co/prod-test"
    )
    print("___________________________________* to talk to NVIDIA NIM bot.")
    print("___________________________________*")
    print("___________________________________*")

    prompt = ""
    try:
        with open(PROMPT_FILENAME, "r") as file:
            prompt = file.read()
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except IOError as e:
        print(f"Error: An I/O error occurred: {e}")
        return None

    # Url to talk to the NVIDIA NIM bot
    room_url = "https://pc-34b1bdc94a7741719b57b2efb82d658e.daily.co/prod-test"

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

    llm = NimLLMService(api_key=os.getenv("NVIDIA_API_KEY"))

    # tts = FastPitchTTSService(api_key=os.getenv("NVIDIA_API_KEY"))
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="79a125e8-cd45-4c13-8a67-188112f4dd22",  # British Lady
    )

    messages = [
        {
            "role": "system",
            "content": prompt,
        },
    ]

    context = OpenAILLMContext(messages)
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

    task = PipelineTask(pipeline, PipelineParams(allow_interruptions=True))

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        # Kick off the conversation.
        messages.append({"role": "system", "content": "Please introduce yourself to the user."})
        await task.queue_frames([LLMMessagesFrame(messages)])

    runner = PipelineRunner()

    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
