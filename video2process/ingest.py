import datetime

from typing import Union, List

from pydantic import BaseModel, Field
from vertexai.generative_models import GenerationConfig, Part
from vertexai.preview.generative_models import GenerativeModel
from vertexai.preview.caching import CachedContent
from google.api_core.exceptions import InvalidArgument

from promptgit import PromptRepo

from .utils import flatten_openapi

class Step(BaseModel):
    speaker: str
    actionSummary: str
    actionDetails: str
    timeStamp: str = Field(description='timestamp in EDT timezone and in the following format %Y-%m-%dT%H:%M:%S%z')

class Process(BaseModel):
    issue: str
    ticketNumber: str
    ticketPlatform: str
    actions: List[Step]

class ProcessFeedback(BaseModel):
    support: bool = Field(description='Are all actions in the playbook in the video?')
    rating: int = Field(description='Rating of the submitted playbook on scale 1 to 5. 5 means playbook is perfect an no change is needed, 1 is no use either due to wrong order of steps or halucinated steps')
    recommendations : str = Field(description='Recomendation for writer how to improve playbook')

prompts = PromptRepo('', dir='prompts')

def generate_process(
        video_uri: str, mime_type: str = 'video/mp4', 
        events: Union[List, str] = None,
        chat: Union[List, str] = None,
        model='gemini-1.5-pro-002',
        cache: bool = True,
        n: int = 5,
        ):

    if cache:
        contents = [
            Part.from_uri(
                video_uri,
                mime_type=mime_type,
            )]

        try:
            cached_content = CachedContent.create(
                model_name="gemini-1.5-pro-002",
                contents=contents,
                ttl=datetime.timedelta(minutes=60),
                display_name="example-cache",
            )
        except InvalidArgument:
            # Video is not big enough for cache (at least 32,769 tokens)
            cached_content = None
    else:
        cached_content = None


    config_step1 = GenerationConfig(response_mime_type='application/json', response_schema=flatten_openapi(Process.schema()))
    if cached_content:
        model_step1 = GenerativeModel.from_cached_content(cached_content=cached_content, generation_config=config_step1)
    else:
        model_step1 = GenerativeModel(model_name=model, generation_config=config_step1)

    content_step1 = [
        Part.from_text(prompts['analyze/chat'])
    ]

    if not cached_content:
        content_step1 += [
            Part.from_text('<VIDEO>'),
            Part.from_uri(video_uri, mime_type=mime_type),
            Part.from_text('</VIDEO>')
        ]

    if events:
        if isinstance(events, list):
            events = '\n'.join(events)
        content_step1 += [
            Part.from_text(prompts['analyze/events'].format(events=events))
        ]

    config_step2 = GenerationConfig(response_mime_type='application/json', response_schema=flatten_openapi(ProcessFeedback.model_json_schema()))
    if cached_content:
        model_step2 = GenerativeModel.from_cached_content(cached_content=cached_content, generation_config=config_step2)
    else:
        model_step2 = GenerativeModel(model_name=model, generation_config=config_step2)


    process_list = []

    for _ in range(n):

        response = model_step1.generate_content(content_step1)
        process = Process.model_validate_json(response.candidates[0].content.parts[0].text)

        content_step2 = [
            Part.from_text(prompts['reflection/header'])
        ]

        if not cached_content:
            content_step2 += [
                Part.from_text('<VIDEO>'),
                Part.from_uri(video_uri, mime_type=mime_type),
                Part.from_text('</VIDEO>')
            ]
        content_step2 += [
            Part.from_text('<PROCESS>'),
            Part.from_text(process.model_dump_json()),
            Part.from_text('</PROCESS>')    
        ]

        response = model_step2.generate_content(content_step2)
        feedback = ProcessFeedback.model_validate_json(response.candidates[0].content.parts[0].text)

        process_list.append((process, feedback))

    return process_list


