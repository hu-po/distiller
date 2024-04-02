import asyncio
import logging

import api_claude
import api_openai
import api_gemini
import api_mistral

log = logging.getLogger(__name__)

async def caption(
    image_path: str = "cover.png",
    prompt: str = "Describe the image",
) -> dict[str, str]:
    tasks = [
        asyncio.create_task(api_claude.async_image(prompt, image_path), name="Claude"),
        asyncio.create_task(api_openai.async_image(prompt, image_path), name="OpenAI"),
        asyncio.create_task(api_gemini.async_image(prompt, image_path), name="Gemini"),
        # Replicate is slow, hence why it's commented out
        # asyncio.create_task(api_mistral.async_image(prompt, image_path), name="Mistral"),
    ]
    results = await asyncio.gather(*tasks)
    results_dict = {} 
    for task, result in zip(tasks, results):
        results_dict[task.get_name()] = result
    log.debug(results_dict)
    return results_dict

if __name__ == "__main__":
    log.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    log.addHandler(handler)
    asyncio.run(caption())
