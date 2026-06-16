import httpx

async def invoke(base_url, payload):
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            f"{base_url}/chat/completions",
            json=payload
        )

        response.raise_for_status()
        return response.json()
