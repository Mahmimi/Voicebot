import asyncio
from time import sleep
import time

async def async_generator():
    for i in range(5):
        asyncio.sleep(1)  # Simulate an asynchronous operation
        yield i

async def main():
    start = time.time()
    async for item in async_generator():
        print(item)
    stop = time.time()
    print(f"Total time taken: {stop - start}")

# Run the event loop to execute the asynchronous code
asyncio.run(main())
