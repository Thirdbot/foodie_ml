import asyncio
import websockets


async def server(websocket,path):
    async for message in websocket:
        print(f"Received message: {message}")
        response = f"Echo: {message}"
        await websocket.send(response)
        print(f"Sent response: {response}")

start_server = websockets.serve(server, "localhost", 8765)
asyncio.get_event_loop().run_forever()