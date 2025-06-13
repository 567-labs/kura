"""
Main web application service. Serves the static frontend with WebSocket terminal.
"""

from pathlib import Path
import modal
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import json
import asyncio

app = modal.App("terminal-py")


@app.function(
    scaledown_window=600,
    timeout=600,
)
@modal.asgi_app()
def web():
    from fastapi.middleware.cors import CORSMiddleware

    web_app = FastAPI()

    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @web_app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        sandbox = None
        try:
            await websocket.accept()
            print("WebSocket connection accepted")

            # Send connection confirmation
            await websocket.send_text(
                json.dumps({"type": "system", "data": "Connected to Modal Sandbox!"})
            )

            # Create a sandbox for this session
            sandbox = modal.Sandbox.create(
                image=modal.Image.debian_slim().pip_install("numpy", "pandas"),
                app=app,  # Pass the app reference
            )

            await websocket.send_text(
                json.dumps(
                    {
                        "type": "system",
                        "data": "Sandbox created and ready for commands!",
                    }
                )
            )

            # Main message loop
            while True:
                try:
                    # Wait for command from client
                    data = await websocket.receive_text()
                    print(f"Received: {data}")

                    command_data = json.loads(data)
                    command = command_data.get("command", "").strip()

                    if not command:
                        continue

                    print(f"Executing command: {command}")

                    # Execute command in sandbox
                    process = sandbox.exec("bash", "-c", command)

                    # Send output line by line
                    try:
                        for line in process.stdout:
                            await websocket.send_text(
                                json.dumps({"type": "stdout", "data": line})
                            )

                        # Send stderr if any
                        try:
                            stderr_content = process.stderr.read()
                            if stderr_content:
                                await websocket.send_text(
                                    json.dumps(
                                        {"type": "stderr", "data": stderr_content}
                                    )
                                )
                        except:
                            pass

                        # Send exit code
                        await websocket.send_text(
                            json.dumps(
                                {
                                    "type": "exit",
                                    "code": process.returncode
                                    if hasattr(process, "returncode")
                                    else 0,
                                }
                            )
                        )

                    except Exception as exec_error:
                        print(f"Execution error: {exec_error}")
                        await websocket.send_text(
                            json.dumps(
                                {
                                    "type": "error",
                                    "data": f"Command execution failed: {str(exec_error)}",
                                }
                            )
                        )

                except WebSocketDisconnect:
                    print("WebSocket disconnected")
                    break
                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {e}")
                    await websocket.send_text(
                        json.dumps({"type": "error", "data": "Invalid JSON format"})
                    )
                except Exception as e:
                    print(f"Unexpected error: {e}")
                    await websocket.send_text(
                        json.dumps(
                            {"type": "error", "data": f"Unexpected error: {str(e)}"}
                        )
                    )

        except WebSocketDisconnect:
            print("WebSocket disconnected during setup")
        except Exception as e:
            print(f"WebSocket error: {e}")
            try:
                await websocket.send_text(
                    json.dumps({"type": "error", "data": f"Connection error: {str(e)}"})
                )
            except:
                pass
        finally:
            # Clean up sandbox
            if sandbox:
                try:
                    sandbox.terminate()
                    print("Sandbox terminated")
                except Exception as e:
                    print(f"Error terminating sandbox: {e}")

    @web_app.get("/")
    async def root():
        return {"message": "Modal Terminal WebSocket Server", "websocket_url": "/ws"}

    @web_app.get("/health")
    async def health():
        return {"status": "healthy"}

    return web_app
