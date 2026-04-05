import json
import logging
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import Response

# Set up clean terminal logging for the "Terminal Flex" in Phase 5
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Anthro-Lite Voice Router")

# IMPORTANT: You will update this with your ngrok forwarding domain later.
# Format: "your-subdomain.ngrok-free.app" (Do NOT include https://)
NGROK_DOMAIN = "sprayful-albertine-speechless.ngrok-free.dev"

@app.post("/incoming-call")
async def handle_incoming_call(request: Request):
    """
    Twilio hits this endpoint when a call comes in.

    We return TwiML to establish a bi-directional WebSocket stream.
    """

    logger.info("[HTTP] Incoming call received from Twilio.")
    
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
    <Response>
        <Connect>
            <Stream url="wss://{NGROK_DOMAIN}/media-stream" />
        </Connect>
    </Response>"""
    
    return Response(content=twiml, media_type="text/xml")

@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    """
    The main asyncio loop. This accepts the connection and receives 
    the raw, base64-encoded audio chunks from Twilio.
    """
    await websocket.accept()
    logger.info("[WS] WebSocket connection accepted.")

    try:
        while True:
            # Twilio sends messages as JSON strings
            message = await websocket.receive_text()
            data = json.loads(message)
            
            event_type = data.get('event')

            if event_type == 'connected':
                logger.info("[Twilio] Stream connected.")
                
            elif event_type == 'start':
                stream_sid = data['start']['streamSid']
                logger.info(f"[Twilio] Stream started. Stream SID: {stream_sid}")
                
            elif event_type == 'media':
                # This is the raw audio payload (mu-law 8kHz, base64 encoded)
                # We log the size here just to verify flow, rather than printing massive base64 strings
                payload = data['media']['payload']
                logger.info(f"[Twilio] Received media chunk -> {len(payload)} bytes")
                
                # In Phase 2, we will instantly forward this 'payload' to Deepgram Nova-2
                
            elif event_type == 'stop':
                logger.info("[Twilio] Stream stopped by user.")
                break
                
    except Exception as e:
        logger.warning(f"[WS] Connection closed or errored: {e}")