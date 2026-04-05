import os
import json
import base64
import asyncio
import websockets
import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import Response
from groq import AsyncGroq
from dotenv import load_dotenv

# Load environment variables securely from .env file
load_dotenv()

app = FastAPI()

# API Keys loaded securely
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not DEEPGRAM_API_KEY or not GROQ_API_KEY:
    raise ValueError("Missing API keys! Check your .env file.")

# Initialize clients
groq_client = AsyncGroq(api_key=GROQ_API_KEY)

# Deepgram config
DG_URL = (
    "wss://api.deepgram.com/v1/listen"
    "?model=nova-2"
    "&encoding=mulaw"
    "&sample_rate=8000"
    "&channels=1"
    "&endpointing=300"
    "&vad_events=true"
    "&interim_results=true"
)

# Global state for barge-in detection
is_ai_speaking = asyncio.Event()

@app.post("/incoming-call")
async def handle_incoming_call(request: Request):
    host = request.headers.get("host")
    protocol = "wss" if "https" in str(request.url) else "ws"
    websocket_url = f"{protocol}://{host}/media-stream"
    
    print(f"📞 Incoming call received!")
    
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{websocket_url}" />
    </Connect>
</Response>"""
    
    return Response(content=twiml, media_type="application/xml")

@app.post("/call-status")
async def handle_call_status(request: Request):
    form_data = await request.form()
    print(f"📊 Call Status: {form_data.get('CallStatus')}")
    return {"status": "received"}

# ============================================
# AI Response Pipeline
# ============================================

async def generate_ai_response(user_transcript: str, twilio_ws: WebSocket, stream_sid: str, conversation_history: list):
    print(f"\n🧠 Generating AI response for: '{user_transcript}'")
    is_ai_speaking.set()
    
    try:
        async for audio_chunk in stream_llm_to_tts(user_transcript, conversation_history):
            await send_audio_to_twilio(audio_chunk, twilio_ws, stream_sid)
        
        await twilio_ws.send_json({
            "event": "mark",
            "streamSid": stream_sid,
            "mark": {"name": "audio_complete"}
        })
        print("✅ AI generation complete (Twilio buffering to phone speaker...)")
        
    except asyncio.CancelledError:
        print("🚨 AI response cancelled due to barge-in")
        is_ai_speaking.clear() 
        raise
    except Exception as e:
        print(f"❌ AI response error: {e}")
        is_ai_speaking.clear()
        import traceback
        traceback.print_exc()

async def stream_llm_to_tts(user_transcript: str, conversation_history: list):
    system_prompt = """You are a helpful AI assistant on a phone call. 
Keep responses concise (1-2 sentences max). Speak naturally and conversationally."""
    
    conversation_history.append({"role": "user", "content": user_transcript})
    messages = [{"role": "system", "content": system_prompt}] + conversation_history

    try:
        completion = await groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            stream=True,
            max_tokens=150,
            temperature=0.7
        )
        
        print("🧠 Groq streaming...")
        full_response = ""
        async for chunk in completion:
            delta = chunk.choices[0].delta
            if delta.content:
                text_chunk = delta.content
                full_response += text_chunk
                print(f"{text_chunk}", end="", flush=True)
        
        print(f"\n📝 Complete response: '{full_response}'")
        conversation_history.append({"role": "assistant", "content": full_response})
        
        if len(conversation_history) > 20:
            del conversation_history[:-20]
        
        print("🔊 Generating speech with Deepgram Aura...")
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.deepgram.com/v1/speak?model=aura-asteria-en&encoding=mulaw&sample_rate=8000",
                headers={
                    "Authorization": f"Token {DEEPGRAM_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={"text": full_response},
                timeout=30.0
            )
            
            chunk_count = 0
            async for audio_chunk in response.aiter_bytes(chunk_size=640):
                chunk_count += 1
                yield audio_chunk
            print(f"🎵 Sent {chunk_count} audio chunks")
            
    except asyncio.CancelledError:
        raise
    except Exception as e:
        print(f"❌ LLM/TTS error: {e}")
        raise

async def send_audio_to_twilio(audio_chunk: bytes, twilio_ws: WebSocket, stream_sid: str):
    try:
        base64_audio = base64.b64encode(audio_chunk).decode('utf-8')
        await twilio_ws.send_json({
            "event": "media",
            "streamSid": stream_sid,
            "media": {"payload": base64_audio}
        })
    except asyncio.CancelledError:
        raise
    except Exception as e:
        print(f"❌ Audio send error: {e}")

# ============================================
# MAIN WEBSOCKET HANDLER
# ============================================

@app.websocket("/media-stream")
async def handle_media_stream(twilio_ws: WebSocket):
    print("📞 Twilio WebSocket connection attempt...")
    await twilio_ws.accept()
    
    auth_headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}
    dg_ws = None
    dg_task = None
    stream_sid = None
    active_ai_task = None
    conversation_history = []
    
    try:
        dg_ws = await websockets.connect(DG_URL, additional_headers=auth_headers)
        print("✅ Deepgram connected successfully")
        
        latest_transcript = ""
        
        async def receive_from_deepgram():
            nonlocal latest_transcript, active_ai_task
            try:
                async for message in dg_ws:
                    result = json.loads(message)
                    
                    if result.get("type") == "Results":
                        channel = result.get("channel", {})
                        alternatives = channel.get("alternatives", [])
                        if alternatives:
                            transcript = alternatives[0].get("transcript", "")
                            is_final = result.get("is_final", False)
                            speech_final = result.get("speech_final", False)
                            
                            # MERGED FIX 2: The Noise-Filtered Barge-In Check (2+ words)
                            word_count = len(transcript.strip().split())
                            if word_count >= 2 and is_ai_speaking.is_set():
                                print(f"\n🚨 BARGE-IN DETECTED: '{transcript}' ({word_count} words)")
                                
                                # Step 1: Cancel AI task
                                if active_ai_task and not active_ai_task.done():
                                    print("❌ Cancelling AI response...")
                                    active_ai_task.cancel()
                                    try:
                                        await active_ai_task
                                    except asyncio.CancelledError:
                                        pass
                                        
                                # Step 2: Clear Twilio buffer
                                if stream_sid:
                                    print("🧹 Flushing Twilio audio buffer...")
                                    await twilio_ws.send_json({
                                        "event": "clear",
                                        "streamSid": stream_sid
                                    })
                                    
                                # Step 3: Reset state
                                is_ai_speaking.clear()
                                latest_transcript = ""
                                print("✅ Ready to listen again\n")
                                continue
                                
                            if transcript:
                                print(f"🎤 {'[FINAL]' if is_final else '[INTERIM]'} {transcript}")
                            
                            if is_final:
                                latest_transcript = transcript
                            
                            # Trigger response only if AI is NOT speaking
                            if speech_final and latest_transcript.strip() and not is_ai_speaking.is_set():
                                print(f"🛑 User finished speaking")
                                active_ai_task = asyncio.create_task(
                                    generate_ai_response(
                                        user_transcript=latest_transcript,
                                        twilio_ws=twilio_ws,
                                        stream_sid=stream_sid,
                                        conversation_history=conversation_history
                                    )
                                )
                                latest_transcript = ""
                                
            except asyncio.CancelledError:
                print("🚫 Deepgram task cancelled")
            except Exception as e:
                print(f"❌ Deepgram error: {e}")

        dg_task = asyncio.create_task(receive_from_deepgram())

        while True:
            try:
                message = await twilio_ws.receive_text()
                data = json.loads(message)
                event_type = data.get('event')

                if event_type == 'start':
                    stream_sid = data.get('start', {}).get('streamSid')
                    print(f"▶️  Stream started: {stream_sid}")
                    
                elif event_type == 'media':
                    payload_bytes = base64.b64decode(data['media']['payload'])
                    try:
                        await dg_ws.send(payload_bytes)
                    except Exception:
                        break
                        
                elif event_type == 'mark':
                    if data.get('mark', {}).get('name') == 'audio_complete':
                        is_ai_speaking.clear()
                        print("🏁 Twilio physically finished playing audio")
                        
                elif event_type == 'stop':
                    print("⏹️  Stream stopped by Twilio")
                    break
                    
            except WebSocketDisconnect:
                break

    except Exception as e:
        print(f"❌ Fatal error: {e}")
        
    finally:
        if dg_task: dg_task.cancel()
        if dg_ws: await dg_ws.close()
        print("👋 Cleanup complete")

@app.get("/")
async def health_check():
    return {"status": "running"}