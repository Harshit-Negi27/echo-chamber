import os
from twilio.rest import Client
from dotenv import load_dotenv

# Load credentials from your .env file
load_dotenv()

account_sid = os.getenv('TWILIO_ACCOUNT_SID')
auth_token = os.getenv('TWILIO_AUTH_TOKEN')

if not account_sid or not auth_token:
    raise ValueError("Missing Twilio credentials in .env file!")

client = Client(account_sid, auth_token)

# Create outbound call FROM Twilio TO your Indian number
call = client.calls.create(
    to='+917217826794',  # YOUR Indian cell number
    from_='+12603515665',  # Your Twilio US number
    url='https://sprayful-albertine-speechless.ngrok-free.dev/incoming-call',  
    method='POST',
    status_callback='https://sprayful-albertine-speechless.ngrok-free.dev/call-status',  
    status_callback_event=['initiated', 'ringing', 'answered', 'completed'],
    status_callback_method='POST'
)

print(f"Call SID: {call.sid}")
print(f"Status: {call.status}")
print("You should receive a call in 3-5 seconds. Answer it!")