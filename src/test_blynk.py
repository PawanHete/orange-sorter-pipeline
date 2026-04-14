import time
import requests # Used to send data via URL

# ==========================================
# CONFIGURATION
# ==========================================
BLYNK_AUTH = 'mYw4T5wtToiJ1sKlkSUIEfV9HRF4DgGW'  # Replace with your Blynk Auth Token

# Server for India (Bangalore)
BLYNK_SERVER = 'blr1.blynk.cloud'

# Start counters at 0
val_v0 = 0
val_v1 = 0

def update_blynk(pin, value):
    """
    Sends a value to a specific Blynk Pin using HTTP GET.
    URL Format: https://{server}/external/api/update?token={token}&{pin}={value}
    """
    url = f"https://{BLYNK_SERVER}/external/api/update?token={BLYNK_AUTH}&{pin}={value}"
    
    try:
        print(f" Sending {value} to {pin}...", end=" ")
        
        # Send request (Timeout after 5s if internet is bad)
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            print(" OK!")
        else:
            print(f" Failed (Code: {response.status_code})")
            
    except Exception as e:
        print(f"\n Network Error: {e}")

# ==========================================
# 🔄 MAIN LOOP (Runs every 5 seconds)
# ==========================================
print("---------------------------------------")
print(f"Starting HTTP Test Loop to {BLYNK_SERVER}")
print(" Press Ctrl+C to stop")
print("---------------------------------------")

try:
    while True:
        # 1. Increment Data
        val_v0 += 1  # Simulate Healthy Count
        val_v1 += 2  # Simulate Unhealthy Count (increments faster)

        # 2. Send Data
        update_blynk("v0", val_v0)
        update_blynk("v1", val_v1)

        # 3. Wait 5 Seconds
        print("Waiting 5 seconds...\n")
        time.sleep(5)

except KeyboardInterrupt:
    print("\n Test Stopped.")