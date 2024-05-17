import cv2
import pyaudio
import wave

# OpenCV video capture
cap = cv2.VideoCapture(0)

# PyAudio audio capture
audio_format = pyaudio.paInt16
audio_channels = 1
audio_rate = 44100
audio_chunk = 1024

p = pyaudio.PyAudio()
stream = p.open(format=audio_format,
                channels=audio_channels,
                rate=audio_rate,
                input=True,
                frames_per_buffer=audio_chunk)

# OpenCV video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

while True:
    # Capture video frame
    ret, frame = cap.read()
    if not ret:
        break

    # Capture audio frame
    audio_data = stream.read(audio_chunk)

    # Write video frame to output
    out.write(frame)

    # Break the loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
stream.stop_stream()
stream.close()
p.terminate()
