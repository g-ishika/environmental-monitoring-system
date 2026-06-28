# API Documentation

## POST /predict
Predict the class of an audio file.

### Request
- File: .wav or .mp3 audio file

### Response
- event_type: string
- confidence: float
- is_alert: boolean