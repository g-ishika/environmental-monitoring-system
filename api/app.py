from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import torch
import numpy as np
from pathlib import Path
import yaml
import uuid

from inference.predictor import Predictor
from inference.alert_system import AlertSystem
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Load configuration
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize components
predictor = Predictor(
    model_path='models_checkpoints/best_model.pt',
    config=config
)
alert_system = AlertSystem(config)

# Create FastAPI app
app = FastAPI(
    title="Environmental Monitoring API",
    description="API for environmental sound monitoring and alerting",
    version="1.0.0"
)


# Pydantic models
class PredictionResponse(BaseModel):
    alert_id: Optional[str] = None
    event_type: str
    confidence: float
    severity: str
    is_alert: bool
    timestamp: float
    location: Optional[str] = "Unknown"


class AlertHistoryResponse(BaseModel):
    total_alerts: int
    alerts: List[dict]


# Endpoints
@app.get("/")
async def root():
    return {
        "message": "Environmental Monitoring API",
        "status": "active",
        "version": "1.0.0"
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_audio(
    file: UploadFile = File(...),
    location: Optional[str] = "Unknown"
):
    """Predict environmental event from audio file"""
    try:
        # Save uploaded file temporarily
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        
        file_path = temp_dir / f"{uuid.uuid4()}_{file.filename}"
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Run prediction
        result = predictor.predict(str(file_path))
        
        # Create alert if needed
        alert_id = None
        severity = "info"
        if result.is_alert:
            alert = alert_system.create_alert(
                prediction_result={
                    'class_name': result.class_name,
                    'confidence': result.confidence,
                    'features': result.features,
                    'audio_path': str(file_path)
                },
                location=location
            )
            alert_system.send_alert(alert)
            alert_id = alert.alert_id
            severity = alert.severity
        
        # Clean up temp file
        file_path.unlink(missing_ok=True)
        
        return PredictionResponse(
            alert_id=alert_id,
            event_type=result.class_name,
            confidence=result.confidence,
            severity=severity,
            is_alert=result.is_alert,
            timestamp=result.timestamp,
            location=location
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    stats = predictor.get_performance_stats()
    return {
        "predictions": stats,
        "alerts": {
            "total": len(alert_system.alert_history),
            "recent": alert_system.get_alert_summary(hours=24)
        }
    }


@app.get("/alerts/history")
async def get_alert_history(hours: Optional[int] = 24):
    """Get alert history"""
    summary = alert_system.get_alert_summary(hours=hours)
    return {
        **summary,
        "recent_alerts": [
            {
                "id": a.alert_id,
                "time": a.timestamp,
                "event": a.event_type,
                "severity": a.severity,
                "location": a.location,
                "confidence": a.confidence
            }
            for a in alert_system.alert_history[-10:]  # Last 10 alerts
        ]
    }


@app.post("/test_alert")
async def test_alert():
    """Send test alert"""
    test_alert = alert_system.create_alert(
        prediction_result={
            'class_name': 'test_event',
            'confidence': 0.95,
            'features': {'test': 'data'}
        },
        location="Test Location"
    )
    alert_system.send_alert(test_alert)
    
    return {
        "message": "Test alert sent",
        "alert_id": test_alert.alert_id
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)