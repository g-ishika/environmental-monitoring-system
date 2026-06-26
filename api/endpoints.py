"""API endpoints for the environmental monitoring system"""

from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import Optional, List
import torch
import numpy as np
from pathlib import Path
import uuid
import json

from inference.predictor import Predictor
from inference.alert_system import AlertSystem
from utils.logger import setup_logger

logger = setup_logger(__name__)

router = APIRouter()

# These will be initialized in app.py
predictor = None
alert_system = None


@router.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Environmental Monitoring API",
        "status": "active",
        "endpoints": [
            "/predict",
            "/stats",
            "/alerts/history",
            "/test_alert"
        ]
    }


@router.post("/predict")
async def predict_audio(
    file: UploadFile = File(...),
    location: str = Query("Unknown", description="Location of the recording")
):
    """Predict environmental event from audio file"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    
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
        
        return {
            "success": True,
            "alert_id": alert_id,
            "event_type": result.class_name,
            "confidence": result.confidence,
            "severity": severity,
            "is_alert": result.is_alert,
            "timestamp": result.timestamp,
            "location": location,
            "all_probabilities": result.all_probs.tolist() if result.all_probs is not None else None
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_stats():
    """Get system statistics"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    
    stats = predictor.get_performance_stats()
    alert_summary = alert_system.get_alert_summary(hours=24) if alert_system else {}
    
    return {
        "predictions": stats,
        "alerts": {
            "total": len(alert_system.alert_history) if alert_system else 0,
            "recent_24h": alert_summary
        },
        "system_status": "running"
    }


@router.get("/alerts/history")
async def get_alert_history(
    hours: Optional[int] = Query(24, description="Number of hours to look back"),
    limit: Optional[int] = Query(50, description="Maximum number of alerts to return")
):
    """Get alert history"""
    if alert_system is None:
        raise HTTPException(status_code=503, detail="Alert system not initialized")
    
    summary = alert_system.get_alert_summary(hours=hours)
    
    # Get recent alerts
    recent_alerts = []
    for alert in alert_system.alert_history[-limit:]:
        recent_alerts.append({
            "id": alert.alert_id,
            "time": alert.timestamp,
            "event": alert.event_type,
            "severity": alert.severity,
            "location": alert.location,
            "confidence": alert.confidence
        })
    
    return {
        **summary,
        "recent_alerts": recent_alerts
    }


@router.post("/test_alert")
async def test_alert():
    """Send test alert"""
    if alert_system is None:
        raise HTTPException(status_code=503, detail="Alert system not initialized")
    
    test_alert = alert_system.create_alert(
        prediction_result={
            'class_name': 'test_event',
            'confidence': 0.95,
            'features': {'test': 'data', 'test_id': str(uuid.uuid4())}
        },
        location="Test Location"
    )
    alert_system.send_alert(test_alert)
    
    return {
        "message": "Test alert sent",
        "alert_id": test_alert.alert_id,
        "severity": test_alert.severity
    }


@router.get("/classes")
async def get_classes():
    """Get list of classes the model can detect"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    
    return {
        "classes": predictor.classes,
        "num_classes": len(predictor.classes)
    }