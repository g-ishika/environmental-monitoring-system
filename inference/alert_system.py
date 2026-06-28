import smtplib
import requests
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
import time
import hashlib

from utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class Alert:
    
    alert_id: str
    timestamp: float
    severity: str
    event_type: str
    location: str
    confidence: float
    details: Dict
    audio_path: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)


class AlertSystem:
    
    
    def __init__(self, config: Dict):
        self.config = config
        self.alert_history = []
        self.max_history = config.get('max_history', 1000)
        
        
        self.channels = config.get('alerting', {}).get('channels', ['email', 'slack'])
        self.severity_levels = config.get('alerting', {}).get('severity_levels', ['info', 'warning', 'critical'])
        
        
        self.credentials = self.load_credentials()
        
        
        self.cooldown_period = config.get('alerting', {}).get('cooldown', 300)
        self.last_alert_time = {}
        
        
        self.alert_counts = {}
        self.last_reset = time.time()
    
    def load_credentials(self) -> Dict:
        """Load credentials from secure file"""
        cred_path = Path('config/credentials.json')
        if cred_path.exists():
            try:
                with open(cred_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load credentials: {e}")
        return {}
    
    def generate_alert_id(self, event_type: str, location: str) -> str:
        """Generate unique alert ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_str = f"{event_type}_{location}_{timestamp}_{time.time()}"
        hash_id = hashlib.md5(unique_str.encode()).hexdigest()[:8]
        return f"ALERT_{timestamp}_{hash_id}"
    
    def check_cooldown(self, event_type: str, location: str) -> bool:
        """Check if event is in cooldown period"""
        key = f"{event_type}_{location}"
        current_time = time.time()
        
        if key in self.last_alert_time:
            if current_time - self.last_alert_time[key] < self.cooldown_period:
                return True
        
        self.last_alert_time[key] = current_time
        return False
    
    def check_rate_limit(self, event_type: str) -> bool:
        """Check rate limiting for alerts"""
        
        if time.time() - self.last_reset > 3600:
            self.alert_counts = {}
            self.last_reset = time.time()
        
        
        if self.alert_counts.get(event_type, 0) >= 10:
            return False
        
        self.alert_counts[event_type] = self.alert_counts.get(event_type, 0) + 1
        return True
    
    def determine_severity(self, confidence: float) -> str:
        """Determine alert severity based on confidence"""
        if confidence >= 0.95:
            return "critical"
        elif confidence >= 0.85:
            return "warning"
        else:
            return "info"
    
    def create_alert(self, prediction_result: Dict, location: str = "Unknown") -> Alert:
        """Create alert from prediction result"""
        confidence = prediction_result.get('confidence', 0.0)
        event_type = prediction_result.get('class_name', 'unknown')
        
        alert = Alert(
            alert_id=self.generate_alert_id(event_type, location),
            timestamp=time.time(),
            severity=self.determine_severity(confidence),
            event_type=event_type,
            location=location,
            confidence=confidence,
            details=prediction_result.get('features', {}),
            audio_path=prediction_result.get('audio_path', '')
        )
        
        # storing in histry
        self.alert_history.append(alert)
        if len(self.alert_history) > self.max_history:
            self.alert_history = self.alert_history[-self.max_history:]
        
        #i forgot what i was doing, hoping it works
        self.log_alert(alert)
        
        return alert
    
    def send_email_alert(self, alert: Alert):
        """Send email alert"""
        if 'email' not in self.channels:
            return
        
        try:
            smtp_config = self.credentials.get('smtp', {})
            if not smtp_config:
                logger.warning("SMTP credentials not found")
                return
            
            # Creatin message
            msg = MIMEMultipart()
            msg['From'] = smtp_config.get('from_email')
            msg['To'] = smtp_config.get('to_email')
            msg['Subject'] = f"[{alert.severity.upper()}] Environmental Alert: {alert.event_type}"
            
            body = self._format_alert_body(alert)
            msg.attach(MIMEText(body, 'plain'))
            
            # Sending email
            with smtplib.SMTP(smtp_config.get('server', 'smtp@gmail.com'), 
                             smtp_config.get('port', 587)) as server:
                server.starttls()
                server.login(smtp_config.get('username'), smtp_config.get('password'))
                server.send_message(msg)
            
            logger.info(f"Email alert sent: {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    def send_slack_alert(self, alert: Alert):
        """Send Slack alert"""
        if 'slack' not in self.channels:
            return
        
        try:
            slack_config = self.credentials.get('slack', {})
            if not slack_config:
                logger.warning("Slack credentials not found")
                return
            
            webhook_url = slack_config.get('webhook_url')
            if not webhook_url:
                return
            
            # some calligraphy
            color = {
                'info': '#3498db',
                'warning': '#f39c12',
                'critical': '#e74c3c'
            }.get(alert.severity, '#95a5a6')
            
            message = self._format_slack_message(alert, color)
            
            
            response = requests.post(webhook_url, json=message)
            if response.status_code == 200:
                logger.info(f"Slack alert sent: {alert.alert_id}")
            else:
                logger.error(f"Slack alert failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
    
    def send_alert(self, alert: Alert):
        """Send alert through all configured channels"""
        # Check cooldown
        if self.check_cooldown(alert.event_type, alert.location):
            logger.debug(f"Alert in cooldown: {alert.event_type}")
            return
        
        # Check rate limit
        if not self.check_rate_limit(alert.event_type):
            logger.warning(f"Rate limit exceeded for {alert.event_type}")
            return
        
        logger.info(f"Sending alert: {alert.alert_id} - {alert.event_type}")
        
        # Send through channels
        if 'email' in self.channels:
            self.send_email_alert(alert)
        
        if 'slack' in self.channels:
            self.send_slack_alert(alert)
        
    
        self.log_alert(alert)
    
    def _format_alert_body(self, alert: Alert) -> str:
        """Format alert email body"""
        return f"""
        ALERT DETAILS
        
        Alert ID: {alert.alert_id}
        Severity: {alert.severity.upper()}
        Event: {alert.event_type}
        Location: {alert.location}
        Confidence: {alert.confidence:.2%}
        Time: {datetime.fromtimestamp(alert.timestamp).strftime('%Y-%m-%d %H:%M:%S')}
        
        FEATURES
    
        {json.dumps(alert.details, indent=2)}
        
        AUDIO
        
        {alert.audio_path if alert.audio_path else "No audio file available"}
        
        This is an automated alert from the Environmental Monitoring System.
        Please investigate immediately.
        """
    
    def _format_slack_message(self, alert: Alert, color: str) -> Dict:
        """Format Slack message"""
        return {
            "attachments": [{
                "color": color,
                "title": f"🚨 {alert.severity.upper()}: {alert.event_type}",
                "fields": [
                    {"title": "Alert ID", "value": alert.alert_id, "short": True},
                    {"title": "Location", "value": alert.location, "short": True},
                    {"title": "Confidence", "value": f"{alert.confidence:.2%}", "short": True},
                    {
                        "title": "Time", 
                        "value": datetime.fromtimestamp(alert.timestamp).strftime('%Y-%m-%d %H:%M:%S'),
                        "short": False
                    }
                ],
                "footer": "Environmental Monitoring System",
                "ts": int(alert.timestamp)
            }]
        }
    
    def log_alert(self, alert: Alert):
        """Log alert to file"""
        log_path = Path('logs/alerts.log')
        log_path.parent.mkdir(exist_ok=True)
        
        with open(log_path, 'a') as f:
            log_entry = {
                'timestamp': alert.timestamp,
                'alert_id': alert.alert_id,
                'severity': alert.severity,
                'event': alert.event_type,
                'location': alert.location,
                'confidence': alert.confidence,
                'details': alert.details
            }
            f.write(json.dumps(log_entry) + '\n')
    
    def get_alert_summary(self, hours: int = 24) -> Dict:
        
        current_time = time.time()
        cutoff_time = current_time - (hours * 3600)
        
        recent_alerts = [
            alert for alert in self.alert_history
            if alert.timestamp >= cutoff_time
        ]
        
        summary = {
            'total_alerts': len(recent_alerts),
            'by_severity': {},
            'by_event_type': {},
            'by_location': {},
            'time_range': f"Last {hours} hours"
        }
        
        for alert in recent_alerts:
        
            summary['by_severity'][alert.severity] = \
                summary['by_severity'].get(alert.severity, 0) + 1
            
            
            summary['by_event_type'][alert.event_type] = \
                summary['by_event_type'].get(alert.event_type, 0) + 1
            
            
            summary['by_location'][alert.location] = \
                summary['by_location'].get(alert.location, 0) + 1
        
        return summary