"""End-to-end monitoring pipeline"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
import json
from datetime import datetime

from inference.predictor import Predictor
from inference.alert_system import AlertSystem
from inference.realtime_monitor import RealTimeMonitor
from utils.logger import setup_logger

logger = setup_logger(__name__)


class MonitoringPipeline:
    """Complete monitoring pipeline for environmental audio"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Initialize components
        self.predictor = Predictor(
            model_path=config['model_path'],
            config=config
        )
        
        self.alert_system = AlertSystem(config)
        
        self.monitor = RealTimeMonitor(
            predictor=self.predictor,
            alert_system=self.alert_system,
            config=config.get('inference', {}),
            device=config.get('inference', {}).get('device', 'cuda')
        )
        
        logger.info("Monitoring pipeline initialized")
    
    def _tensor_to_value(self, tensor):
        """Convert tensor to JSON-serializable value"""
        if isinstance(tensor, torch.Tensor):
            if tensor.numel() == 1:
                return tensor.item()
            else:
                return tensor.cpu().numpy().tolist()
        elif isinstance(tensor, np.ndarray):
            return tensor.tolist()
        elif isinstance(tensor, (np.int64, np.int32, np.int16, np.int8)):
            return int(tensor)
        elif isinstance(tensor, (np.float64, np.float32, np.float16)):
            return float(tensor)
        else:
            return tensor
    
    def run_batch_inference(self, audio_files: List[Union[str, Path]]) -> List[Dict]:
        """Run batch inference on multiple audio files"""
        results = []
        
        for audio_path in audio_files:
            try:
                result = self.predictor.predict(str(audio_path))
                
                # Convert features to JSON-serializable format
                features_dict = {}
                for k, v in result.features.items():
                    features_dict[k] = self._tensor_to_value(v)
                
                result_dict = {
                    'file': str(audio_path),
                    'prediction': result.class_name,
                    'confidence': result.confidence,
                    'is_alert': result.is_alert,
                    'timestamp': result.timestamp,
                    'features': features_dict
                }
                
                # Send alert if needed
                if result.is_alert:
                    alert = self.alert_system.create_alert(
                        prediction_result=result_dict,
                        location=self.config.get('location', 'monitoring_site')
                    )
                    self.alert_system.send_alert(alert)
                    result_dict['alert_id'] = alert.alert_id
                    result_dict['severity'] = alert.severity
                
                results.append(result_dict)
                logger.info(f" {Path(audio_path).name}: {result.class_name} ({result.confidence:.2%})")
                
            except Exception as e:
                logger.error(f"Error processing {audio_path}: {e}")
                results.append({
                    'file': str(audio_path),
                    'error': str(e)
                })
        
        return results
    
    def start_realtime_monitoring(self, audio_source: Optional[callable] = None):
        """Start real-time monitoring"""
        self.monitor.start(audio_source)
    
    def stop_realtime_monitoring(self):
        """Stop real-time monitoring"""
        self.monitor.stop()
    
    def generate_report(self) -> Dict:
        """Generate comprehensive monitoring report"""
        performance = self.monitor.get_performance_report()
        alert_summary = self.alert_system.get_alert_summary(hours=24)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'system_info': {
                'version': self.config.get('version', '1.0.0'),
                'location': self.config.get('location', 'Unknown'),
                'model': self.config.get('model', {}).get('architecture', 'resnet50')
            },
            'performance': performance,
            'alerts': alert_summary,
            'predictor_stats': self.predictor.get_performance_stats()
        }
    
    def save_report(self, report: Optional[Dict] = None, path: Optional[str] = None):
        """Save report to file"""
        if report is None:
            report = self.generate_report()
        
        if path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            path = f"outputs/report_{timestamp}.json"
        
        path = Path(path)
        path.parent.mkdir(exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report saved to {path}")
        return path