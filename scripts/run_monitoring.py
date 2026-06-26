#!/usr/bin/env python
import argparse
import yaml
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from pipelines.monitoring_pipeline import MonitoringPipeline
from utils.logger import setup_logger

logger = setup_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Run environmental monitoring system'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['batch', 'realtime', 'test'],
        default='test',
        help='Operation mode'
    )
    parser.add_argument(
        '--audio_dir',
        type=str,
        default='audio_data/raw',
        help='Directory containing audio files for batch processing'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='models_checkpoints/best_model.pt',
        help='Path to trained model'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override model path
    config['model_path'] = args.model_path
    
    # Initialize pipeline
    pipeline = MonitoringPipeline(config)
    
    if args.mode == 'batch' and args.audio_dir:
        # Batch processing mode
        audio_files = list(Path(args.audio_dir).glob('*.wav'))
        audio_files += list(Path(args.audio_dir).glob('*.mp3'))
        
        if not audio_files:
            logger.error(f"No audio files found in {args.audio_dir}")
            return
        
        logger.info(f"Processing {len(audio_files)} audio files...")
        results = pipeline.run_batch_inference(audio_files)
        
        # Print summary
        alerts = [r for r in results if r.get('is_alert', False)]
        logger.info(f"Results: {len(results)} processed, {len(alerts)} alerts triggered")
        
    elif args.mode == 'realtime':
        # Real-time monitoring mode
        logger.info("Starting real-time monitoring...")
        pipeline.monitor.start()
        
        try:
            while True:
                time.sleep(1)
                # Print performance every 10 seconds
                if int(time.time()) % 10 == 0:
                    report = pipeline.generate_report()
                    logger.info(f"Status: {report['performance']['status']}")
                    logger.info(f"Alerts triggered: {report['performance']['alerts_triggered']}")
                    
        except KeyboardInterrupt:
            logger.info("Stopping monitoring...")
            pipeline.monitor.stop()
    
    else:
        # Test mode
        logger.info("Running in test mode...")
        report = pipeline.generate_report()
        print("\n" + "="*50)
        print("ENVIRONMENTAL MONITORING SYSTEM - TEST REPORT")
        print("="*50)
        
        # Get performance data with safe defaults
        performance = report.get('performance', {})
        
        print(f"Status: {performance.get('status', 'unknown')}")
        print(f"Processed chunks: {performance.get('processed_windows', 0)}")
        print(f"Alerts triggered: {performance.get('alerts_triggered', 0)}")
        print(f"Alerts per hour: {performance.get('alerts_per_hour', 0):.2f}")
        
        # Check inference stats
        if 'inference_stats' in performance:
            stats = performance['inference_stats']
            print(f"Average inference time: {stats.get('mean_time', 0)*1000:.2f}ms")
            print(f"FPS: {stats.get('fps', 0):.2f}")
        
        # Show alert summary if available
        if 'alerts' in report:
            alert_summary = report['alerts']
            print(f"\nAlert Summary (24h):")
            print(f"  Total alerts: {alert_summary.get('total_alerts', 0)}")
            if 'by_event_type' in alert_summary:
                print(f"  By event type: {alert_summary['by_event_type']}")
        
        print("="*50)


if __name__ == "__main__":
    main()