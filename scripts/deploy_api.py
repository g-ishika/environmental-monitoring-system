#!/usr/bin/env python
"""Deploy the environmental monitoring API"""

import argparse
import uvicorn
import yaml
from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description='Deploy Environmental Monitoring API')
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Host to bind to'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Port to bind to'
    )
    parser.add_argument(
        '--reload',
        action='store_true',
        help='Enable auto-reload for development'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Number of worker processes'
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 60)
    print("ENVIRONMENTAL MONITORING API")
    print("=" * 60)
    print(f"Configuration: {args.config}")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Workers: {args.workers}")
    print(f"Model: {config.get('model_path', 'models_checkpoints/best_model.pt')}")
    print(f"Device: {config.get('inference', {}).get('device', 'cuda')}")
    print("=" * 60)
    print("API Documentation: http://localhost:8000/docs")
    print("=" * 60)
    
    # Run the API
    uvicorn.run(
        "api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers
    )


if __name__ == "__main__":
    main()