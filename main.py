#!/usr/bin/env python3

import argparse
import os
from controller.realtime_controller import RealTimeController
from record.record_chords import ChordRecorder
from model.train import ModelTrainer

def main():
    parser = argparse.ArgumentParser(description='Ukulele Subway Surfers Controller')
    parser.add_argument('--mode', choices=['record', 'train', 'play'], required=True,
                       help='Mode: record samples, train model, or play game')
    parser.add_argument('--detection', choices=['model', 'pitch'], default='model',
                       help='Detection method: trained model or pitch detection')
    parser.add_argument('--label', type=str, help='Chord label for recording mode')
    parser.add_argument('--duration', type=int, default=2, help='Recording duration per sample')
    parser.add_argument('--samples', type=int, default=50, help='Number of samples to record')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--debug', action='store_true', help='Enable debug visualizations')
    
    #python main.py --mode train --epochs 150
    #python main.py --mode record --label C_major --samples 30
    #python main.py --mode train
    #python main.py --mode play --detection model
    
    
    
    args = parser.parse_args()
    
    if args.mode == 'record':
        if not args.label:
            print("Error: --label required for recording mode")
            return
        recorder = ChordRecorder()
        recorder.record_samples(args.label, args.samples, args.duration)
        
    elif args.mode == 'train':
        trainer = ModelTrainer()
        trainer.train(epochs=args.epochs)
        
    elif args.mode == 'play':
        controller = RealTimeController(
            detection_mode=args.detection,
            debug=args.debug
        )
        controller.start()

if __name__ == "__main__":
    main()