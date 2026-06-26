"""Model optimization utilities"""

import torch
import torch.nn.utils.prune as prune
import numpy as np
from typing import Dict, Optional, Tuple
import warnings


class ModelOptimizer:
    """Model optimization for inference efficiency"""
    
    def __init__(self, model: torch.nn.Module):
        self.model = model
    
    def prune_model(self, amount: float = 0.2, method: str = 'l1_unstructured'):
        """Prune model weights"""
        parameters_to_prune = []
        
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                parameters_to_prune.append((module, 'weight'))
        
        if method == 'l1_unstructured':
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=amount
            )
        else:
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.RandomUnstructured,
                amount=amount
            )
        
        # Remove pruning masks
        for module, _ in parameters_to_prune:
            prune.remove(module, 'weight')
        
        return self.model
    
    def quantize_model(self, quant_type: str = 'dynamic') -> torch.nn.Module:
        """Quantize model for faster inference"""
        if quant_type == 'dynamic':
            # Dynamic quantization
            self.model.eval()
            self.model = torch.quantization.quantize_dynamic(
                self.model,
                {torch.nn.Linear, torch.nn.Conv2d},
                dtype=torch.qint8
            )
        else:
            # Static quantization (requires calibration)
            self.model.eval()
            self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            torch.quantization.prepare(self.model, inplace=True)
            torch.quantization.convert(self.model, inplace=True)
        
        return self.model
    
    def convert_to_onnx(self, input_shape: Tuple[int, ...], onnx_path: str = "model.onnx"):
        """Convert model to ONNX format"""
        self.model.eval()
        dummy_input = torch.randn(1, *input_shape)
        
        torch.onnx.export(
            self.model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        return onnx_path
    
    def convert_to_torchscript(self, script_path: str = "model.pt"):
        """Convert model to TorchScript"""
        self.model.eval()
        
        # Script the model
        scripted_model = torch.jit.script(self.model)
        
        # Save
        scripted_model.save(script_path)
        
        return script_path
    
    def get_model_size(self) -> Dict:
        """Get model size information"""
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        
        # Estimate model size in MB
        param_size = total_params * 4 / (1024 * 1024)  # 4 bytes per float
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': non_trainable_params,
            'estimated_size_mb': param_size
        }
    
    def optimize_for_inference(self):
        """Apply multiple optimizations for inference"""
        # Prune
        self.prune_model(amount=0.1)
        
        # Convert to eval mode
        self.model.eval()
        
        # Fuse operations
        self.model = torch.quantization.fuse_modules(self.model, [['conv1', 'bn1']])
        
        return self.model


class ModelCompressor:
    """Model compression utilities"""
    
    @staticmethod
    def apply_knowledge_distillation(
        teacher_model: torch.nn.Module,
        student_model: torch.nn.Module,
        train_loader,
        epochs: int = 10,
        temperature: float = 3.0
    ):
        """Apply knowledge distillation"""
        import torch.nn.functional as F
        
        teacher_model.eval()
        student_model.train()
        
        optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)
        
        for epoch in range(epochs):
            for data, targets in train_loader:
                # Forward pass teacher
                with torch.no_grad():
                    teacher_outputs = teacher_model(data)
                    teacher_probs = F.softmax(teacher_outputs / temperature, dim=1)
                
                # Forward pass student
                student_outputs = student_model(data)
                student_probs = F.log_softmax(student_outputs / temperature, dim=1)
                
                # Distillation loss
                distill_loss = F.kl_div(student_probs, teacher_probs, reduction='batchmean')
                
                # Student loss
                student_loss = F.cross_entropy(student_outputs, targets)
                
                # Combined loss
                loss = 0.7 * distill_loss + 0.3 * student_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        return student_model
    
    @staticmethod
    def apply_weight_sharing(model: torch.nn.Module, num_clusters: int = 16):
        """Apply weight sharing for compression"""
        with torch.no_grad():
            for param in model.parameters():
                if param.requires_grad:
                    # Flatten and cluster
                    flat = param.data.flatten()
                    centroids = torch.linspace(flat.min(), flat.max(), num_clusters)
                    
                    # Assign to nearest centroid
                    distances = torch.cdist(flat.unsqueeze(1), centroids.unsqueeze(1))
                    assignments = torch.argmin(distances, dim=1)
                    
                    # Replace with centroids
                    param.data = centroids[assignments].reshape(param.shape)
        
        return model