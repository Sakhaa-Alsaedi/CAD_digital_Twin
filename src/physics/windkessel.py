"""
Enhanced Windkessel Model Implementation

This module provides improved implementations of Windkessel models for 
cardiac hemodynamics simulation with better error handling, validation,
and extensibility.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from scipy.integrate import odeint
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class WindkesselParameters:
    """
    Data class for Windkessel model parameters.
    
    Attributes:
        Rs: Systemic resistance (mmHg·s/ml)
        Rm: Mitral valve resistance (mmHg·s/ml)
        Ra: Aortic valve resistance (mmHg·s/ml)
        Rc: Characteristic resistance (mmHg·s/ml)
        Ca: Aortic compliance (ml/mmHg)
        Cs: Systemic compliance (ml/mmHg)
        Cr: Venous compliance (ml/mmHg)
        Ls: Aortic inductance (mmHg·s²/ml)
        Emax: Maximum elastance (mmHg/ml)
        Emin: Minimum elastance (mmHg/ml)
        Tc: Cardiac cycle time (s)
        Vd: Dead volume (ml)
    """
    Rs: float = 1.0
    Rm: float = 0.005
    Ra: float = 0.001
    Rc: float = 0.0398
    Ca: float = 0.08
    Cs: float = 1.33
    Cr: float = 4.400
    Ls: float = 0.0005
    Emax: float = 2.0
    Emin: float = 0.02
    Tc: float = 1.0
    Vd: float = 10.0
    
    def validate(self) -> bool:
        """Validate parameter ranges."""
        validations = [
            (self.Rs > 0, "Rs must be positive"),
            (self.Rm > 0, "Rm must be positive"),
            (self.Ra > 0, "Ra must be positive"),
            (self.Rc > 0, "Rc must be positive"),
            (self.Ca > 0, "Ca must be positive"),
            (self.Cs > 0, "Cs must be positive"),
            (self.Cr > 0, "Cr must be positive"),
            (self.Ls > 0, "Ls must be positive"),
            (self.Emax > self.Emin, "Emax must be greater than Emin"),
            (self.Tc > 0, "Tc must be positive"),
            (self.Vd >= 0, "Vd must be non-negative")
        ]
        
        for condition, message in validations:
            if not condition:
                logger.error(f"Parameter validation failed: {message}")
                return False
        return True


class WindkesselModel:
    """
    Enhanced Windkessel model for cardiac hemodynamics simulation.
    
    This class provides a comprehensive implementation of the Windkessel model
    with improved numerical stability, parameter validation, and extensibility.
    """
    
    def __init__(self, parameters: Optional[WindkesselParameters] = None):
        """
        Initialize the Windkessel model.
        
        Args:
            parameters: WindkesselParameters object. If None, default parameters are used.
        """
        self.parameters = parameters or WindkesselParameters()
        if not self.parameters.validate():
            raise ValueError("Invalid parameters provided")
        
        logger.info("WindkesselModel initialized successfully")
    
    def elastance(self, t: float) -> float:
        """
        Calculate time-varying elastance function.
        
        Args:
            t: Time (s)
            
        Returns:
            Elastance value (mmHg/ml)
        """
        # Normalize time to cardiac cycle
        t_norm = t - int(t / self.parameters.Tc) * self.parameters.Tc
        tn = t_norm / (0.2 + 0.15 * self.parameters.Tc)
        
        # Double Hill function for elastance
        elastance_val = (
            (self.parameters.Emax - self.parameters.Emin) * 1.55 * 
            (tn / 0.7) ** 1.9 / ((tn / 0.7) ** 1.9 + 1) * 
            1 / ((tn / 1.17) ** 21.9 + 1) + self.parameters.Emin
        )
        
        return elastance_val
    
    def pressure_lv(self, volume_lv: float, t: float) -> float:
        """
        Calculate left ventricular pressure.
        
        Args:
            volume_lv: Left ventricular volume relative to dead volume (ml)
            t: Time (s)
            
        Returns:
            Left ventricular pressure (mmHg)
        """
        return self.elastance(t) * volume_lv
    
    @staticmethod
    def rectifier(u: float) -> float:
        """
        Rectifier function for valve behavior.
        
        Args:
            u: Input value
            
        Returns:
            Rectified value (max(0, u))
        """
        return max(0.0, u)
    
    def heart_ode(self, y: List[float], t: float) -> List[float]:
        """
        Cardiac hemodynamics ODE system.
        
        Args:
            y: State vector [V_lv, P_la, P_a, P_ao, Q_t]
            t: Time (s)
            
        Returns:
            Derivative vector dy/dt
        """
        x1, x2, x3, x4, x5 = y
        
        # Calculate left ventricular pressure
        P_lv = self.pressure_lv(x1, t)
        
        # Extract parameters for readability
        p = self.parameters
        
        # ODE system
        dydt = [
            self.rectifier(x2 - P_lv) / p.Rm - self.rectifier(P_lv - x4) / p.Ra,
            (x3 - x2) / (p.Rs * p.Cr) - self.rectifier(x2 - P_lv) / (p.Cr * p.Rm),
            (x2 - x3) / (p.Rs * p.Cs) + x5 / p.Cs,
            -x5 / p.Ca + self.rectifier(P_lv - x4) / (p.Ca * p.Ra),
            (x4 - x3 - p.Rc * x5) / p.Ls
        ]
        
        return dydt
    
    def simulate(self, 
                 initial_conditions: Optional[List[float]] = None,
                 n_cycles: int = 5,
                 time_points_per_cycle: int = 60000) -> Dict[str, np.ndarray]:
        """
        Simulate cardiac hemodynamics.
        
        Args:
            initial_conditions: Initial state [V_lv, P_la, P_a, P_ao, Q_t]
            n_cycles: Number of cardiac cycles to simulate
            time_points_per_cycle: Number of time points per cycle
            
        Returns:
            Dictionary containing simulation results
        """
        # Set default initial conditions if not provided
        if initial_conditions is None:
            start_v = 120.0  # Initial LV volume relative to Vd
            start_pla = start_v * self.elastance(0.0)
            start_pao = 75.0
            start_pa = start_pao
            start_qt = 0.0
            initial_conditions = [start_v, start_pla, start_pa, start_pao, start_qt]
        
        # Time vector
        t = np.linspace(0, self.parameters.Tc * n_cycles, 
                       int(time_points_per_cycle * n_cycles))
        
        try:
            # Solve ODE
            sol = odeint(self.heart_ode, initial_conditions, t)
            
            # Extract results
            V_lv = sol[:, 0] + self.parameters.Vd
            P_lv = np.array([self.pressure_lv(v, ti) for ti, v in zip(t, sol[:, 0])])
            P_la = sol[:, 1]
            P_a = sol[:, 2]
            P_ao = sol[:, 3]
            Q_t = sol[:, 4]
            
            # Calculate derived metrics
            results = {
                'time': t,
                'V_lv': V_lv,
                'P_lv': P_lv,
                'P_la': P_la,
                'P_a': P_a,
                'P_ao': P_ao,
                'Q_t': Q_t,
                'elastance': np.array([self.elastance(ti) for ti in t])
            }
            
            # Calculate clinical metrics for last cycle
            last_cycle_start = (n_cycles - 1) * time_points_per_cycle
            last_cycle_end = n_cycles * time_points_per_cycle
            
            ved = np.max(V_lv[last_cycle_start:last_cycle_end])
            ves = np.min(V_lv[last_cycle_start:last_cycle_end])
            ef = (ved - ves) / ved * 100.0
            
            results.update({
                'VED': ved,
                'VES': ves,
                'EF': ef,
                'stroke_volume': ved - ves
            })
            
            logger.info(f"Simulation completed: EF={ef:.1f}%, VED={ved:.1f}ml, VES={ves:.1f}ml")
            
            return results
            
        except Exception as e:
            logger.error(f"Simulation failed: {str(e)}")
            raise


class WindkesselSimulator:
    """
    High-level simulator for batch processing and parameter studies.
    """
    
    def __init__(self):
        """Initialize the simulator."""
        self.models = {}
        logger.info("WindkesselSimulator initialized")
    
    def create_parameter_grid(self, 
                            parameter_ranges: Dict[str, Tuple[float, float, int]]) -> List[WindkesselParameters]:
        """
        Create a grid of parameters for parameter studies.
        
        Args:
            parameter_ranges: Dictionary with parameter names as keys and 
                            (min, max, num_points) tuples as values
                            
        Returns:
            List of WindkesselParameters objects
        """
        import itertools
        
        # Create parameter arrays
        param_arrays = {}
        for param_name, (min_val, max_val, num_points) in parameter_ranges.items():
            param_arrays[param_name] = np.linspace(min_val, max_val, num_points)
        
        # Generate all combinations
        param_combinations = list(itertools.product(*param_arrays.values()))
        param_names = list(parameter_ranges.keys())
        
        # Create WindkesselParameters objects
        parameter_sets = []
        for combination in param_combinations:
            params_dict = dict(zip(param_names, combination))
            
            # Create parameters object with defaults and update with specific values
            params = WindkesselParameters()
            for name, value in params_dict.items():
                setattr(params, name, value)
            
            if params.validate():
                parameter_sets.append(params)
            else:
                logger.warning(f"Invalid parameter combination skipped: {params_dict}")
        
        logger.info(f"Created {len(parameter_sets)} valid parameter combinations")
        return parameter_sets
    
    def batch_simulate(self, 
                      parameter_sets: List[WindkesselParameters],
                      n_cycles: int = 5) -> List[Dict[str, np.ndarray]]:
        """
        Run batch simulations for multiple parameter sets.
        
        Args:
            parameter_sets: List of WindkesselParameters objects
            n_cycles: Number of cardiac cycles to simulate
            
        Returns:
            List of simulation results
        """
        results = []
        
        for i, params in enumerate(parameter_sets):
            try:
                model = WindkesselModel(params)
                result = model.simulate(n_cycles=n_cycles)
                results.append(result)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Completed {i + 1}/{len(parameter_sets)} simulations")
                    
            except Exception as e:
                logger.error(f"Simulation {i} failed: {str(e)}")
                results.append(None)
        
        successful_sims = sum(1 for r in results if r is not None)
        logger.info(f"Batch simulation completed: {successful_sims}/{len(parameter_sets)} successful")
        
        return results


# Neural network interpolator for parameter-to-output mapping
class ParameterInterpolator(nn.Module):
    """
    Neural network for interpolating between parameter sets and outputs.
    """
    
    def __init__(self, n_parameters: int, n_outputs: int, hidden_size: int = 256):
        """
        Initialize the interpolator network.
        
        Args:
            n_parameters: Number of input parameters
            n_outputs: Number of output values
            hidden_size: Size of hidden layers
        """
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(n_parameters, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_outputs)
        ).double()
        
        self.n_parameters = n_parameters
        self.n_outputs = n_outputs
        
        logger.info(f"ParameterInterpolator initialized: {n_parameters} -> {n_outputs}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(x)
    
    def train_interpolator(self, 
                          parameters: torch.Tensor, 
                          outputs: torch.Tensor,
                          epochs: int = 30000,
                          learning_rate: float = 0.01,
                          validation_split: float = 0.2) -> Dict[str, List[float]]:
        """
        Train the interpolator network.
        
        Args:
            parameters: Input parameter tensor
            outputs: Target output tensor
            epochs: Number of training epochs
            learning_rate: Learning rate
            validation_split: Fraction of data for validation
            
        Returns:
            Training history dictionary
        """
        # Split data
        n_samples = len(parameters)
        n_val = int(n_samples * validation_split)
        indices = torch.randperm(n_samples)
        
        train_params = parameters[indices[n_val:]]
        train_outputs = outputs[indices[n_val:]]
        val_params = parameters[indices[:n_val]]
        val_outputs = outputs[indices[:n_val]]
        
        # Setup training
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training
            self.train()
            optimizer.zero_grad()
            train_pred = self(train_params)
            train_loss = criterion(train_pred, train_outputs)
            train_loss.backward()
            optimizer.step()
            
            # Validation
            if epoch % 1000 == 0:
                self.eval()
                with torch.no_grad():
                    val_pred = self(val_params)
                    val_loss = criterion(val_pred, val_outputs)
                
                train_losses.append(train_loss.item())
                val_losses.append(val_loss.item())
                
                logger.info(f"Epoch {epoch}: Train Loss = {train_loss.item():.6f}, "
                          f"Val Loss = {val_loss.item():.6f}")
        
        return {'train_losses': train_losses, 'val_losses': val_losses}

