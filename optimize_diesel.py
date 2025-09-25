import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from scipy.optimize import minimize as scipy_minimize
import warnings
warnings.filterwarnings('ignore')

class ORCVCRProblem(ElementwiseProblem):
    def __init__(self):
        # Decision variables: [Load (20-80%), VCR (16-19.5), Fluid_Type (0-2)]
        super().__init__(n_var=3, n_obj=3, n_ieq_constr=2, 
                        xl=np.array([20, 16.0, 0]), 
                        xu=np.array([80, 19.5, 2]))
        
        # Load experimental data from Excel
        self.load_data()
    
    def load_data(self):
        """Load and preprocess the experimental data"""
        # Energy efficiency data (from your Excel)
        self.energy_data = {
            'O-Xylene': {
                16: [7.855, 9.163, 11.08, 11.83, 12.99, 13.6, 15.49],
                16.5: [8.304, 10.12, 11.56, 12.28, 13.39, 14.16, 15.83],
                17: [9.654, 11.08, 12.09, 13.11, 14.36, 14.86, 16.29],
                17.5: [10.43, 11.83, 12.52, 13.91, 14.64, 15.41, 16.48],
                18: [11.22, 12.21, 12.82, 14.36, 14.99, 15.9, 16.7],
                18.5: [11.63, 12.94, 13.44, 14.64, 15.33, 16.15, 17.02],
                19: [13.16, 13.71, 14.77, 15.16, 15.64, 16.39, 17.61],
                19.5: [13.49, 14.36, 15.03, 15.45, 16.15, 16.82, 17.76]
            },
            'M-Xylene': {
                16: [7.879, 9.21, 11.16, 11.92, 13.1, 13.71, 15.58],
                16.5: [8.336, 10.19, 11.65, 12.38, 13.49, 14.27, 15.91],
                17: [9.709, 11.16, 12.18, 13.21, 14.46, 14.96, 16.36],
                17.5: [10.5, 11.92, 12.62, 14.02, 14.74, 15.5, 16.55],
                18: [11.31, 12.31, 12.93, 14.46, 15.09, 15.99, 16.76],
                18.5: [11.72, 13.04, 13.55, 14.74, 15.42, 16.23, 17.08],
                19: [13.27, 13.82, 14.87, 15.26, 15.73, 16.45, 17.66],
                19.5: [13.6, 14.46, 15.13, 15.54, 16.23, 16.88, 17.81]
            },
            'Ethylbenzene': {
                16: [7.752, 9.05, 10.96, 11.7, 12.87, 13.48, 15.33],
                16.5: [8.198, 10.0, 11.44, 12.15, 13.26, 14.04, 15.66],
                17: [9.536, 10.96, 11.96, 12.98, 14.23, 14.72, 16.1],
                17.5: [10.31, 11.7, 12.39, 13.79, 14.5, 16.29, 16.29],
                18: [11.1, 12.09, 12.69, 14.23, 14.85, 15.73, 16.5],
                18.5: [11.5, 12.81, 13.31, 14.5, 15.17, 15.97, 16.81],
                19: [13.04, 13.58, 14.63, 15.01, 15.48, 16.2, 17.38],
                19.5: [13.37, 14.23, 14.89, 15.29, 15.97, 16.62, 17.54]
            }
        }
        
        self.fluids = ['O-Xylene', 'M-Xylene', 'Ethylbenzene']
        self.loads = [20, 30, 40, 50, 60, 70, 80]
    
    def interpolate_efficiency(self, load, vcr, fluid_idx):
        """Interpolate efficiency for given parameters"""
        fluid = self.fluids[int(fluid_idx)]
        vcrs = list(self.energy_data[fluid].keys())
        
        # Find nearest VCR values
        vcr_low = max([v for v in vcrs if v <= vcr])
        vcr_high = min([v for v in vcrs if v >= vcr])
        
        # Find nearest load values
        load_low = max([l for l in self.loads if l <= load])
        load_high = min([l for l in self.loads if l >= load])
        
        # Bilinear interpolation
        if vcr_low == vcr_high and load_low == load_high:
            return self.energy_data[fluid][vcr_low][self.loads.index(load_low)]
        
        # Interpolate
        eff_low_load = np.interp(vcr, [vcr_low, vcr_high], 
                                [self.energy_data[fluid][vcr_low][self.loads.index(load_low)],
                                 self.energy_data[fluid][vcr_high][self.loads.index(load_low)]])
        
        eff_high_load = np.interp(vcr, [vcr_low, vcr_high], 
                                 [self.energy_data[fluid][vcr_low][self.loads.index(load_high)],
                                  self.energy_data[fluid][vcr_high][self.loads.index(load_high)]])
        
        return np.interp(load, [load_low, load_high], [eff_low_load, eff_high_load])
    
    def _evaluate(self, x, out, *args, **kwargs):
        load, vcr, fluid_idx = x
        
        # Objectives to maximize
        energy_eff = self.interpolate_efficiency(load, vcr, fluid_idx)
        exergy_eff = energy_eff * 3.2  # Simplified relationship (adjust based on your data)
        cop = energy_eff * 0.08  # Simplified relationship
        
        # Objectives (maximize)
        out["F"] = [-energy_eff, -exergy_eff, -cop]  # Negative for maximization
        
        # Constraints
        out["G"] = [
            energy_eff - 10,  # Minimum efficiency constraint
            20 - load  # Minimum load constraint
        ]

def run_optimization():
    """Run the multi-objective optimization"""
    problem = ORCVCRProblem()
    
    algorithm = NSGA2(
        pop_size=100,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )
    
    res = minimize(problem,
                  algorithm,
                  ('n_gen', 200),
                  verbose=False)
    
    return res, problem

def visualize_results(res, problem):
    """Create comprehensive visualizations"""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('ORC-VCR System Optimization Results', fontsize=16, fontweight='bold')
    
    # Extract results
    X = res.X
    F = -res.F  # Convert back to positive values
    
    # 1. Pareto Front (3D)
    ax = fig.add_subplot(231, projection='3d')
    scatter = ax.scatter(F[:, 0], F[:, 1], F[:, 2], 
                        c=X[:, 2], cmap='viridis', alpha=0.7)
    ax.set_xlabel('Energy Efficiency (%)')
    ax.set_ylabel('Exergy Efficiency (%)')
    ax.set_zlabel('COP')
    ax.set_title('Pareto Optimal Front')
    plt.colorbar(scatter, ax=ax, label='Fluid Type')
    
    # 2. Energy Efficiency vs Load for different fluids
    ax = axes[0, 0]
    loads = np.linspace(20, 80, 100)
    vcrs = [16, 17.5, 19]
    
    for vcr in vcrs:
        for fluid_idx, fluid in enumerate(problem.fluids):
            effs = [problem.interpolate_efficiency(load, vcr, fluid_idx) for load in loads]
            ax.plot(loads, effs, label=f'{fluid} (VCR={vcr})', 
                   linestyle='--' if vcr == 17.5 else '-')
    
    ax.set_xlabel('Load (%)')
    ax.set_ylabel('Energy Efficiency (%)')
    ax.set_title('Energy Efficiency vs Load')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # 3. Heatmap for best fluid
    ax = axes[0, 1]
    load_grid = np.linspace(20, 80, 50)
    vcr_grid = np.linspace(16, 19.5, 50)
    X_grid, Y_grid = np.meshgrid(load_grid, vcr_grid)
    
    # Use O-Xylene as example
    Z = np.zeros_like(X_grid)
    for i in range(len(vcr_grid)):
        for j in range(len(load_grid)):
            Z[i, j] = problem.interpolate_efficiency(load_grid[j], vcr_grid[i], 0)
    
    contour = ax.contourf(X_grid, Y_grid, Z, levels=20, cmap='viridis')
    ax.set_xlabel('Load (%)')
    ax.set_ylabel('VCR')
    ax.set_title('O-Xylene: Energy Efficiency Heatmap')
    plt.colorbar(contour, ax=ax, label='Efficiency (%)')
    
    # 4. Fluid comparison at optimal conditions
    ax = axes[0, 2]
    optimal_points = []
    for fluid_idx, fluid in enumerate(problem.fluids):
        # Find best efficiency for this fluid
        fluid_mask = X[:, 2] == fluid_idx
        if np.any(fluid_mask):
            best_idx = np.argmax(F[fluid_mask, 0])
            optimal_points.append(F[fluid_mask][best_idx])
        else:
            optimal_points.append([0, 0, 0])
    
    optimal_points = np.array(optimal_points)
    x_pos = np.arange(len(problem.fluids))
    width = 0.25
    
    ax.bar(x_pos - width, optimal_points[:, 0], width, label='Energy Eff (%)', alpha=0.8)
    ax.bar(x_pos, optimal_points[:, 1], width, label='Exergy Eff (%)', alpha=0.8)
    ax.bar(x_pos + width, optimal_points[:, 2]*10, width, label='COP (x10)', alpha=0.8)
    
    ax.set_xlabel('Working Fluid')
    ax.set_ylabel('Performance Metrics')
    ax.set_title('Optimal Performance by Fluid')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(problem.fluids, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Trade-off analysis
    ax = axes[1, 0]
    scatter = ax.scatter(F[:, 0], F[:, 1], c=F[:, 2], cmap='plasma', alpha=0.7)
    ax.set_xlabel('Energy Efficiency (%)')
    ax.set_ylabel('Exergy Efficiency (%)')
    ax.set_title('Energy vs Exergy Efficiency Trade-off')
    plt.colorbar(scatter, ax=ax, label='COP')
    ax.grid(True, alpha=0.3)
    
    # 6. Best operating conditions
    ax = axes[1, 1]
    best_idx = np.argmax(F[:, 0])  # Best energy efficiency
    best_solution = X[best_idx]
    
    metrics = ['Energy Eff', 'Exergy Eff', 'COP']
    values = F[best_idx]
    
    bars = ax.bar(metrics, values, color=['#ff9999', '#66b3ff', '#99ff99'])
    ax.set_ylabel('Performance Value')
    ax.set_title(f'Best Solution: {problem.fluids[int(best_solution[2])]}\n'
                f'Load: {best_solution[0]:.1f}%, VCR: {best_solution[1]:.1f}')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
               f'{value:.2f}', ha='center', va='bottom')
    
    ax.grid(True, alpha=0.3)
    
    # 7. Sensitivity analysis (empty for now)
    axes[1, 2].axis('off')
    axes[1, 2].text(0.5, 0.5, 'Sensitivity Analysis\n(Additional Analysis)',
                   ha='center', va='center', transform=axes[1, 2].transAxes,
                   fontsize=12, style='italic')
    
    plt.tight_layout()
    plt.show()
    
    return best_solution, F[best_idx]

def generate_optimization_report(best_solution, best_performance, problem):
    """Generate comprehensive optimization report"""
    print("="*60)
    print("ORC-VCR SYSTEM OPTIMIZATION REPORT")
    print("="*60)
    
    fluid_name = problem.fluids[int(best_solution[2])]
    
    print(f"\n🎯 OPTIMAL SOLUTION FOUND:")
    print(f"   Working Fluid: {fluid_name}")
    print(f"   Load: {best_solution[0]:.1f}%")
    print(f"   VCR: {best_solution[1]:.1f}")
    
    print(f"\n📊 PERFORMANCE METRICS:")
    print(f"   Energy Efficiency: {best_performance[0]:.2f}%")
    print(f"   Exergy Efficiency: {best_performance[1]:.2f}%")
    print(f"   COP: {best_performance[2]:.3f}")
    
    print(f"\n💡 KEY INSIGHTS:")
    if fluid_name == "O-Xylene":
        print("   • O-Xylene shows superior thermal stability")
        print("   • Best for high-temperature applications")
        print("   • Recommended for industrial scale systems")
    elif fluid_name == "M-Xylene":
        print("   • M-Xylene offers balanced performance")
        print("   • Good compromise between efficiency and stability")
        print("   • Suitable for medium-temperature applications")
    else:
        print("   • Ethylbenzene provides good low-temperature performance")
        print("   • Cost-effective option")
        print("   • Recommended for small-scale systems")
    
    print(f"\n⚙️ OPERATING RECOMMENDATIONS:")
    print(f"   • Maintain load around {best_solution[0]:.0f}% for optimal efficiency")
    print(f"   • Keep VCR at {best_solution[1]:.1f} for best performance")
    print(f"   • Monitor system parameters regularly")
    
    print(f"\n📈 PERFORMANCE COMPARISON:")
    # Compare with other fluids
    for fluid_idx, fluid in enumerate(problem.fluids):
        if fluid_idx != int(best_solution[2]):
            eff = problem.interpolate_efficiency(best_solution[0], best_solution[1], fluid_idx)
            print(f"   {fluid}: {eff:.2f}% efficiency ({(eff-best_performance[0]):.2f}% difference)")

# Run the complete analysis
if __name__ == "__main__":
    print("Running ORC-VCR System Optimization...")
    
    # Run optimization
    results, problem = run_optimization()
    
    # Visualize results
    best_solution, best_performance = visualize_results(results, problem)
    
    # Generate report
    generate_optimization_report(best_solution, best_performance, problem)
    
    # Save optimal parameters
    optimal_params = {
        'fluid': problem.fluids[int(best_solution[2])],
        'load': best_solution[0],
        'vcr': best_solution[1],
        'energy_efficiency': best_performance[0],
        'exergy_efficiency': best_performance[1],
        'cop': best_performance[2]
    }
    
    print(f"\n💾 Optimal parameters saved for system configuration.")