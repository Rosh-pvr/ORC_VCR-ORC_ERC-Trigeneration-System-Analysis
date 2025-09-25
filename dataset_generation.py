def generate_synthetic_dataset():
    """Generate comprehensive dataset for analysis"""
    np.random.seed(42)
    
    n_samples = 1000
    data = []
    
    for _ in range(n_samples):
        load = np.random.uniform(20, 80)
        vcr = np.random.uniform(16, 19.5)
        fluid_idx = np.random.randint(0, 3)
        fluid = ['O-Xylene', 'M-Xylene', 'Ethylbenzene'][fluid_idx]
        
        # Base efficiency with some noise
        base_eff = 10 + (load-20)*0.1 + (vcr-16)*0.5
        if fluid == 'O-Xylene':
            base_eff += 2.0
        elif fluid == 'M-Xylene':
            base_eff += 1.0
        
        # Add some realistic noise
        energy_eff = base_eff + np.random.normal(0, 0.5)
        exergy_eff = energy_eff * 3.2 + np.random.normal(0, 0.3)
        cop = energy_eff * 0.08 + np.random.normal(0, 0.02)
        
        data.append([load, vcr, fluid, energy_eff, exergy_eff, cop])
    
    df = pd.DataFrame(data, columns=['Load', 'VCR', 'Fluid', 
                                    'Energy_Efficiency', 'Exergy_Efficiency', 'COP'])
    
    # Save dataset
    df.to_csv('orc_vcr_optimization_dataset.csv', index=False)
    print(f"Dataset generated with {len(df)} samples")
    return df

# Generate dataset if needed
# dataset = generate_synthetic_dataset()