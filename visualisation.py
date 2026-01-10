import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

def plot_percolation_stats_IU(L_values, mSI, sSI, mSU, sSU, mBI, sBI, mBU, sBU):
    plt.figure(figsize=(12, 7))
    plt.errorbar(L_values, mSI, yerr=sSI, fmt='o-', color='blue', label='Site I')
    plt.errorbar(L_values, mSU, yerr=sSU, fmt='s-', color='cyan', label='Site U')
    plt.errorbar(L_values, mBI, yerr=sBI, fmt='^-', color='red', label='Bond I')
    plt.errorbar(L_values, mBU, yerr=sBU, fmt='x-', color='orange', label='Bond U')
    plt.xlabel('Linear System Size ($L$)')
    plt.ylabel('Critical Probability ($p_c$)')
    plt.title('Site vs Bond Percolation (I & U Criteria)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    print("\nDrawing tiling...")
    fig = plt.figure(figsize=(16, 12), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    to_screen = [1, 0, 0, 0, 1, 0]
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    plt.title("Hat Tile Aperiodic Tiling", fontsize=16, pad=20)
    plt.show()

def plot_extrapolation_IU(L_values, mSI, mSU, mBI, mBU, exponent=-3/4):
    """
    Plots two separate figures for Site and Bond percolation extrapolation.
    """
    L_values = np.array(L_values)
    X = L_values ** exponent
    
    # Data configuration for separate plots
    configs = [
        {
            'title': 'Site Percolation Finite-Size Scaling',
            'data': [
                ('Intersection', mSI, 'blue', 'o'),
                ('Union', mSU, 'red', 's'),
                ('Average', 0.5 * (np.array(mSI) + np.array(mSU)), 'green', '^')
            ]
        },
        {
            'title': 'Bond Percolation Finite-Size Scaling',
            'data': [
                ('Intersection', mBI, 'blue', 'o'),
                ('Union', mBU, 'red', 's'),
                ('Average', 0.5 * (np.array(mBI) + np.array(mBU)), 'green', '^')
            ]
        }
    ]

    for config in configs:
        plt.figure(figsize=(12, 7), dpi=100)
        
        for label, y, color, marker in config['data']:
            y = np.array(y)
            # Perform linear regression to extrapolate to L -> infinity (X = 0)
            res = linregress(X, y)
            pc_inf = res.intercept
            
            # Plot the raw data points
            plt.plot(X, y, marker, color=color, markersize=8, label=f'{label} Data')
            
            # Plot the extrapolation line
            X_line = np.linspace(0, max(X), 100)
            Y_line = res.slope * X_line + pc_inf
            plt.plot(X_line, Y_line, '--', color=color, 
                     label=f'{label} Fit: $p_c(\\infty)$={pc_inf:.5f}')
            
            # Mark the intercept at the Y-axis
            plt.plot(0, pc_inf, 'x', color=color, markersize=10)

        plt.xlabel(f'$L^{{{exponent}}}$', fontsize=12)
        plt.ylabel('Mean Critical Probability ($\\bar{p}_c$)', fontsize=12)
        plt.title(config['title'], fontsize=14)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xlim(left=-0.005) # Ensure the intercept is clearly visible
        plt.tight_layout()
        plt.show()