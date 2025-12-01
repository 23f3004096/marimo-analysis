import marimo

__generated_with = "0.9.0"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Email: 23f3004096@ds.study.iitm.ac.in
    # This notebook demonstrates interactive data analysis
    # with automatic reactivity between cells
    
    return mo, np, pd, plt


@app.cell
def __(mo):
    # Create an interactive slider widget
    # This slider controls the sample size for our analysis
    # Range: 10 to 1000 samples
    
    sample_size_slider = mo.ui.slider(
        start=10,
        stop=1000,
        step=10,
        value=100,
        label="Sample Size: "
    )
    
    return sample_size_slider,


@app.cell
def __(mo, sample_size_slider):
    # Display the slider widget
    # This cell shows the interactive control to the user
    
    mo.md(f"""
    ## Interactive Data Analysis Dashboard
    
    **Analyst:** 23f3004096@ds.study.iitm.ac.in
    
    ### Adjust Sample Size
    
    Use the slider below to change the number of data points in the analysis.
    All visualizations and statistics will update automatically!
    
    {sample_size_slider}
    """)


@app.cell
def __(np, sample_size_slider):
    # Generate synthetic dataset based on slider value
    # This cell DEPENDS on sample_size_slider
    # When slider changes, this cell automatically re-runs
    
    # Get the current slider value
    n_samples = sample_size_slider.value
    
    # Generate random data with a linear relationship
    np.random.seed(42)
    x_data = np.linspace(0, 10, n_samples)
    y_data = 2.5 * x_data + 1.5 + np.random.normal(0, 2, n_samples)
    
    # Calculate correlation
    correlation = np.corrcoef(x_data, y_data)[0, 1]
    
    return n_samples, x_data, y_data, correlation


@app.cell
def __(mo, n_samples, correlation):
    # Dynamic markdown output that changes based on widget state
    # This cell DEPENDS on n_samples and correlation
    # Output updates automatically when slider changes
    
    mo.md(f"""
    ### Dataset Statistics
    
    - **Current Sample Size:** {n_samples} data points
    - **Correlation Coefficient:** {correlation:.4f}
    - **Relationship Strength:** {'Strong' if abs(correlation) > 0.7 else 'Moderate' if abs(correlation) > 0.4 else 'Weak'}
    
    ---
    
    The correlation coefficient measures the linear relationship between variables.
    Values closer to ±1 indicate stronger relationships.
    """)


@app.cell
def __(plt, x_data, y_data, n_samples, correlation):
    # Create visualization based on generated data
    # This cell DEPENDS on x_data, y_data, n_samples, correlation
    # Plot automatically updates when slider changes
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Scatter plot
    ax.scatter(x_data, y_data, alpha=0.6, s=50, c='steelblue', edgecolors='navy')
    
    # Add trend line
    z = np.polyfit(x_data, y_data, 1)
    p = np.poly1d(z)
    ax.plot(x_data, p(x_data), "r--", linewidth=2, label=f'Trend line (r={correlation:.3f})')
    
    # Labels and title
    ax.set_xlabel('Independent Variable (X)', fontsize=12)
    ax.set_ylabel('Dependent Variable (Y)', fontsize=12)
    ax.set_title(f'Linear Relationship Analysis (n={n_samples})', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_output = fig
    plt.close()
    
    return plot_output,


@app.cell
def __(mo, plot_output):
    # Display the plot
    # This cell DEPENDS on plot_output
    
    mo.md(f"""
    ### Visualization
    
    {mo.as_html(plot_output)}
    
    The scatter plot shows individual data points, and the red dashed line 
    represents the best-fit linear trend.
    """)


@app.cell
def __(mo, pd, x_data, y_data):
    # Create a summary statistics table
    # This cell DEPENDS on x_data and y_data
    # Table updates automatically when slider changes
    
    summary_df = pd.DataFrame({
        'Variable': ['X (Independent)', 'Y (Dependent)'],
        'Mean': [x_data.mean(), y_data.mean()],
        'Std Dev': [x_data.std(), y_data.std()],
        'Min': [x_data.min(), y_data.min()],
        'Max': [x_data.max(), y_data.max()]
    })
    
    # Round to 3 decimal places
    summary_df = summary_df.round(3)
    
    return summary_df,


@app.cell
def __(mo, summary_df):
    # Display the summary table
    # This cell DEPENDS on summary_df
    
    mo.md(f"""
    ### Summary Statistics
    
    {mo.as_html(summary_df)}
    
    ---
    
    **Data Flow Documentation:**
    
    1. **Slider Widget** → Controls sample size
    2. **Data Generation** → Creates x_data, y_data based on sample size
    3. **Statistics Calculation** → Computes correlation from data
    4. **Visualization** → Plots data and trend line
    5. **Summary Table** → Shows descriptive statistics
    
    All components are **reactive** - change the slider and watch everything update!
    
    **Contact:** 23f3004096@ds.study.iitm.ac.in
    """)


if __name__ == "__main__":
    app.run()
