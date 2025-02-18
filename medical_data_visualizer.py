import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Load the dataset
df = pd.read_csv('medical_examination.csv')

# 2. Add the "overweight" column
df['overweight'] = (df['weight'] / (df['height'] / 100) ** 2 > 25).astype(int)

# 3. Normalize "cholesterol" and "gluc" values (1 → 0, others → 1)
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)


# 4. Function to generate the categorical plot
def draw_cat_plot():
    # 5. Reshape data for visualization
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # 6. Group data by "cardio", "variable", and "value"
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

    # 7. Create the categorical plot
    catplot = sns.catplot(
        x='variable', y='total', hue='value', col='cardio',
        data=df_cat, kind='bar', height=4, aspect=1.2
    )

    # 8. Convert Seaborn's FacetGrid to a Matplotlib Figure
    fig = catplot.fig  

    # 9. Save the figure
    fig.savefig('catplot.png')

    return fig


# 10. Function to generate the heatmap
def draw_heat_map():
    # 11. Clean the dataset by removing incorrect or extreme values
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &  # Diastolic pressure must be ≤ systolic
        (df['height'] >= df['height'].quantile(0.025)) &  # Height within 2.5%-97.5% range
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &  # Weight within 2.5%-97.5% range
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 12. Compute the correlation matrix
    corr = df_heat.corr()

    # 13. Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14. Set up the Matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # 15. Draw the heatmap
    sns.heatmap(corr, mask=mask, annot=True, fmt=".1f", cmap='coolwarm', center=0, square=True, linewidths=0.5, cbar_kws={'shrink': 0.5})

    # 16. Save the figure
    fig.savefig('heatmap.png')

    return fig
