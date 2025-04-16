import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

def generate_styled_table():
    os.makedirs("results/tables", exist_ok=True)
    
    data = {
        'System': ['Google Translate', 'MarianMT'],
        'BLEU': [0.406784, 0.032831],
        'METEOR': [0.613945, 0.154972],
        'chrF': [0.738542, 0.216781],
        'ROUGE': [0.678223, 0.388651]
    }

    df = pd.DataFrame(data)
    
    styled_df = df.style.background_gradient(cmap='Blues', subset=['BLEU', 'METEOR', 'chrF', 'ROUGE']) \
        .format({'BLEU': '{:.2f}', 'METEOR': '{:.2f}', 'chrF': '{:.2f}', 'ROUGE': '{:.2f}'}) \
        .set_caption('Machine Translation System Performance') \
        .set_table_styles([
            {'selector': 'caption', 'props': [('font-size', '16px'), 
                                             ('font-weight', 'bold'),
                                             ('text-align', 'center')]},
            {'selector': 'th', 'props': [('font-weight', 'bold'), 
                                         ('background-color', '#f2f2f2'),
                                         ('text-align', 'center')]},
            {'selector': 'td', 'props': [('text-align', 'center')]}
        ])
    
    html_path = 'results/tables/styled_results_table.html'
    styled_df.to_html(html_path)
    print(f"HTML table saved to: {html_path}")
    
    plt.figure(figsize=(8, 3))
    ax = plt.subplot(111, frame_on=False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    
    table = plt.table(
        cellText=[[f"{x:.2f}" for x in df.iloc[i, 1:].values] for i in range(len(df))],
        rowLabels=df['System'],
        colLabels=df.columns[1:],
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    for i, key in enumerate(df.columns[1:]):
        cell = table[(0, i)]
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#4472C4')
    
    for i in range(len(df)):
        for j in range(len(df.columns[1:])):
            cell = table[(i+1, j)]
            value = df.iloc[i, j+1]
            max_val = df.iloc[:, j+1].max()
            min_val = df.iloc[:, j+1].min()
            norm_value = (value - min_val) / (max_val - min_val) if max_val > min_val else 0.5
            cell.set_facecolor(plt.cm.Blues(0.3 + 0.7 * norm_value))
    
    plt_path = 'results/tables/results_table.png'
    plt.savefig(plt_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"PNG table saved to: {plt_path}")
    
    return html_path, plt_path

if __name__ == "__main__":
    html_path, plt_path = generate_styled_table()
    print("Table generation complete!")