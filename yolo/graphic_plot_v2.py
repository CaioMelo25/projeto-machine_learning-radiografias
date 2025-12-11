import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_v2_results():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    csv_path = os.path.join(base_dir, 'runs', 'detect', 'treino_v2_small_rect', 'results.csv')
    
    output_dir = os.path.join(base_dir, 'graficos_analise_v2')

    if not os.path.exists(csv_path):
        return

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    plt.style.use('ggplot')
    
   
    fig_loss, ax_loss = plt.subplots(1, 3, figsize=(18, 5))
    fig_loss.suptitle('Treino v2 (Small): Evolução do Erro', fontsize=16)

    # Box Loss
    ax_loss[0].plot(df['epoch'], df['train/box_loss'], label='Treino', linewidth=2)
    ax_loss[0].plot(df['epoch'], df['val/box_loss'], label='Validação', linewidth=2, linestyle='--')
    ax_loss[0].set_title('Erro de Caixa (Box Loss)')
    ax_loss[0].set_xlabel('Épocas')
    ax_loss[0].legend()

    # Class Loss
    ax_loss[1].plot(df['epoch'], df['train/cls_loss'], label='Treino', linewidth=2)
    ax_loss[1].plot(df['epoch'], df['val/cls_loss'], label='Validação', linewidth=2, linestyle='--')
    ax_loss[1].set_title('Erro de Classificação (Cls Loss)')
    ax_loss[1].set_xlabel('Épocas')
    ax_loss[1].legend()

    # DFL Loss
    ax_loss[2].plot(df['epoch'], df['train/dfl_loss'], label='Treino', linewidth=2)
    ax_loss[2].plot(df['epoch'], df['val/dfl_loss'], label='Validação', linewidth=2, linestyle='--')
    ax_loss[2].set_title('Erro de Refinamento (DFL Loss)')
    ax_loss[2].set_xlabel('Épocas')
    ax_loss[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'v2_analise_perdas.png'), dpi=300)
    plt.close(fig_loss)

  
    fig_metrics, ax_metrics = plt.subplots(1, 2, figsize=(14, 6))
    fig_metrics.suptitle('Treino v2 (Small): Qualidade do Modelo', fontsize=16)

    # Precision & Recall
    ax_metrics[0].plot(df['epoch'], df['metrics/precision(B)'], label='Precisão', color='blue')
    ax_metrics[0].plot(df['epoch'], df['metrics/recall(B)'], label='Recall (Revocação)', color='green')
    ax_metrics[0].set_title('Precisão vs. Recall')
    ax_metrics[0].set_xlabel('Épocas')
    ax_metrics[0].set_ylabel('Score (0-1)')
    ax_metrics[0].legend()
    ax_metrics[0].grid(True, alpha=0.3)

    # mAP (A Métrica Principal)
    ax_metrics[1].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP @ 50% (Principal)', color='red', linewidth=2)
    ax_metrics[1].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP @ 50-95%', color='orange', linestyle='--')
    ax_metrics[1].set_title('Evolução do mAP')
    ax_metrics[1].set_xlabel('Épocas')
    ax_metrics[1].set_ylabel('mAP')
    ax_metrics[1].legend()
    ax_metrics[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'v2_analise_metricas.png'), dpi=300)
    plt.close(fig_metrics)


if __name__ == "__main__":
    plot_v2_results()