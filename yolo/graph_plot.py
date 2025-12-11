import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_training_results():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    

    csv_path = os.path.join(base_dir, 'runs', 'detect', 'treino_v1_osteoporose', 'results.csv')
    
    output_dir = os.path.join(base_dir, 'graficos_analise')

    if not os.path.exists(csv_path):
        print(f"ERRO: CSV não encontrado em: {csv_path}")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Leitura
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    plt.style.use('ggplot')
    
    # 1. Gráfico de Loss
    fig_loss, ax_loss = plt.subplots(1, 3, figsize=(18, 5))
    fig_loss.suptitle('Curvas de Perda (Training Losses)', fontsize=16)

    ax_loss[0].plot(df['epoch'], df['train/box_loss'], label='Treino', linewidth=2)
    ax_loss[0].plot(df['epoch'], df['val/box_loss'], label='Validação', linewidth=2, linestyle='--')
    ax_loss[0].set_title('Box Loss')
    ax_loss[0].legend()

    ax_loss[1].plot(df['epoch'], df['train/cls_loss'], label='Treino', linewidth=2)
    ax_loss[1].plot(df['epoch'], df['val/cls_loss'], label='Validação', linewidth=2, linestyle='--')
    ax_loss[1].set_title('Class Loss')
    ax_loss[1].legend()

    ax_loss[2].plot(df['epoch'], df['train/dfl_loss'], label='Treino', linewidth=2)
    ax_loss[2].plot(df['epoch'], df['val/dfl_loss'], label='Validação', linewidth=2, linestyle='--')
    ax_loss[2].set_title('DFL Loss')
    ax_loss[2].legend()

    plt.savefig(os.path.join(output_dir, 'analise_perdas.png'), dpi=300)
    plt.close(fig_loss)

    # 2. Gráfico de Métricas
    fig_metrics, ax_metrics = plt.subplots(1, 2, figsize=(14, 6))
    fig_metrics.suptitle('Métricas de Validação', fontsize=16)

    ax_metrics[0].plot(df['epoch'], df['metrics/precision(B)'], label='Precision', color='blue')
    ax_metrics[0].plot(df['epoch'], df['metrics/recall(B)'], label='Recall', color='green')
    ax_metrics[0].set_title('Precision e Recall')
    ax_metrics[0].legend()

    ax_metrics[1].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@50', color='red', linewidth=2)
    ax_metrics[1].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@50-95', color='orange', linestyle='--')
    ax_metrics[1].set_title('mAP (Acurácia)')
    ax_metrics[1].legend()

    plt.savefig(os.path.join(output_dir, 'analise_metricas.png'), dpi=300)
    plt.close(fig_metrics)

if __name__ == "__main__":
    plot_training_results()