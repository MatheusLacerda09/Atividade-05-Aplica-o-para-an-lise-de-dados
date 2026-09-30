import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    df_transacoes = pd.read_csv('transacoes.csv', encoding='latin1')
    df_cotacoes = pd.read_csv('cotacoes.csv', encoding='utf-8')
    
    df_transacoes['valor'] = df_transacoes['valor'].fillna(df_transacoes.groupby('estado_cliente')['valor'].transform('median'))
    df_transacoes['plataforma'] = 'Mobile'
    
    df_transacoes['data_transacao'] = pd.to_datetime(df_transacoes['data_transacao'])
    df_transacoes['data_transacao'] = df_transacoes['data_transacao'].dt.tz_localize('America/Sao_Paulo', ambiguous='NaT', nonexistent='NaT')
    
    df_transacoes['dia_semana'] = df_transacoes['data_transacao'].dt.day_name()
    df_transacoes['mes'] = df_transacoes['data_transacao'].dt.month
    
    df_transacoes = df_transacoes.drop_duplicates(keep='first')
    
    df_cotacoes['data'] = pd.to_datetime(df_cotacoes['data'])
    df_cotacoes['cotacao_usd'] = df_cotacoes['cotacao_usd'].ffill()
    
    df_transacoes['data_base'] = df_transacoes['data_transacao'].dt.date
    df_cotacoes['data_base'] = df_cotacoes['data'].dt.date
    
    df_consolidado = pd.merge(df_transacoes, df_cotacoes, on='data_base', how='left')
    
    filtro_setembro_sudeste = (df_consolidado['mes'] == 9) & ((df_consolidado['estado_cliente'] == 'SP') | (df_consolidado['estado_cliente'] == 'RJ')) & (df_consolidado['valor'] > 5000.0)
    df_filtrado_setembro = df_consolidado[filtro_setembro_sudeste]
    print(f"Total de transações em SP/RJ no mês 9 com valor > R$5000: {len(df_filtrado_setembro)}")
    
    risco_dict = {'C100': 'Baixo', 'C101': 'Alto', 'C102': 'Medio', 'C103': 'Baixo', 'C104': 'Alto'}
    df_consolidado['nivel_risco'] = df_consolidado['id_cliente'].map(risco_dict)
    
    pivot_risco = pd.pivot_table(
        df_consolidado, 
        values='valor', 
        index='mes', 
        columns='nivel_risco', 
        aggfunc='sum', 
        margins=True, 
        fill_value=0
    )
    print("\n--- Tabela Dinâmica de Risco ---")
    print(pivot_risco)
    
    df_consolidado['z_score_estado'] = df_consolidado.groupby('estado_cliente')['valor'].transform(
        lambda x: (x - x.mean()) / x.std()
    )
    
    df_anomalias = df_consolidado[df_consolidado['z_score_estado'] > 2.5].copy()
    print(f"\nTotal de potenciais anomalias detectadas (Z-Score > 2.5): {len(df_anomalias)}")
    
    vendas_diarias = df_consolidado.groupby('data_base')['valor'].sum().reset_index()
    vendas_diarias['media_movel_7d'] = vendas_diarias['valor'].rolling(window=7, min_periods=1).mean()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(vendas_diarias['data_base'], vendas_diarias['valor'], label='Total Diário', color='#1f77b4', marker='o', alpha=0.6)
    ax.plot(vendas_diarias['data_base'], vendas_diarias['media_movel_7d'], label='Média Móvel (7 dias)', color='#d62728', linewidth=2.5)
    
    ax.set_ylim(bottom=0)
    
    ax.set_title('Desempenho Diário de Transações e Média Móvel Suavizada', fontsize=14)
    ax.set_xlabel('Data da Transação', fontsize=12)
    ax.set_ylabel('Volume de Transações (R$)', fontsize=12)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    fig.savefig('relatorio_visual.png', dpi=300)
    print("\nGráfico salvo como 'relatorio_visual.png'.")
    plt.show()

if __name__ == "__main__":
    main()