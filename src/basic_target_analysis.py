"""
Análisis de Conversión - Campaña de Marketing Bancario
========================================================
Genera visualizaciones para analizar tasas de conversión por diferentes segmentos.
Versión refactorizada con configuraciones centralizadas para nuevos datasets.
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
from typing import Dict, Optional

from src.config import (
    DEFAULT_COLUMNS, DEFAULT_VALUES, DEFAULT_GENERATIONS, DEFAULT_ANALYSIS,
    ColumnMapping, ValueMapping, GenerationConfig, AnalysisConfig
)
from src.data_validator import validate_data, DataValidationError


# ============================================================================
# Utilidades de Color y Estilo
# ============================================================================

def interpolate_color(color1: str, color2: str, t: float) -> str:
    """Interpola entre dos colores hexadecimales."""
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def rgb_to_hex(rgb):
        return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))
    
    rgb1 = hex_to_rgb(color1)
    rgb2 = hex_to_rgb(color2)
    rgb = tuple(int(rgb1[i] + (rgb2[i] - rgb1[i]) * t) for i in range(3))
    return rgb_to_hex(rgb)


def get_blue_gradient(value: float, min_val: float = 0.0, max_val: float = 1.0) -> str:
    """Genera un color de la paleta azul basado en un valor normalizado."""
    normalized = (value - min_val) / (max_val - min_val) if max_val > min_val else 0.0
    normalized = max(0.0, min(1.0, normalized))
    
    colors = ['#6ce5e8', '#41b8d5', '#2d8bba', '#004aad']
    
    if normalized <= 0.33:
        t = normalized / 0.33
        return interpolate_color('#6ce5e8', '#41b8d5', t)
    elif normalized <= 0.66:
        t = (normalized - 0.33) / 0.33
        return interpolate_color('#41b8d5', '#2d8bba', t)
    else:
        t = (normalized - 0.66) / 0.34
        return interpolate_color('#2d8bba', '#004aad', t)


def setup_style() -> Dict[str, str]:
    """Configura el estilo visual para todos los gráficos."""
    plt.style.use('seaborn-v0_8-whitegrid')
    
    colors = {
        'primary': '#004aad',
        'secondary': '#41b8d5',
        'accent': '#6ce5e8',
        'warning': '#e74c3c',
        'neutral': '#2d8bba',
        'background': '#f8f9fa'
    }
    
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': '#333333',
        'axes.labelcolor': '#333333',
        'axes.titlecolor': '#1e3a5f',
        'xtick.color': '#333333',
        'ytick.color': '#333333',
        'text.color': '#333333',
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'figure.titlesize': 16,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.facecolor': 'white',
        'savefig.edgecolor': 'none'
    })
    
    return colors


# ============================================================================
# Utilidades de Datos
# ============================================================================

def get_column_safe(df: pd.DataFrame, col_name: str, 
                   columns: Optional[ColumnMapping] = None) -> str:
    """Obtiene el nombre de columna de forma segura usando la configuración."""
    if columns is None:
        columns = DEFAULT_COLUMNS
    
    col_attr = getattr(columns, col_name, None)
    if col_attr is None:
        raise KeyError(f"Atributo '{col_name}' no encontrado en ColumnMapping")
    
    if col_attr not in df.columns:
        raise KeyError(f"Columna '{col_attr}' no encontrada en el DataFrame")
    
    return col_attr


def get_all_columns(df: pd.DataFrame, 
                   columns: Optional[ColumnMapping] = None) -> Dict[str, str]:
    """Obtiene todos los nombres de columnas necesarias de forma segura."""
    if columns is None:
        columns = DEFAULT_COLUMNS
    
    col_names = {}
    for attr in ['target', 'contact', 'month', 'day', 'age', 'balance', 
                 'duration', 'campaign', 'housing', 'loan', 'job']:
        col_names[attr] = get_column_safe(df, attr, columns)
    
    return col_names


def create_output_dir(analysis_config: Optional[AnalysisConfig] = None) -> Path:
    """Crea el directorio para guardar los gráficos."""
    if analysis_config is None:
        analysis_config = DEFAULT_ANALYSIS
    
    output_dir = analysis_config.output_dir
    if not output_dir.is_absolute():
        output_dir = Path(__file__).parent.parent / output_dir
    
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# ============================================================================
# Funciones de Visualización
# ============================================================================

def plot_conversion_rate(df: pd.DataFrame, colors: Dict, output_dir: Path,
                         columns: Optional[ColumnMapping] = None,
                         values: Optional[ValueMapping] = None) -> float:
    """Gráfico 1: Tasa General de Conversión (Donut Chart)."""
    if columns is None:
        columns = DEFAULT_COLUMNS
    if values is None:
        values = DEFAULT_VALUES
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    target_col = columns.target
    conversion = df[target_col].value_counts()
    sizes = [
        conversion.get(values.target_negative, 0), 
        conversion.get(values.target_positive, 0)
    ]
    chart_colors = [colors['neutral'], colors['accent']]
    
    wedges, texts, autotexts = ax.pie(
        sizes, 
        explode=(0, 0.05),
        labels=['No Convirtió', 'Convirtió'],
        colors=chart_colors,
        autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*sum(sizes)):,})',
        startangle=90,
        pctdistance=0.75,
        wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2)
    )
    
    for autotext in autotexts:
        autotext.set_fontsize(11)
        autotext.set_fontweight('bold')
    
    centre_circle = plt.Circle((0, 0), 0.35, fc='white')
    ax.add_patch(centre_circle)
    
    total = sum(sizes)
    tasa = sizes[1] / total * 100
    ax.text(0, 0.05, f'{tasa:.1f}%', ha='center', va='center', 
            fontsize=24, fontweight='bold', color=colors['primary'])
    ax.text(0, -0.15, 'Tasa de\nConversión', ha='center', va='center', 
            fontsize=10, color=colors['neutral'])
    
    ax.set_title('Tasa General de Conversión de la Campaña', 
                 fontsize=16, fontweight='bold', pad=20, color=colors['primary'])
    
    plt.tight_layout()
    fig.savefig(output_dir / '01_tasa_conversion_general.png')
    plt.close(fig)
    
    return tasa


def plot_conversion_by_contact(df: pd.DataFrame, colors: Dict, output_dir: Path,
                               columns: Optional[ColumnMapping] = None,
                               values: Optional[ValueMapping] = None):
    """Gráfico 2: Conversión por Canal de Contacto."""
    if columns is None:
        columns = DEFAULT_COLUMNS
    if values is None:
        values = DEFAULT_VALUES
    
    target_col = get_column_safe(df, 'target', columns)
    contact_col = get_column_safe(df, 'contact', columns)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    contact_data = df.groupby(contact_col)[target_col].value_counts().unstack(fill_value=0)
    contact_data['total'] = contact_data.sum(axis=1)
    contact_data['tasa_conversion'] = (
        contact_data.get(values.target_positive, 0) / contact_data['total'] * 100
    ).round(2)
    contact_data = contact_data.sort_values('tasa_conversion', ascending=True)
    
    # Subplot 1: Volumen
    ax1 = axes[0]
    y_pos = range(len(contact_data))
    
    ax1.barh(y_pos, contact_data.get(values.target_negative, 0), 
             color=colors['neutral'], label='No Convirtió', height=0.6)
    ax1.barh(y_pos, contact_data.get(values.target_positive, 0), 
             left=contact_data.get(values.target_negative, 0),
             color=colors['accent'], label='Convirtió', height=0.6)
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([c.title() for c in contact_data.index])
    ax1.set_xlabel('Número de Clientes')
    ax1.set_title('Volumen por Canal de Contacto', fontweight='bold', color=colors['primary'])
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), 
               ncol=2, frameon=True, framealpha=0.95, edgecolor='#cccccc')
    
    for i, (idx, row) in enumerate(contact_data.iterrows()):
        ax1.text(row['total'] + 100, i, f'{int(row["total"]):,}', 
                va='center', fontsize=11, fontweight='bold', color=colors['primary'])
    
    # Subplot 2: Tasa de conversión
    ax2 = axes[1]
    bars = ax2.bar(
        [c.title() for c in contact_data.index],
        contact_data['tasa_conversion'],
        color=[colors['secondary'] if t < contact_data['tasa_conversion'].max() 
               else colors['accent'] for t in contact_data['tasa_conversion']],
        edgecolor='white',
        linewidth=1.5
    )
    
    for bar, val in zip(bars, contact_data['tasa_conversion']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax2.set_ylabel('Tasa de Conversión (%)')
    ax2.set_title('Tasa de Conversión por Canal', fontweight='bold', color=colors['primary'])
    ax2.set_ylim(0, max(contact_data['tasa_conversion']) * 1.2)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    fig.savefig(output_dir / '02_conversion_por_canal.png')
    plt.close(fig)
    
    return contact_data[['total', 'tasa_conversion']]


def plot_conversion_by_month(df: pd.DataFrame, colors: Dict, output_dir: Path,
                            columns: Optional[ColumnMapping] = None,
                            values: Optional[ValueMapping] = None,
                            analysis_config: Optional[AnalysisConfig] = None):
    """Gráfico 3: Conversión por Mes (Análisis Temporal)."""
    if columns is None:
        columns = DEFAULT_COLUMNS
    if values is None:
        values = DEFAULT_VALUES
    if analysis_config is None:
        analysis_config = DEFAULT_ANALYSIS
    
    target_col = get_column_safe(df, 'target', columns)
    month_col = get_column_safe(df, 'month', columns)
    
    orden_meses = values.months_order
    nombres_meses = [analysis_config.translations.get('months', {}).get(m, m.title()) 
                     for m in orden_meses]
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    month_data = df.groupby(month_col)[target_col].value_counts().unstack(fill_value=0)
    month_data['total'] = month_data.sum(axis=1)
    month_data['tasa_conversion'] = (
        month_data.get(values.target_positive, 0) / month_data['total'] * 100
    ).round(2)
    
    meses_presentes = [m for m in orden_meses if m in month_data.index]
    month_data = month_data.reindex(meses_presentes)
    nombres_filtrados = [nombres_meses[orden_meses.index(m)] for m in meses_presentes]
    
    # Subplot 1: Volumen
    ax1 = axes[0]
    x = range(len(month_data))
    width = 0.35
    
    ax1.bar([i - width/2 for i in x], month_data.get(values.target_negative, 0), 
            width, label='No Convirtió', color=colors['neutral'])
    ax1.bar([i + width/2 for i in x], month_data.get(values.target_positive, 0), 
            width, label='Convirtió', color=colors['accent'])
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(nombres_filtrados)
    ax1.set_ylabel('Número de Clientes')
    ax1.set_title('Volumen de Contactos por Mes', fontweight='bold', 
                  fontsize=14, color=colors['primary'])
    ax1.legend(loc='upper right', framealpha=0.9)
    
    ax1_twin = ax1.twinx()
    ax1_twin.plot(x, month_data['total'], 'o-', color=colors['primary'], 
                  linewidth=2, markersize=6, label='Total')
    ax1_twin.set_ylabel('Total de Contactos', color=colors['primary'])
    ax1_twin.tick_params(axis='y', labelcolor=colors['primary'])
    
    # Subplot 2: Tasa de conversión
    ax2 = axes[1]
    min_tasa = month_data['tasa_conversion'].min()
    max_tasa = month_data['tasa_conversion'].max()
    bar_colors = [get_blue_gradient(t, min_tasa, max_tasa) for t in month_data['tasa_conversion']]
    
    bars = ax2.bar(nombres_filtrados, month_data['tasa_conversion'], 
                   color=bar_colors, edgecolor='white', linewidth=1.5)
    
    promedio = month_data['tasa_conversion'].mean()
    ax2.axhline(y=promedio, color=colors['warning'], linestyle='--', 
                linewidth=2, label=f'Promedio: {promedio:.1f}%')
    
    for bar, val in zip(bars, month_data['tasa_conversion']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax2.set_ylabel('Tasa de Conversión (%)')
    ax2.set_xlabel('Mes')
    ax2.set_title('Tasa de Conversión Mensual', fontweight='bold', 
                  fontsize=14, color=colors['primary'])
    ax2.legend(loc='upper right')
    ax2.set_ylim(0, max(month_data['tasa_conversion']) * 1.25)
    
    plt.tight_layout()
    fig.savefig(output_dir / '03_conversion_por_mes.png')
    plt.close(fig)
    
    return month_data[['total', 'tasa_conversion']]


def plot_demographic_analysis(df: pd.DataFrame, colors: Dict, output_dir: Path,
                              columns: Optional[ColumnMapping] = None,
                              values: Optional[ValueMapping] = None,
                              generations: Optional[GenerationConfig] = None,
                              analysis_config: Optional[AnalysisConfig] = None) -> Dict:
    """Gráfico 4: Análisis demográfico y de comportamiento."""
    if columns is None:
        columns = DEFAULT_COLUMNS
    if values is None:
        values = DEFAULT_VALUES
    if generations is None:
        generations = DEFAULT_GENERATIONS
    if analysis_config is None:
        analysis_config = DEFAULT_ANALYSIS
    
    # Obtener columnas
    target_col = get_column_safe(df, 'target', columns)
    duration_col = get_column_safe(df, 'duration', columns)
    balance_col = get_column_safe(df, 'balance', columns)
    age_col = get_column_safe(df, 'age', columns)
    campaign_col = get_column_safe(df, 'campaign', columns)
    housing_col = get_column_safe(df, 'housing', columns)
    loan_col = get_column_safe(df, 'loan', columns)
    job_col = get_column_safe(df, 'job', columns)
    day_col = get_column_safe(df, 'day', columns)
    
    df_yes = df[df[target_col] == values.target_positive]
    df_no = df[df[target_col] == values.target_negative]
    
    # --- Gráfico 4A: Duración de Contactos ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    box_data = [df_no[duration_col], df_yes[duration_col]]
    bp = ax1.boxplot(box_data, tick_labels=['No Convirtió', 'Convirtió'], 
                     patch_artist=True, widths=0.6)
    
    bp['boxes'][0].set_facecolor(colors['neutral'])
    bp['boxes'][1].set_facecolor(colors['accent'])
    for box in bp['boxes']:
        box.set_edgecolor('#333333')
        box.set_linewidth(1.5)
    for median in bp['medians']:
        median.set_color('#333333')
        median.set_linewidth(2)
    
    ax1.set_ylabel('Duración (segundos)')
    ax1.set_title('Distribución de Duración del Contacto', fontweight='bold', color=colors['primary'])
    
    for i, data in enumerate([df_no[duration_col], df_yes[duration_col]]):
        median = data.median()
        mean = data.mean()
        ax1.text(i + 1, ax1.get_ylim()[1] * 0.95, 
                f'Mediana: {median:.0f}s\nPromedio: {mean:.0f}s', 
                ha='center', va='top', fontsize=9, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax2 = axes[1]
    ax2.hist(df_no[duration_col], bins=30, alpha=0.7, label='No Convirtió', 
             color=colors['neutral'], edgecolor='white')
    ax2.hist(df_yes[duration_col], bins=30, alpha=0.7, label='Convirtió', 
             color=colors['accent'], edgecolor='white')
    ax2.axvline(df_no[duration_col].median(), color=colors['neutral'], 
                linestyle='--', linewidth=2, label=f'Mediana No: {df_no[duration_col].median():.0f}s')
    ax2.axvline(df_yes[duration_col].median(), color=colors['accent'], 
                linestyle='--', linewidth=2, label=f'Mediana Sí: {df_yes[duration_col].median():.0f}s')
    ax2.set_xlabel('Duración (segundos)')
    ax2.set_ylabel('Frecuencia')
    ax2.set_title('Histograma de Duración', fontweight='bold', color=colors['primary'])
    ax2.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    fig.savefig(output_dir / '04_duracion_contactos.png')
    plt.close(fig)
    
    # --- Gráfico 4A-2: Duración sin outliers (P95) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    p95 = df[duration_col].quantile(analysis_config.outlier_percentile)
    df_no_filtered = df_no[df_no[duration_col] <= p95]
    df_yes_filtered = df_yes[df_yes[duration_col] <= p95]
    
    ax1 = axes[0]
    box_data = [df_no_filtered[duration_col], df_yes_filtered[duration_col]]
    bp = ax1.boxplot(box_data, tick_labels=['No Convirtió', 'Convirtió'], 
                     patch_artist=True, widths=0.6, showfliers=False)
    
    bp['boxes'][0].set_facecolor(colors['neutral'])
    bp['boxes'][1].set_facecolor(colors['accent'])
    for box in bp['boxes']:
        box.set_edgecolor('#333333')
        box.set_linewidth(1.5)
    for median in bp['medians']:
        median.set_color('#333333')
        median.set_linewidth(2)
    
    ax1.set_ylabel('Duración (segundos)')
    ax1.set_title(f'Duración del Contacto (P95 <= {p95:.0f}s)', fontweight='bold', color=colors['primary'])
    
    for i, data in enumerate([df_no_filtered[duration_col], df_yes_filtered[duration_col]]):
        median = data.median()
        q25 = data.quantile(0.25)
        q75 = data.quantile(0.75)
        ax1.text(i + 1, ax1.get_ylim()[1] * 0.95, 
                f'Mediana: {median:.0f}s\nQ25: {q25:.0f}s | Q75: {q75:.0f}s', 
                ha='center', va='top', fontsize=9, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax2 = axes[1]
    bins = 25
    ax2.hist(df_no_filtered[duration_col], bins=bins, alpha=0.7, label='No Convirtió', 
             color=colors['neutral'], edgecolor='white')
    ax2.hist(df_yes_filtered[duration_col], bins=bins, alpha=0.7, label='Convirtió', 
             color=colors['accent'], edgecolor='white')
    ax2.axvline(df_no_filtered[duration_col].median(), color='#555555', 
                linestyle='--', linewidth=2, label=f'Mediana No: {df_no_filtered[duration_col].median():.0f}s')
    ax2.axvline(df_yes_filtered[duration_col].median(), color=colors['accent'], 
                linestyle='-', linewidth=2.5, label=f'Mediana Sí: {df_yes_filtered[duration_col].median():.0f}s')
    
    ax2.set_xlabel('Duración (segundos)')
    ax2.set_ylabel('Frecuencia')
    ax2.set_title(f'Histograma Sin Outliers (n={len(df_no_filtered)+len(df_yes_filtered):,})', 
                  fontweight='bold', color=colors['primary'])
    ax2.legend(loc='upper right', fontsize=9)
    
    total_original = len(df)
    total_filtrado = len(df_no_filtered) + len(df_yes_filtered)
    pct_incluido = total_filtrado / total_original * 100
    fig.text(0.5, 0.02, 
             f'Nota: Se muestran {total_filtrado:,} de {total_original:,} registros ({pct_incluido:.1f}%) - Filtrado al percentil 95', 
             ha='center', fontsize=9, style='italic', color='#666666')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    fig.savefig(output_dir / '04b_duracion_sin_outliers.png')
    plt.close(fig)
    
    # --- Gráfico 4B: Balance/Salario ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    box_data = [df_no[balance_col], df_yes[balance_col]]
    bp = ax1.boxplot(box_data, tick_labels=['No Convirtió', 'Convirtió'], 
                     patch_artist=True, widths=0.6, showfliers=False)
    
    bp['boxes'][0].set_facecolor(colors['neutral'])
    bp['boxes'][1].set_facecolor(colors['accent'])
    for box in bp['boxes']:
        box.set_edgecolor('#333333')
        box.set_linewidth(1.5)
    for median in bp['medians']:
        median.set_color('#333333')
        median.set_linewidth(2)
    
    ax1.set_ylabel('Balance (EUR)')
    ax1.set_title('Distribución de Balance Bancario', fontweight='bold', color=colors['primary'])
    
    for i, data in enumerate([df_no[balance_col], df_yes[balance_col]]):
        median = data.median()
        mean = data.mean()
        ax1.text(i + 1, ax1.get_ylim()[1] * 0.95, 
                f'Mediana: {median:,.0f}\nPromedio: {mean:,.0f}', 
                ha='center', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax2 = axes[1]
    metrics = ['Promedio', 'Mediana']
    no_vals = [df_no[balance_col].mean(), df_no[balance_col].median()]
    yes_vals = [df_yes[balance_col].mean(), df_yes[balance_col].median()]
    
    x = range(len(metrics))
    width = 0.35
    bars1 = ax2.bar([i - width/2 for i in x], no_vals, width, 
                    label='No Convirtió', color=colors['neutral'])
    bars2 = ax2.bar([i + width/2 for i in x], yes_vals, width, 
                    label='Convirtió', color=colors['accent'])
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.set_ylabel('Balance (EUR)')
    ax2.set_title('Comparativa de Balance', fontweight='bold', color=colors['primary'])
    ax2.legend()
    
    for bar in bars1 + bars2:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{bar.get_height():,.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    fig.savefig(output_dir / '05_balance_salario.png')
    plt.close(fig)
    
    # --- Gráfico 4C: Edad por Generación ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    gen_colors_local = {
        'Millennials': '#3498db',
        'Gen X': '#2ecc71',
        'Baby Boomers': '#f39c12',
        'Silent Generation': '#e74c3c'
    }
    gen_order = ['Millennials', 'Gen X', 'Baby Boomers', 'Silent Generation']
    
    age_ranges = generations.get_age_ranges()
    def classify_gen(age):
        if age >= age_ranges['Silent Generation'][0]:
            return 'Silent Generation'
        elif age >= age_ranges['Baby Boomers'][0]:
            return 'Baby Boomers'
        elif age >= age_ranges['Gen X'][0]:
            return 'Gen X'
        else:
            return 'Millennials'
    
    df_no_gen = df_no.copy()
    df_yes_gen = df_yes.copy()
    df_no_gen['generation'] = df_no_gen[age_col].apply(classify_gen)
    df_yes_gen['generation'] = df_yes_gen[age_col].apply(classify_gen)
    
    ax1 = axes[0]
    gen_data_no = [df_no_gen[df_no_gen['generation'] == g][age_col].values for g in gen_order]
    gen_data_yes = [df_yes_gen[df_yes_gen['generation'] == g][age_col].values for g in gen_order]
    
    positions_no = [i - 0.2 for i in range(len(gen_order))]
    positions_yes = [i + 0.2 for i in range(len(gen_order))]
    
    bp_no = ax1.boxplot(gen_data_no, positions=positions_no, widths=0.35, 
                        patch_artist=True, showfliers=False)
    bp_yes = ax1.boxplot(gen_data_yes, positions=positions_yes, widths=0.35, 
                         patch_artist=True, showfliers=False)
    
    for i, (box_no, box_yes) in enumerate(zip(bp_no['boxes'], bp_yes['boxes'])):
        box_no.set_facecolor(colors['neutral'])
        box_no.set_alpha(0.7)
        box_yes.set_facecolor(gen_colors_local[gen_order[i]])
        box_yes.set_alpha(0.9)
    
    for element in ['whiskers', 'caps', 'medians']:
        for item in bp_no[element]:
            item.set_color('#555555')
        for item in bp_yes[element]:
            item.set_color('#333333')
            item.set_linewidth(1.5)
    
    ax1.set_xticks(range(len(gen_order)))
    ax1.set_xticklabels(gen_order, rotation=15, ha='right')
    ax1.set_ylabel('Edad (años)')
    ax1.set_title('Distribución de Edad por Generación', fontweight='bold', color=colors['primary'])
    
    legend_elements = [Patch(facecolor=colors['neutral'], alpha=0.7, label='No Convirtió'),
                       Patch(facecolor=colors['accent'], label='Convirtió')]
    ax1.legend(handles=legend_elements, loc='upper left')
    
    ax2 = axes[1]
    all_df = pd.concat([df_no_gen, df_yes_gen])
    gen_stats_local = all_df.groupby('generation').agg(
        total=(target_col, 'count'),
        conversiones=(target_col, lambda x: (x == values.target_positive).sum())
    )
    gen_stats_local['tasa'] = (gen_stats_local['conversiones'] / gen_stats_local['total'] * 100).round(1)
    gen_stats_local['no_conv'] = gen_stats_local['total'] - gen_stats_local['conversiones']
    gen_stats_local = gen_stats_local.reindex(gen_order)
    
    x = range(len(gen_order))
    width = 0.6
    
    ax2.bar(x, gen_stats_local['no_conv'], width, 
            label='No Convirtió', color=colors['neutral'])
    ax2.bar(x, gen_stats_local['conversiones'], width, 
            bottom=gen_stats_local['no_conv'],
            label='Convirtió', color=colors['accent'])
    
    for i, (idx, row) in enumerate(gen_stats_local.iterrows()):
        ax2.text(i, row['total'] + 30, f'{row["tasa"]:.1f}%', 
                ha='center', va='bottom', fontsize=11, fontweight='bold',
                color=gen_colors_local[idx])
    
    avg_rate = (gen_stats_local['conversiones'].sum() / gen_stats_local['total'].sum()) * 100
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(gen_order, rotation=15, ha='right')
    ax2.set_ylabel('Número de Clientes')
    ax2.set_xlabel('Generación')
    ax2.set_title('Volumen y Tasa de Conversión por Generación', fontweight='bold', color=colors['primary'])
    ax2.legend(loc='upper right')
    
    age_ranges_text = ['19-31 años', '32-47 años', '48-66 años', '67+ años']
    for i, ar in enumerate(age_ranges_text):
        ax2.text(i, -max(gen_stats_local['total']) * 0.08, ar, 
                ha='center', va='top', fontsize=9, style='italic', color='#666666')
    
    ax2.set_ylim(-max(gen_stats_local['total']) * 0.12, max(gen_stats_local['total']) * 1.15)
    
    plt.tight_layout()
    fig.savefig(output_dir / '06_edad.png')
    plt.close(fig)
    
    # --- Gráfico 4D: Cantidad de Contactos ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    box_data = [df_no[campaign_col], df_yes[campaign_col]]
    bp = ax1.boxplot(box_data, tick_labels=['No Convirtió', 'Convirtió'], 
                     patch_artist=True, widths=0.6, showfliers=False)
    
    bp['boxes'][0].set_facecolor(colors['neutral'])
    bp['boxes'][1].set_facecolor(colors['accent'])
    for box in bp['boxes']:
        box.set_edgecolor('#333333')
        box.set_linewidth(1.5)
    for median in bp['medians']:
        median.set_color('#333333')
        median.set_linewidth(2)
    
    ax1.set_ylabel('Número de Contactos')
    ax1.set_title('Contactos Durante la Campaña', fontweight='bold', color=colors['primary'])
    
    for i, data in enumerate([df_no[campaign_col], df_yes[campaign_col]]):
        median = data.median()
        mean = data.mean()
        ax1.text(i + 1, ax1.get_ylim()[1] * 0.95, 
                f'Mediana: {median:.0f}\nPromedio: {mean:.1f}', 
                ha='center', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax2 = axes[1]
    
    def categorize_contacts(x):
        if x == 1:
            return '1 contacto'
        elif x == 2:
            return '2 contactos'
        elif x == 3:
            return '3 contactos'
        elif x <= 5:
            return '4-5 contactos'
        else:
            return '6+ contactos'
    
    contact_order = ['1 contacto', '2 contactos', '3 contactos', '4-5 contactos', '6+ contactos']
    
    df_no_copy = df_no.copy()
    df_yes_copy = df_yes.copy()
    df_no_copy['contact_group'] = df_no_copy[campaign_col].apply(categorize_contacts)
    df_yes_copy['contact_group'] = df_yes_copy[campaign_col].apply(categorize_contacts)
    
    no_contact_pct = df_no_copy['contact_group'].value_counts(normalize=True).reindex(contact_order) * 100
    yes_contact_pct = df_yes_copy['contact_group'].value_counts(normalize=True).reindex(contact_order) * 100
    
    x = range(len(contact_order))
    width = 0.35
    ax2.bar([i - width/2 for i in x], no_contact_pct.fillna(0), width, 
            label='No Convirtió', color=colors['neutral'])
    ax2.bar([i + width/2 for i in x], yes_contact_pct.fillna(0), width, 
            label='Convirtió', color=colors['accent'])
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(contact_order, rotation=15, ha='right')
    ax2.set_xlabel('Número de Contactos')
    ax2.set_ylabel('Porcentaje (%)')
    ax2.set_title('Distribución por Cantidad de Contactos', fontweight='bold', color=colors['primary'])
    ax2.legend()
    
    plt.tight_layout()
    fig.savefig(output_dir / '07_cantidad_contactos.png')
    plt.close(fig)
    
    # --- Gráfico 4F: Ocupación/Profesión ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    job_stats = df.groupby(job_col).agg(
        total=(target_col, 'count'),
        conversiones=(target_col, lambda x: (x == values.target_positive).sum())
    )
    job_stats['tasa_conversion'] = (job_stats['conversiones'] / job_stats['total'] * 100).round(2)
    job_stats = job_stats.sort_values('tasa_conversion', ascending=True)
    
    job_names = analysis_config.translations.get('job', {})
    
    ax1 = axes[0]
    y_pos = range(len(job_stats))
    colors_bars = [colors['accent'] if t >= job_stats['tasa_conversion'].median() 
                   else colors['secondary'] for t in job_stats['tasa_conversion']]
    
    bars = ax1.barh(y_pos, job_stats['tasa_conversion'], color=colors_bars, 
                    edgecolor='white', height=0.7)
    
    avg_rate = df[target_col].apply(lambda x: 1 if x == values.target_positive else 0).mean() * 100
    ax1.axvline(avg_rate, color=colors['warning'], linestyle='--', linewidth=2, 
                label=f'Promedio General: {avg_rate:.1f}%')
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([job_names.get(j, j.title()) for j in job_stats.index])
    ax1.set_xlabel('Tasa de Conversión (%)')
    ax1.set_title('Tasa de Conversión por Ocupación', fontweight='bold', color=colors['primary'])
    ax1.legend(loc='lower right')
    
    for bar, (idx, row) in zip(bars, job_stats.iterrows()):
        ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{row["tasa_conversion"]:.1f}%', va='center', fontsize=10, fontweight='bold')
    
    ax1.set_xlim(0, max(job_stats['tasa_conversion']) * 1.15)
    
    ax2 = axes[1]
    job_stats_vol = job_stats.sort_values('total', ascending=True)
    y_pos = range(len(job_stats_vol))
    
    ax2.barh(y_pos, job_stats_vol['total'] - job_stats_vol['conversiones'], 
             color=colors['neutral'], label='No Convirtió', height=0.7)
    ax2.barh(y_pos, job_stats_vol['conversiones'], 
             left=job_stats_vol['total'] - job_stats_vol['conversiones'],
             color=colors['accent'], label='Convirtió', height=0.7)
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([job_names.get(j, j.title()) for j in job_stats_vol.index])
    ax2.set_xlabel('Número de Clientes')
    ax2.set_title('Volumen por Ocupación', fontweight='bold', color=colors['primary'])
    ax2.legend(loc='lower right')
    
    for i, (idx, row) in enumerate(job_stats_vol.iterrows()):
        ax2.text(row['total'] + 10, i, f'{int(row["total"]):,}', 
                va='center', fontsize=9, color=colors['primary'])
    
    plt.tight_layout()
    fig.savefig(output_dir / '08_ocupacion.png')
    plt.close(fig)
    
    # --- Gráfico 4G: Ranking de Ocupaciones ---
    fig, ax = plt.subplots(figsize=(12, 7))
    
    job_stats_sorted = job_stats.sort_values('tasa_conversion', ascending=False)
    
    x = range(len(job_stats_sorted))
    sizes = (job_stats_sorted['total'] / job_stats_sorted['total'].max()) * 1500 + 100
    
    min_rate = job_stats_sorted['tasa_conversion'].min()
    max_rate = job_stats_sorted['tasa_conversion'].max()
    bubble_colors = [get_blue_gradient(r, min_rate, max_rate) for r in job_stats_sorted['tasa_conversion']]
    
    ax.scatter(x, job_stats_sorted['tasa_conversion'], s=sizes, 
               c=bubble_colors, alpha=0.7, edgecolors='white', linewidth=2)
    
    ax.axhline(avg_rate, color=colors['warning'], linestyle='--', linewidth=2, 
               label=f'Promedio: {avg_rate:.1f}%')
    
    ax.set_xticks(x)
    ax.set_xticklabels([job_names.get(j, j.title()) for j in job_stats_sorted.index], 
                       rotation=45, ha='right')
    ax.set_ylabel('Tasa de Conversión (%)')
    ax.set_title('Ranking de Ocupaciones por Tasa de Conversión\n(Tamaño de burbuja = volumen de clientes)', 
                 fontweight='bold', color=colors['primary'], fontsize=14)
    ax.legend(loc='upper right')
    
    for i, (idx, row) in enumerate(job_stats_sorted.iterrows()):
        label = f'{row["tasa_conversion"]:.1f}%\n({int(row["total"])})'
        ax.annotate(label, (i, row['tasa_conversion']), 
                   textcoords="offset points", xytext=(0, 15), 
                   ha='center', fontsize=8, fontweight='bold')
    
    ax.set_ylim(0, max(job_stats_sorted['tasa_conversion']) * 1.25)
    
    plt.tight_layout()
    fig.savefig(output_dir / '09_ocupacion_ranking.png')
    plt.close(fig)
    
    # --- Gráfico 4H: Análisis por Generación ---
    age_ranges = generations.get_age_ranges()
    
    def classify_generation(age):
        if age >= age_ranges['Silent Generation'][0]:
            return 'Silent Generation'
        elif age >= age_ranges['Baby Boomers'][0]:
            return 'Baby Boomers'
        elif age >= age_ranges['Gen X'][0]:
            return 'Gen X'
        else:
            return 'Millennials'
    
    df_copy = df.copy()
    df_copy['generation'] = df_copy[age_col].apply(classify_generation)
    
    gen_order = ['Millennials', 'Gen X', 'Baby Boomers', 'Silent Generation']
    gen_colors_map = {
        'Millennials': '#6ce5e8',
        'Gen X': '#41b8d5',
        'Baby Boomers': '#2d8bba',
        'Silent Generation': '#004aad'
    }
    
    gen_stats = df_copy.groupby('generation').agg(
        total=(target_col, 'count'),
        conversiones=(target_col, lambda x: (x == values.target_positive).sum()),
        edad_promedio=(age_col, 'mean')
    )
    gen_stats['tasa_conversion'] = (gen_stats['conversiones'] / gen_stats['total'] * 100).round(2)
    gen_stats['no_conversiones'] = gen_stats['total'] - gen_stats['conversiones']
    gen_stats = gen_stats.reindex([g for g in gen_order if g in gen_stats.index])
    
    fig = plt.figure(figsize=(16, 10))
    
    # Subplot 1: Pie chart
    ax1 = fig.add_subplot(2, 2, 1)
    pie_colors = [gen_colors_map.get(g, colors['neutral']) for g in gen_stats.index]
    wedges, texts, autotexts = ax1.pie(
        gen_stats['total'], 
        labels=gen_stats.index,
        colors=pie_colors,
        autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*gen_stats["total"].sum()):,})',
        startangle=90,
        explode=[0.02] * len(gen_stats),
        wedgeprops=dict(edgecolor='white', linewidth=2)
    )
    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_fontweight('bold')
    ax1.set_title('Distribución por Generación', fontweight='bold', 
                  fontsize=14, color=colors['primary'])
    
    # Subplot 2: Volumen
    ax2 = fig.add_subplot(2, 2, 2)
    x = range(len(gen_stats))
    bar_colors = [gen_colors_map.get(g, colors['neutral']) for g in gen_stats.index]
    
    bars = ax2.bar(x, gen_stats['total'], color=bar_colors, edgecolor='white', linewidth=1.5)
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(gen_stats.index, rotation=15, ha='right')
    ax2.set_ylabel('Número de Clientes')
    ax2.set_title('Volumen por Generación', fontweight='bold', 
                  fontsize=14, color=colors['primary'])
    
    for bar, (idx, row) in zip(bars, gen_stats.iterrows()):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f'{int(row["total"]):,}', ha='center', va='bottom', 
                fontsize=11, fontweight='bold')
    
    ax2.set_ylim(0, max(gen_stats['total']) * 1.15)
    
    # Subplot 3: Tasa de conversión
    ax3 = fig.add_subplot(2, 2, 3)
    bars = ax3.bar(x, gen_stats['tasa_conversion'], color=bar_colors, 
                   edgecolor='white', linewidth=1.5)
    
    avg_rate = df_copy[target_col].apply(lambda x: 1 if x == values.target_positive else 0).mean() * 100
    ax3.axhline(avg_rate, color=colors['warning'], linestyle='--', linewidth=2,
               label=f'Promedio General: {avg_rate:.1f}%')
    
    ax3.set_xticks(x)
    ax3.set_xticklabels(gen_stats.index, rotation=15, ha='right')
    ax3.set_ylabel('Tasa de Conversión (%)')
    ax3.set_title('Tasa de Conversión por Generación', fontweight='bold', 
                  fontsize=14, color=colors['primary'])
    ax3.legend(loc='upper left')
    
    for bar, (idx, row) in zip(bars, gen_stats.iterrows()):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{row["tasa_conversion"]:.1f}%', ha='center', va='bottom', 
                fontsize=11, fontweight='bold')
    
    ax3.set_ylim(0, max(gen_stats['tasa_conversion']) * 1.2)
    
    # Subplot 4: Tabla resumen
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    
    table_data = []
    for idx, row in gen_stats.iterrows():
        table_data.append([
            idx,
            f'{int(row["total"]):,}',
            f'{int(row["conversiones"]):,}',
            f'{row["tasa_conversion"]:.1f}%',
            f'{row["edad_promedio"]:.0f} años'
        ])
    
    table = ax4.table(
        cellText=table_data,
        colLabels=['Generación', 'Total', 'Conversiones', 'Tasa Conv.', 'Edad Prom.'],
        cellLoc='center',
        loc='center',
        colWidths=[0.25, 0.15, 0.18, 0.17, 0.18]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    for i in range(5):
        table[(0, i)].set_facecolor(colors['primary'])
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    for i, (idx, row) in enumerate(gen_stats.iterrows(), start=1):
        for j in range(5):
            table[(i, j)].set_facecolor(gen_colors_map.get(idx, 'white'))
            table[(i, j)].set_alpha(0.3)
    
    ax4.set_title('Resumen por Generación', fontweight='bold', 
                  fontsize=14, color=colors['primary'], pad=20)
    
    plt.suptitle('Análisis de Clientes por Generación', 
                 fontsize=18, fontweight='bold', color=colors['primary'], y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    fig.savefig(output_dir / '10_generaciones.png')
    plt.close(fig)
    
    # --- Gráfico 4I: Préstamos (Hipoteca y Personal) ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    avg_rate = (df[target_col] == values.target_positive).mean() * 100
    
    # Housing (Hipoteca)
    ax1 = axes[0]
    housing_stats = df.groupby(housing_col).agg(
        total=(target_col, 'count'),
        conversiones=(target_col, lambda x: (x == values.target_positive).sum())
    )
    housing_stats['tasa'] = (housing_stats['conversiones'] / housing_stats['total'] * 100).round(2)
    housing_stats['no_conv'] = housing_stats['total'] - housing_stats['conversiones']
    housing_order = [values.no_loan, values.has_loan]
    housing_stats = housing_stats.reindex([h for h in housing_order if h in housing_stats.index])
    
    labels_h = ['Sin Hipoteca', 'Con Hipoteca']
    x_h = range(len(housing_stats))
    
    ax1.bar(x_h, housing_stats['no_conv'], label='No Convirtió', 
            color=colors['neutral'], edgecolor='white', linewidth=1.5)
    ax1.bar(x_h, housing_stats['conversiones'], 
            bottom=housing_stats['no_conv'],
            label='Convirtió', color=colors['accent'], edgecolor='white', linewidth=1.5)
    
    max_total = max(housing_stats['total'])
    for i, (idx, row) in enumerate(housing_stats.iterrows()):
        ax1.text(i, row['total'] + max_total * 0.05, f'{row["tasa"]:.1f}%', 
                ha='center', va='bottom', fontsize=13, fontweight='bold',
                color=colors['accent'] if row['tasa'] > avg_rate else colors['warning'],
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                         edgecolor=colors['accent'] if row['tasa'] > avg_rate else colors['warning'],
                         alpha=0.9, linewidth=2))
        ax1.text(i, row['total'] + max_total * 0.12, f'n={int(row["total"]):,}', 
                ha='center', va='bottom', fontsize=10, style='italic', color='#666666')
    
    ax1.set_xticks(x_h)
    ax1.set_xticklabels(labels_h, fontsize=11)
    ax1.set_ylabel('Número de Clientes', fontsize=11)
    ax1.set_title('Préstamo Hipotecario (Housing)', fontweight='bold', 
                  fontsize=13, color=colors['primary'], pad=10)
    ax1.legend(loc='upper left', fontsize=10, framealpha=0.95, 
               edgecolor='#cccccc', frameon=True)
    ax1.set_ylim(0, max(housing_stats['total']) * 1.35)
    
    # Loan (Préstamo Personal)
    ax2 = axes[1]
    loan_stats = df.groupby(loan_col).agg(
        total=(target_col, 'count'),
        conversiones=(target_col, lambda x: (x == values.target_positive).sum())
    )
    loan_stats['tasa'] = (loan_stats['conversiones'] / loan_stats['total'] * 100).round(2)
    loan_stats['no_conv'] = loan_stats['total'] - loan_stats['conversiones']
    loan_order = [values.no_loan, values.has_loan]
    loan_stats = loan_stats.reindex([l for l in loan_order if l in loan_stats.index])
    
    labels_l = ['Sin Préstamo', 'Con Préstamo']
    x_l = range(len(loan_stats))
    
    ax2.bar(x_l, loan_stats['no_conv'], label='No Convirtió', 
            color=colors['neutral'], edgecolor='white', linewidth=1.5)
    ax2.bar(x_l, loan_stats['conversiones'], 
            bottom=loan_stats['no_conv'],
            label='Convirtió', color=colors['accent'], edgecolor='white', linewidth=1.5)
    
    max_total_loan = max(loan_stats['total'])
    for i, (idx, row) in enumerate(loan_stats.iterrows()):
        ax2.text(i, row['total'] + max_total_loan * 0.05, f'{row["tasa"]:.1f}%', 
                ha='center', va='bottom', fontsize=13, fontweight='bold',
                color=colors['accent'] if row['tasa'] > avg_rate else colors['warning'],
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                         edgecolor=colors['accent'] if row['tasa'] > avg_rate else colors['warning'],
                         alpha=0.9, linewidth=2))
        ax2.text(i, row['total'] + max_total_loan * 0.12, f'n={int(row["total"]):,}', 
                ha='center', va='bottom', fontsize=10, style='italic', color='#666666')
    
    ax2.set_xticks(x_l)
    ax2.set_xticklabels(labels_l, fontsize=11)
    ax2.set_ylabel('Número de Clientes', fontsize=11)
    ax2.set_title('Préstamo Personal (Loan)', fontweight='bold', 
                  fontsize=13, color=colors['primary'], pad=10)
    ax2.legend(loc='upper left', fontsize=10, framealpha=0.95, 
               edgecolor='#cccccc', frameon=True)
    ax2.set_ylim(0, max(loan_stats['total']) * 1.35)
    
    # Comparativa
    ax3 = axes[2]
    comparativa = pd.DataFrame({
        'Categoria': ['Sin Hipoteca', 'Con Hipoteca', 'Sin Préstamo', 'Con Préstamo'],
        'Tasa': [
            housing_stats.loc[values.no_loan, 'tasa'] if values.no_loan in housing_stats.index else 0,
            housing_stats.loc[values.has_loan, 'tasa'] if values.has_loan in housing_stats.index else 0,
            loan_stats.loc[values.no_loan, 'tasa'] if values.no_loan in loan_stats.index else 0,
            loan_stats.loc[values.has_loan, 'tasa'] if values.has_loan in loan_stats.index else 0
        ],
        'Tipo': ['Hipoteca', 'Hipoteca', 'Préstamo', 'Préstamo']
    })
    
    bar_colors_comp = ['#6ce5e8', '#41b8d5', '#2d8bba', '#004aad']
    bars = ax3.barh(range(len(comparativa)), comparativa['Tasa'], color=bar_colors_comp,
                    edgecolor='white', height=0.6)
    
    ax3.axvline(avg_rate, color=colors['warning'], linestyle='--', linewidth=2,
               label=f'Promedio: {avg_rate:.1f}%')
    
    ax3.set_yticks(range(len(comparativa)))
    ax3.set_yticklabels(comparativa['Categoria'])
    ax3.set_xlabel('Tasa de Conversión (%)')
    ax3.set_title('Comparativa de Tasas', fontweight='bold', color=colors['primary'])
    ax3.legend(loc='lower right')
    
    for bar, val in zip(bars, comparativa['Tasa']):
        ax3.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=10, fontweight='bold')
    
    ax3.set_xlim(0, max(comparativa['Tasa']) * 1.2)
    
    plt.suptitle('Análisis de Préstamos (Hipoteca y Personal)', 
                 fontsize=16, fontweight='bold', color=colors['primary'])
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    fig.savefig(output_dir / '11_prestamos.png')
    plt.close(fig)
    
    # --- Gráfico 4J: Día del Mes de Contacto ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    day_stats = df.groupby(day_col).agg(
        total=(target_col, 'count'),
        conversiones=(target_col, lambda x: (x == values.target_positive).sum())
    )
    day_stats['tasa'] = (day_stats['conversiones'] / day_stats['total'] * 100).round(2)
    
    ax1 = axes[0]
    ax1.bar(day_stats.index, day_stats['total'], color=colors['secondary'], 
            edgecolor='white', alpha=0.8)
    ax1.set_xlabel('Día del Mes')
    ax1.set_ylabel('Número de Contactos')
    ax1.set_title('Volumen de Contactos por Día del Mes', fontweight='bold', color=colors['primary'])
    ax1.set_xticks(range(1, 32, 2))
    
    ax2 = axes[1]
    min_tasa = day_stats['tasa'].min()
    max_tasa = day_stats['tasa'].max()
    day_colors = [get_blue_gradient(t, min_tasa, max_tasa) for t in day_stats['tasa']]
    
    bars = ax2.bar(day_stats.index, day_stats['tasa'], color=day_colors, edgecolor='white')
    
    ax2.axhline(avg_rate, color=colors['warning'], linestyle='--', linewidth=2,
               label=f'Promedio: {avg_rate:.1f}%')
    
    ax2.set_xlabel('Día del Mes')
    ax2.set_ylabel('Tasa de Conversión (%)')
    ax2.set_title('Tasa de Conversión por Día del Mes', fontweight='bold', color=colors['primary'])
    ax2.set_xticks(range(1, 32, 2))
    ax2.legend(loc='upper right')
    
    best_day = day_stats['tasa'].idxmax()
    ax2.annotate(f'Mejor: Día {best_day}\n({day_stats.loc[best_day, "tasa"]:.1f}%)', 
                xy=(best_day, day_stats.loc[best_day, 'tasa']),
                xytext=(best_day + 3, day_stats.loc[best_day, 'tasa'] + 3),
                fontsize=9, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=colors['accent']))
    
    plt.tight_layout()
    fig.savefig(output_dir / '12_dia_contacto.png')
    plt.close(fig)
    
    # Retornar estadísticas resumen
    return {
        'duracion': {
            'no_convertido': {'mediana': df_no[duration_col].median(), 'promedio': df_no[duration_col].mean()},
            'convertido': {'mediana': df_yes[duration_col].median(), 'promedio': df_yes[duration_col].mean()}
        },
        'balance': {
            'no_convertido': {'mediana': df_no[balance_col].median(), 'promedio': df_no[balance_col].mean()},
            'convertido': {'mediana': df_yes[balance_col].median(), 'promedio': df_yes[balance_col].mean()}
        },
        'edad': {
            'no_convertido': {'mediana': df_no[age_col].median(), 'promedio': df_no[age_col].mean()},
            'convertido': {'mediana': df_yes[age_col].median(), 'promedio': df_yes[age_col].mean()}
        },
        'contactos': {
            'no_convertido': {'mediana': df_no[campaign_col].median(), 'promedio': df_no[campaign_col].mean()},
            'convertido': {'mediana': df_yes[campaign_col].median(), 'promedio': df_yes[campaign_col].mean()}
        }
    }


def generate_summary_table(df: pd.DataFrame, colors: Dict, output_dir: Path,
                           columns: Optional[ColumnMapping] = None,
                           values: Optional[ValueMapping] = None):
    """Genera una tabla resumen ejecutiva como imagen."""
    if columns is None:
        columns = DEFAULT_COLUMNS
    if values is None:
        values = DEFAULT_VALUES
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    
    total_clientes = len(df)
    target_col = columns.target
    conversiones = (df[target_col] == values.target_positive).sum()
    tasa_general = conversiones / total_clientes * 100 if total_clientes > 0 else 0
    
    contact_col = columns.contact
    canal_stats = df.groupby(contact_col)[target_col].apply(
        lambda x: (x == values.target_positive).mean() * 100
    )
    mejor_canal = canal_stats.idxmax() if len(canal_stats) > 0 else "N/A"
    tasa_mejor_canal = canal_stats.max() if len(canal_stats) > 0 else 0
    
    month_col = columns.month
    mes_stats = df.groupby(month_col)[target_col].apply(
        lambda x: (x == values.target_positive).mean() * 100
    )
    mejor_mes = mes_stats.idxmax() if len(mes_stats) > 0 else "N/A"
    tasa_mejor_mes = mes_stats.max() if len(mes_stats) > 0 else 0
    
    data = [
        ['Total de Clientes Contactados', f'{total_clientes:,}'],
        ['Total de Conversiones', f'{conversiones:,}'],
        ['Tasa de Conversión General', f'{tasa_general:.2f}%'],
        ['', ''],
        ['Canal Más Efectivo', f'{mejor_canal.title()} ({tasa_mejor_canal:.2f}%)'],
        ['Mes con Mayor Conversión', f'{mejor_mes.title()} ({tasa_mejor_mes:.2f}%)'],
    ]
    
    table = ax.table(
        cellText=data,
        colLabels=['Métrica', 'Valor'],
        cellLoc='left',
        loc='center',
        colWidths=[0.5, 0.3]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)
    
    for i in range(2):
        table[(0, i)].set_facecolor(colors['primary'])
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    for i in range(1, len(data) + 1):
        for j in range(2):
            if i == 4:
                table[(i, j)].set_facecolor(colors['background'])
                table[(i, j)].set_height(0.02)
            else:
                table[(i, j)].set_facecolor('white' if i % 2 == 0 else colors['background'])
    
    ax.set_title('Resumen Ejecutivo - Análisis de Conversión', 
                 fontsize=16, fontweight='bold', pad=20, color=colors['primary'])
    
    plt.tight_layout()
    fig.savefig(output_dir / '00_resumen_ejecutivo.png')
    plt.close(fig)


# ============================================================================
# Función Principal
# ============================================================================

def basic_target_analysis(
    df: pd.DataFrame, 
    show_plots: bool = False,
    columns: Optional[ColumnMapping] = None,
    values: Optional[ValueMapping] = None,
    generations: Optional[GenerationConfig] = None,
    analysis_config: Optional[AnalysisConfig] = None,
    validate: bool = True
) -> Dict:
    """
    Ejecuta el análisis completo de la variable objetivo.
    
    Acepta configuraciones para ser aplicable a nuevos datasets.
    Si no se proporcionan, usa las configuraciones por defecto.
    """
    if columns is None:
        columns = DEFAULT_COLUMNS
    if values is None:
        values = DEFAULT_VALUES
    if generations is None:
        generations = DEFAULT_GENERATIONS
    if analysis_config is None:
        analysis_config = DEFAULT_ANALYSIS
    
    if validate:
        try:
            is_valid, issues = validate_data(df, columns, values, strict=True)
            if not is_valid:
                raise DataValidationError(f"Validación fallida: {issues}")
        except DataValidationError:
            raise
        except Exception as e:
            print(f"[!] Advertencia: Error en validación: {e}")
            print("[!] Continuando sin validación estricta...")
    
    print("\n" + "="*60)
    print("  ANÁLISIS DE CONVERSIÓN - CAMPAÑA DE MARKETING BANCARIO")
    print("="*60)
    
    colors = setup_style()
    output_dir = create_output_dir(analysis_config)
    
    print(f"\n[*] Guardando visualizaciones en: {output_dir}")
    
    target_col = get_column_safe(df, 'target', columns)
    
    print("\n[*] Generando gráficos...")
    
    print("   [1/14] Resumen ejecutivo...")
    generate_summary_table(df, colors, output_dir, columns, values)
    
    print("   [2/14] Tasa general de conversión...")
    tasa_general = plot_conversion_rate(df, colors, output_dir, columns, values)
    
    print("   [3/14] Análisis por canal de contacto...")
    canal_stats = plot_conversion_by_contact(df, colors, output_dir, columns, values)
    
    print("   [4/14] Análisis temporal por mes...")
    mes_stats = plot_conversion_by_month(df, colors, output_dir, columns, values, analysis_config)
    
    print("   [5/14] Duración de contactos...")
    print("   [6/14] Duración sin outliers (P95)...")
    print("   [7/14] Balance/Salario...")
    print("   [8/14] Edad por Generación...")
    print("   [9/14] Cantidad de contactos...")
    print("   [10/14] Ocupación - Tasa de conversión...")
    print("   [11/14] Ocupación - Ranking...")
    print("   [12/14] Análisis por Generación...")
    print("   [13/14] Préstamos (Hipoteca y Personal)...")
    print("   [14/14] Día de contacto...")
    demo_stats = plot_demographic_analysis(
        df, colors, output_dir, 
        columns=columns, 
        values=values, 
        generations=generations,
        analysis_config=analysis_config
    )
    
    print("\n" + "-"*60)
    print("  RESUMEN DE RESULTADOS")
    print("-"*60)
    print(f"\n  > Tasa de conversión general: {tasa_general:.2f}%")
    print(f"  > Total de clientes analizados: {len(df):,}")
    print(f"  > Conversiones totales: {(df[target_col] == values.target_positive).sum():,}")
    
    print("\n  Archivos generados:")
    for f in sorted(output_dir.glob("*.png")):
        print(f"     - {f.name}")
    
    print("\n" + "="*60)
    print("  [OK] Análisis completado exitosamente")
    print("="*60 + "\n")
    
    return {
        'tasa_conversion_general': tasa_general,
        'total_clientes': len(df),
        'total_conversiones': (df[target_col] == values.target_positive).sum(),
        'estadisticas_canal': canal_stats,
        'estadisticas_mes': mes_stats,
        'estadisticas_demograficas': demo_stats,
        'ruta_figuras': output_dir
    }


if __name__ == "__main__":
    from pathlib import Path
    data_path = Path(__file__).parent.parent / "data" / "processed" / "bank_clean.csv"
    df = pd.read_csv(data_path)
    results = basic_target_analysis(df)
